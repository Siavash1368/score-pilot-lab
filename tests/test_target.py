"""Tests for scorelab.target on a small synthetic ``accepted`` table (no real database needed)."""

from __future__ import annotations

import math
from datetime import date

import duckdb
import pandas as pd
import pytest

from scorelab import config, target

SNAPSHOT = date.fromisoformat(config.PERFORMANCE_SNAPSHOT)
W = config.PERFORMANCE_WINDOW_MONTHS


def _add_months(d: date, m: int) -> date:
    y, mo = divmod(d.month - 1 + m, 12)
    return date(d.year + y, mo + 1, 1)


# One row per case: (id, issue_d, term, loan_status, months_to_last_payment or None, fico_low, fico_high)
ISSUE_OBS = date(2015, 6, 1)          # observable at 24 months (2017-06 <= snapshot)
ISSUE_LATE = date(2018, 1, 1)         # not observable at 24 months, observable at 12
CASES = [
    (1, ISSUE_OBS, " 36 months", "Charged Off", None, 660, 664),   # never paid -> bad
    (2, ISSUE_OBS, " 36 months", "Charged Off", 10, 700, 704),     # 90+ at month 14 -> bad
    (3, ISSUE_OBS, " 60 months", "Charged Off", 20, 700, 704),     # 90+ at month 24 -> bad (boundary)
    (4, ISSUE_OBS, " 60 months", "Charged Off", 21, 700, 704),     # first miss 22, 90+ at 25 -> indeterminate
    (5, ISSUE_OBS, " 60 months", "Charged Off", 23, 700, 704),     # first miss 24, 90+ at 27 -> indeterminate
    (6, ISSUE_OBS, " 60 months", "Charged Off", 30, 700, 704),     # delinquent after window -> good
    (7, ISSUE_OBS, " 36 months", "Fully Paid", 36, 720, 724),      # good
    (8, ISSUE_OBS, " 60 months", "Current", 45, 720, 724),         # good
    (9, ISSUE_OBS, " 60 months", "Late (31-120 days)", 43, 700, 704),  # late after window -> good
    (10, ISSUE_LATE, " 36 months", "Charged Off", 5, 680, 684),    # unobservable at 24; 60+ at month 8 -> early_bad
    (11, ISSUE_LATE, " 36 months", "Current", 13, 680, 684),       # unobservable; early good
    (12, ISSUE_OBS, " 36 months", "Does not meet the credit policy. Status:Charged Off", 3, 660, 664),
    (13, ISSUE_OBS, " 36 months", "Fully Paid", 36, 720, 724),      # joint application -> excluded
]
JOINT_IDS = {13}


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE accepted (
            id BIGINT, issue_d DATE, term VARCHAR, loan_status VARCHAR, last_pymnt_d DATE,
            fico_range_low DOUBLE, fico_range_high DOUBLE, revol_util DOUBLE, application_type VARCHAR
        )
    """)
    rows = []
    for i, issue, term, status, m_last, lo, hi in CASES:
        last = _add_months(issue, m_last) if m_last is not None else None
        rows.append((i, issue, term, status, last, lo, hi, 50.0, "Joint App" if i in JOINT_IDS else "Individual"))
    c.executemany("INSERT INTO accepted VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    c.execute("CREATE VIEW v_accepted_asof AS SELECT id, issue_d, loan_status, fico_range_low, "
              "fico_range_high, revol_util FROM accepted")
    target.create_views(c)
    return c


def labels(con) -> dict[int, dict]:
    df = con.sql("SELECT * FROM v_split ORDER BY id").df().set_index("id")
    return df.to_dict(orient="index")


def test_snapshot_matches_data_assumption():
    assert SNAPSHOT >= date(2019, 3, 1)
    assert _add_months(ISSUE_OBS, W) <= SNAPSHOT
    assert _add_months(ISSUE_LATE, W) > SNAPSHOT


def test_bad_labels(con):
    lab = labels(con)
    for i in (1, 2, 3):
        assert lab[i]["label"] == "bad" and lab[i]["bad"] == 1 and lab[i]["indeterminate"] == 0, i


def test_indeterminate_labels(con):
    lab = labels(con)
    for i in (4, 5):
        assert lab[i]["label"] == "indeterminate", i
        assert lab[i]["bad"] == 0 and lab[i]["indeterminate"] == 1, i


def test_good_labels_including_delinquency_after_window(con):
    lab = labels(con)
    for i in (6, 7, 8, 9):
        assert lab[i]["label"] == "good" and lab[i]["bad"] == 0, i


def test_unobservable_rows_have_null_label_but_early_indicator(con):
    lab = labels(con)
    for i in (10, 11):
        assert lab[i]["observable"] == 0
        assert pd.isna(lab[i]["label"]) and pd.isna(lab[i]["bad"]), i
        assert lab[i]["early_observable"] == 1
    assert lab[10]["early_bad"] == 1
    assert lab[11]["early_bad"] == 0


def test_credit_policy_rows_are_flagged(con):
    lab = labels(con)
    assert lab[12]["exclusion"] == "credit_policy"
    assert lab[13]["exclusion"] == "joint_application"
    assert all(pd.isna(lab[i]["exclusion"]) for i in range(1, 12))


def test_term_months_parsed(con):
    lab = labels(con)
    assert lab[1]["term_months"] == 36 and lab[3]["term_months"] == 60


def test_split_assignment(con):
    lab = labels(con)
    assert lab[1]["split"] == "validate"      # 2015 vintage
    assert lab[10]["split"] == "monitor"      # 2018Q1


def test_modeling_frame_drops_indeterminates_and_adds_incumbent(con):
    df = target.modeling_frame(con, splits=("validate",), columns=["id", "revol_util"])
    assert set(df["label"]) == {"bad", "good"}
    assert 4 not in set(df["id"]) and 5 not in set(df["id"])
    assert 12 not in set(df["id"])                       # excluded
    assert "fico_mid" in df.columns and df.loc[df.id == 1, "fico_mid"].iloc[0] == 662.0
    df2 = target.modeling_frame(con, splits=("validate",), include_indeterminate=True,
                                columns=["id"])
    assert 4 in set(df2["id"]) and pd.isna(df2.loc[df2.id == 4, "bad"].iloc[0])


def test_label_summary_counts(con):
    s = target.label_summary(con, by="split").set_index("split")
    assert s.loc["validate", "n_bad"] == 3
    assert s.loc["validate", "n_indeterminate"] == 2
    assert s.loc["validate", "n_good"] == 4
    assert s.loc["validate", "bad_rate_excl_indet"] == pytest.approx(3 / 7, abs=1e-4)


def test_prior_correction_roundtrip():
    # Fitting on a 50/50 sample when the population bad rate is 10% lowers the intercept by ln(9).
    assert target.prior_correct_intercept(0.0, 0.5, 0.1) == pytest.approx(-math.log(9))
    with pytest.raises(ValueError):
        target.prior_correct_intercept(0.0, 0.0, 0.1)


def test_sample_predicate_is_deterministic_nested_and_seeded():
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("CREATE TABLE s AS SELECT range AS id FROM range(200000)")
    con.execute("CREATE TABLE a AS SELECT range AS id, range % 7 AS x FROM range(200000)")

    def draw(frac, seed=42):
        q = f"SELECT s.id FROM s JOIN a USING (id) WHERE a.x >= 0 {target.sample_predicate(frac, seed, 's.id')}"
        return set(con.execute(q).df().id)

    first = draw(0.05)
    assert all(draw(0.05) == first for _ in range(3))                   # same rows every run, after a join
    assert len(first) / 200000 == pytest.approx(0.05, abs=0.003)
    assert first <= draw(0.10)                                          # smaller sample nested in larger
    assert len(first & draw(0.05, seed=7)) < 0.2 * len(first)           # another seed, another draw
    assert target.sample_predicate(None) == ""
    with pytest.raises(ValueError):
        target.sample_predicate(1.5)


def test_exclusion_waterfall_counts_each_step():
    con = duckdb.connect()
    con.execute("""CREATE TABLE v_split AS SELECT * FROM (VALUES
        ('train', NULL, 1, 'bad'), ('train', NULL, 1, 'good'), ('train', NULL, 1, 'indeterminate'),
        ('train', 'credit_policy', 1, 'good'), ('validate', NULL, 1, 'good'), ('oot', NULL, 1, 'bad'),
        ('oot', 'joint_application', 1, 'good'),
        ('monitor', NULL, 0, NULL), (NULL, NULL, 1, 'good')) t(split, exclusion, observable, label)""")
    w = target.exclusion_waterfall(con)
    assert list(w["total"]) == [9, 7, 6, 5, 5, 4]
    assert list(w["removed"].iloc[1:]) == [2, 1, 1, 0, 1]
    assert w.loc["minus indeterminates (labelled sample)", "train"] == 2
    assert w["bads_remaining"].iloc[-1] == 2 and w["bads_remaining"].iloc[:-1].isna().all()
