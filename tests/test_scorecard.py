import numpy as np
import pandas as pd
import pytest

from scorelab import binning as B, scorecard as S, config


@pytest.fixture(scope="module")
def fitted(synth):
    tr = synth[synth.split == "train"]
    specs = B.fit_all(tr, ["util", "age", "inq"], directions={"util": 1, "age": -1, "inq": 1}, min_bads=30)
    W = B.transform_woe(tr, specs)
    card = S.fit_logistic(W, tr["bad"], ["util", "age", "inq"])
    return tr, specs, W, card


def test_coefficients_negative_and_significant(fitted):
    tr, specs, W, card = fitted
    assert all(card.coef[a] < 0 for a in card.attributes)
    assert all(card.p_values[a] < 0.01 for a in card.attributes)
    assert 0 < card.pseudo_r2 < 1


def test_moment_matching_in_sample(fitted):
    tr, specs, W, card = fitted
    p = S.score(tr, card, specs)["p_bad"]
    assert p.mean() == pytest.approx(tr["bad"].mean(), abs=1e-6)       # intercept column
    for a in card.attributes:                                            # each WOE column
        x = W[a + S.WOE_SUFFIX]
        assert (x * tr["bad"]).sum() == pytest.approx((x * p).sum(), rel=1e-4)


def test_points_sum_to_score_and_scaling(fitted):
    tr, specs, W, card = fitted
    pts = S.points_table(card, specs)
    sc = S.score(tr.head(200), card, specs, clip=False)
    total = np.zeros(200)
    for a in card.attributes:
        idx = B.bin_index(tr.head(200)[a], specs[a])
        lookup = pts[pts.attribute == a].set_index("bin")["points"]
        total += lookup.loc[idx].to_numpy()
    assert np.allclose(total, sc["score"].to_numpy(), atol=1.0)          # points are rounded to 0.1
    # at BASE_ODDS the score equals BASE_SCORE; PDO points double the odds
    eta_base = -np.log(config.BASE_ODDS)
    assert card.offset - card.factor * eta_base == pytest.approx(config.BASE_SCORE)
    assert card.factor * np.log(2) == pytest.approx(config.PDO)


def test_reason_codes_are_largest_point_losses(fitted):
    tr, specs, W, card = fitted
    rc = S.reason_codes(tr.head(20), card, specs, n_codes=2)
    assert rc.shape == (20, 4)
    assert (rc["reason_1_points_lost"] >= rc["reason_2_points_lost"]).all()


def test_stepwise_rejects_wrong_sign_and_correlated(fitted):
    tr, specs, W, card = fitted
    W2 = W.copy(); W2["util_copy_woe"] = W2["util_woe"]; specs2 = dict(specs)
    sel, log = S.stepwise_select(W2, tr["bad"], ["util", "util_copy", "age", "inq"], max_corr=0.7)
    assert "util" in sel and "util_copy" not in sel
    assert (log.loc[log.attribute == "util_copy", "decision"] == "rejected").all()


def test_save_load_roundtrip(fitted, tmp_path):
    tr, specs, W, card = fitted
    S.save_card(card, specs, tmp_path / "card")
    c2, s2 = S.load_card(tmp_path / "card")
    assert c2.attributes == card.attributes and c2.intercept == pytest.approx(card.intercept)
    assert (S.score(tr.head(50), c2, s2)["score"].values == S.score(tr.head(50), card, specs)["score"].values).all()


def test_missing_reason_replaces_attribute_reason_and_is_counted_once(fitted):
    tr, specs, W, card = fitted
    # Make "age" missing for every row in the age missing bin, then map it, and util, to one text.
    rows = tr.head(400).copy()
    miss_rows = rows["age"].isna()
    assert miss_rows.any()
    plain = S.reason_codes(rows, card, specs)
    mapped = S.reason_codes(rows, card, specs, missing_reasons={"age": "No age on file"})
    cols = [f"reason_{i + 1}" for i in range(4)]
    texts = mapped.loc[miss_rows, cols].to_numpy().ravel()
    assert not any(str(t).startswith("age [") for t in texts)        # never the attribute's own reason when missing
    gave_age_reason = plain.loc[miss_rows, cols].apply(lambda r: any(str(t).startswith("age [") for t in r), axis=1)
    assert (mapped.loc[miss_rows & gave_age_reason, cols].apply(lambda r: "No age on file" in set(r), axis=1)).all()
    both = S.reason_codes(rows, card, specs, missing_reasons={"age": "Shared", "util": "Shared"})
    assert (both[cols].apply(lambda r: list(r).count("Shared"), axis=1) <= 1).all()   # one code, not two
