"""Partial dependence, interaction strength, crossed attributes and reason-code agreement (NB07c)."""

import numpy as np
import pandas as pd
import pytest

from scorelab import challengers as C, explain as E


@pytest.fixture(scope="module")
def interaction_data():
    """Bad depends on x1 and x2 through a pure interaction, and on x3 additively."""
    rng = np.random.default_rng(3)
    n = 20000
    df = pd.DataFrame({"x1": rng.uniform(-1, 1, n), "x2": rng.uniform(-1, 1, n), "x3": rng.uniform(-1, 1, n)})
    eta = -2.0 + 2.5 * (df.x1 > 0) * (df.x2 > 0) + 1.0 * df.x3
    df["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    return df.iloc[:14000], df.iloc[14000:]


@pytest.fixture(scope="module")
def booster(interaction_data):
    tr, va = interaction_data
    return C.fit_xgb(tr, tr["bad"], va, va["bad"], ["x1", "x2", "x3"], monotone=False,
                     params={"max_depth": 3, "eta": 0.2}, n_rounds=300, early_stopping=20)


def test_partial_dependence_follows_additive_effect(booster, interaction_data):
    _, va = interaction_data
    pdt, ice = E.partial_dependence(booster, va, "x3", n_grid=8, max_rows=500, ice=5)
    assert pdt.mean_logit.iloc[-1] > pdt.mean_logit.iloc[0] + 0.8      # true slope 1.0 over ~1.9 units
    assert ice.shape == (len(pdt), 5) and not pdt.is_missing.any()


def test_friedman_h_separates_interaction_from_additive(booster, interaction_data):
    _, va = interaction_data
    h_inter = E.friedman_h(booster, va, "x1", "x2", n_rows=120)
    h_add = E.friedman_h(booster, va, "x1", "x3", n_rows=120)
    assert h_inter > 0.15 and h_add < 0.05 and h_inter > 4 * h_add


def test_shap_interaction_pairs_ranks_true_pair_first(booster, interaction_data):
    _, va = interaction_data
    pairs = E.shap_interaction_pairs(booster, va, max_rows=2000)
    assert {pairs.loc[0, "a"], pairs.loc[0, "b"]} == {"x1", "x2"}
    assert pairs.share_of_interaction.sum() == pytest.approx(1.0, abs=1e-3)


def test_cross_attribute_levels_and_threshold(booster, interaction_data):
    tr, _ = interaction_data
    t1 = E.top_threshold(booster, "x1")
    assert abs(t1) < 0.15                                               # the GBM finds the cut at 0
    df = tr.copy(); df.loc[df.index[:10], "x2"] = np.nan
    cross = E.cross_attribute(df, "x1", "x2", t1, 0.0)
    assert cross.nunique() == 6 and cross.name == "x1_x_x2"            # 2 sides x (2 sides + missing)
    hi = cross.str.contains("x1>") & cross.str.contains("x2>")
    assert df.loc[hi, "bad"].mean() > df.loc[~hi, "bad"].mean() + 0.2


def test_reason_code_agreement():
    card = pd.DataFrame({"reason_1": ["A [x]", "A [x]", "C [y]"], "reason_2": ["B [y]", "B [z]", ""],
                         "reason_3": ["", "", ""], "reason_4": ["", "", ""]})
    gbm = pd.DataFrame({"reason_1": ["A", "B", "D"], "reason_2": ["B", "A", ""], "reason_3": ["", "", ""],
                        "reason_4": ["", "", ""]})
    s, rows = E.reason_code_agreement(card, gbm)
    assert list(rows.overlap) == [2, 2, 0]
    assert s["share_same_first_reason"] == pytest.approx(1 / 3)
    assert s["share_no_reason_in_common"] == pytest.approx(1 / 3)
    assert s["mean_rank_gap_shared"] == pytest.approx(0.5)              # row 1: 0, row 2: 1
    assert E.first_reason_crosstab(card, gbm).to_numpy().sum() == 3
