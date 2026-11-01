"""Monitoring rule (NB09): each period judged against its own Gini interval and against the Gini the model
expects on that period's population (computed exactly from its PDs); actions mapped to evidence."""

import numpy as np
import pandas as pd
import pytest

from scorelab import monitoring as M, validation as V


def decide(**kw):
    base = dict(psi_value=0.01, gini_dev=0.30, gini_now=0.29, gini_now_ci=(0.28, 0.30), calib_slope=1.0, max_csi=0.05,
                calib_intercept_shift=0.0)
    return M.decide_action(**{**base, **kw})


def act(**kw):
    return decide(**kw)["action"]


DROP = dict(gini_now=0.25, gini_now_ci=(0.24, 0.26))          # reference 0.30 above the period's own interval


def test_actions_follow_the_evidence():
    assert act() == "no action"                                                        # inside the period's own CI
    assert act(calib_intercept_shift=0.25) == "recalibrate"                            # level off, ranking at reference
    assert act(calib_slope=0.8) == "recalibrate"
    assert act(**DROP, expected_gini_ci=(0.27, 0.31)) == "re-estimate"                 # gap 0.04 > half width 0.02
    assert act(**DROP, expected_gini_ci=(0.27, 0.31), max_csi=0.4) == "redevelop"
    assert act(**DROP, expected_gini_ci=(0.23, 0.27)) == "investigate population (no model change)"   # at expected
    assert act(**DROP, expected_gini_ci=(0.23, 0.27), calib_intercept_shift=0.3) == "investigate population (no model change); recalibrate"
    assert act(max_csi=0.4) == "investigate population (no model change)"              # population moved, model holds


def test_watch_band_is_half_the_interval_width():
    # expected 0.26, half width 0.02: a gap up to 0.02 is watch, a larger gap is decay
    assert act(**DROP, expected_gini_ci=(0.24, 0.28)) == "watch (no model action)"            # gap 0.01
    assert act(**DROP, expected_gini_ci=(0.249, 0.289)) == "watch (no model action)"          # gap 0.019, just inside
    assert act(**DROP, expected_gini_ci=(0.252, 0.292)) == "re-estimate"                      # gap 0.022
    assert act(**DROP, expected_gini_ci=(0.24, 0.28), calib_slope=0.8) == "watch (no model action); recalibrate"
    d = decide(**DROP, expected_gini_ci=(0.24, 0.28))
    assert d["versus_expected"] == "watch" and d["gap_to_expected"] == pytest.approx(0.01) and d["half_width"] == pytest.approx(0.02)


BREACH = dict(**DROP, expected_gini_ci=(0.27, 0.31))          # gap 0.04 beyond half width 0.02
OK = dict(gini_now=0.29, gini_now_ci=(0.28, 0.30))             # at the reference


def test_persistence_first_breach_is_watch_second_triggers_action():
    assert act(**BREACH, prior_breach=False) == "watch (first breach, no model action)"
    assert act(**BREACH, prior_breach=False, calib_slope=0.8) == "watch (first breach, no model action); recalibrate"
    assert act(**BREACH, prior_breach=True) == "re-estimate"
    assert act(**BREACH, prior_breach=True, max_csi=0.4) == "redevelop"
    assert act(**BREACH) == "re-estimate"                       # a single period (scenario): as if the breach persisted


def test_decide_periods_carries_breaches_forward():
    base = dict(psi_value=0.01, gini_dev=0.30, calib_slope=1.0, max_csi=0.05, calib_intercept_shift=0.0)
    seq = [dict(base, **OK, expected_gini_ci=(0.27, 0.31)), dict(base, **BREACH), dict(base, **BREACH), dict(base, **BREACH),
           dict(base, gini_now=float("nan"), gini_now_ci=None, expected_gini_ci=None),      # outcome not observable: resets
           dict(base, **BREACH), dict(base, **OK, expected_gini_ci=(0.27, 0.31)), dict(base, **BREACH, max_csi=0.4)]
    actions = [d["action"] for d in M.decide_periods(seq)]
    assert actions == ["no action", "watch (first breach, no model action)", "re-estimate", "re-estimate",
                       "distribution checks only (24m not observable)", "watch (first breach, no model action)",
                       "no action", "watch (first breach, no model action)"]


def test_without_expected_gini_every_drop_counts_as_decay():
    d = decide(**DROP)
    assert d["action"] == "re-estimate" and d["versus_expected"] == "below expected" and "expected_gini" not in d


def test_same_drop_is_judged_by_the_periods_own_interval():
    assert act(gini_now=0.29, gini_now_ci=(0.285, 0.295)) == "re-estimate"            # tight CI: significant
    assert act(gini_now=0.29, gini_now_ci=(0.27, 0.31)) == "no action"                 # wide CI: not significant


def brute_force_gini(p, score):
    """Expected concordant pairs / expected pairs, over all ordered pairs i != j (higher score = safer)."""
    num = den = 0.0
    for i in range(len(p)):
        for j in range(len(p)):
            if i == j:
                continue
            w = p[i] * (1 - p[j]); den += w
            num += w * (1.0 if score[i] < score[j] else 0.5 if score[i] == score[j] else 0.0)
    return 2 * num / den - 1


def test_expected_gini_equals_pairwise_definition_with_ties():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.5, 60); score = np.round(-np.log(p / (1 - p)) * 3 + rng.normal(0, 1, 60))   # rounded: ties
    assert len(np.unique(score)) < 60
    assert M.expected_gini(p, score)["expected_gini"] == pytest.approx(brute_force_gini(p, score), abs=1e-12)


def test_expected_gini_is_deterministic_and_interval_uses_se():
    rng = np.random.default_rng(6); p = rng.uniform(0.02, 0.4, 5000); s = -p
    a, b = M.expected_gini(p, s, gini_se=0.01), M.expected_gini(p, s, gini_se=0.01)
    assert a == b
    assert a["expected_ci"] == pytest.approx((a["expected_gini"] - 0.0196, a["expected_gini"] + 0.0196))


@pytest.fixture(scope="module")
def truth():
    rng = np.random.default_rng(3); n = 30000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = -2.2 + 0.6 * x1 + 0.4 * x2; p = 1 / (1 + np.exp(-eta))
    return pd.DataFrame({"x1": x1, "x2": x2, "eta": eta, "p": p, "score": -eta, "bad": (rng.uniform(size=n) < p).astype(int)})


def test_expected_gini_matches_the_average_over_simulated_outcomes(truth):
    rng = np.random.default_rng(9); p, s = truth["p"].to_numpy(), truth["score"].to_numpy()
    sims = [V.auc_gini_ks((rng.uniform(size=len(p)) < p).astype(int), s)["gini"] for _ in range(100)]
    assert M.expected_gini(p, s)["expected_gini"] == pytest.approx(np.mean(sims), abs=0.003)


def classify(frame, reference):
    obs = V.delong_ci(frame["bad"].to_numpy(dtype=int), frame["score"])
    x = M.expected_gini(frame["p"], frame["score"], gini_se=obs["gini_se"])
    return decide(gini_dev=reference, gini_now=obs["gini"], gini_now_ci=obs["gini_ci"], expected_gini_ci=x["expected_ci"], max_csi=0.5)


def test_expected_gini_separates_decay_from_population_shift(truth):
    ref = V.auc_gini_ks(truth["bad"], truth["score"])["gini"]
    # Ordering decay: 30% of outcomes shuffled -> far below expected
    assert classify(M.simulate_drift(truth, "ordering", strength=0.3), ref)["versus_expected"] == "below expected"
    # True covariate shift: the population narrows, Gini falls below the reference, but the model is still right
    c = classify(M.simulate_drift(truth, "covariate", attributes=["x1"], strength=1.0), ref)
    assert c["below_reference"] and c["versus_expected"] != "below expected" and c["action"] != "redevelop"


def test_covariate_shift_moves_x_but_keeps_y_given_x(synth):
    oot = synth[synth.split != "oot"].reset_index(drop=True)
    d = M.simulate_drift(oot, "covariate", attributes=["util"], strength=0.5)
    assert len(d) == len(oot) // 2 and not d.duplicated().any()                    # no duplicated rows
    assert M.realised_shift(oot, d, ["util"])["util"] > 0.1                          # P(X) moved
    # P(Y|X) unchanged: bad rate within a util band is the same before and after
    band = lambda f: f.loc[f.util.between(60, 80), "bad"].mean()
    assert band(d) == pytest.approx(band(oot), abs=0.02)
