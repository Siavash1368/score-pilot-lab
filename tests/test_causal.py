"""Double machine learning recovers a known effect where the naive comparison is confounded (NB07e)."""

import numpy as np
import pandas as pd
import pytest

from scorelab import causal as CA


@pytest.fixture(scope="module")
def confounded():
    rng = np.random.default_rng(11)
    n = 12000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    e = 1 / (1 + np.exp(-(0.9 * X.x1 - 0.5 * X.x2)))                     # risky applicants pick treatment
    d = (rng.uniform(size=n) < e).astype(int)
    base = 1 / (1 + np.exp(-(-2.2 + 0.9 * X.x1 + 0.4 * np.sin(2 * X.x2))))
    y = (rng.uniform(size=n) < np.clip(base + 0.04 * d, 0, 1)).astype(int)  # true risk difference 0.04
    return y, d, X


def test_plr_recovers_effect_naive_does_not(confounded):
    y, d, X = confounded
    r = CA.dml_plr(y, d, X, n_folds=3)
    assert r["ci"][0] < 0.04 < r["ci"][1] and r["se"] > 0 and 0.6 < r["propensity_auc"] < 0.9
    naive = CA.naive_and_adjusted(y, d, X).set_index("estimator").effect
    assert naive["raw difference in bad rate"] > 0.08                      # confounded upward


def test_benchmark_handles_duplicate_missing_flags_without_warning(confounded):
    import warnings
    y, d, X = confounded
    Xd = X.assign(x3=X.x1 * 2.0, x4=np.nan, x5=np.nan)                 # dependent column, identical missing flags
    Xd.loc[: len(Xd) // 3, ["x4", "x5"]] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = CA.naive_and_adjusted(y, d, Xd).set_index("estimator").effect
    assert 0.0 < out["logistic regression, average marginal effect"] < out["raw difference in bad rate"]


def test_irm_ate_and_att(confounded):
    y, d, X = confounded
    r = CA.dml_irm(y, d, X, n_folds=3)
    assert r["ate_ci"][0] < 0.04 < r["ate_ci"][1]
    assert r["att_ci"][0] < 0.04 < r["att_ci"][1] and r["n_treated"] == int(np.sum(d))
    ov = CA.overlap_table(r["propensity"], d)
    assert ov.n.sum() == len(y) and (ov.treated + ov.control == ov.n).all()


def test_control_matrix_one_hot():
    df = pd.DataFrame({"a": [1.0, np.nan, 2.0], "purpose": ["car", "debt", "home"]})
    X = CA.control_matrix(df, ["a"], ["purpose"])
    assert list(X.columns) == ["a", "purpose_debt", "purpose_home"] and np.isnan(X.loc[1, "a"])   # first level dropped
    assert "not the effect of a policy" in CA.CAVEAT
