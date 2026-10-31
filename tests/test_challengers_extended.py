"""Random forest, EBM, MLP, stacking, Optuna tuning and the comparison table (NB07b)."""

import numpy as np
import pandas as pd
import pytest

from scorelab import challengers as C, tuning as T, validation as V

ATTRS = ["util", "age", "inq"]


@pytest.fixture(scope="module")
def parts(synth):
    return tuple(synth[synth.split == s] for s in ("train", "validate", "oot"))


def test_random_forest_ranks_and_predict_p(parts):
    tr, va, oot = parts
    rf = C.fit_random_forest(tr, tr["bad"], ATTRS, params={"n_estimators": 40})
    p = C.predict_p(rf, oot)
    assert p.shape == (len(oot),) and 0 <= p.min() and p.max() <= 1
    assert V.auc_gini_ks(oot["bad"], p, higher_is_safer=False)["gini"] > 0.2


def test_ebm_monotone_main_effects_only(parts):
    tr, va, oot = parts
    ebm = C.fit_ebm(tr, tr["bad"], ATTRS, monotone=True, params={"outer_bags": 1, "max_rounds": 300})
    assert len(ebm.term_names_) == len(ATTRS)                     # no pairwise terms when monotone
    assert V.auc_gini_ks(oot["bad"], C.predict_p(ebm, oot), higher_is_safer=False)["gini"] > 0.2


def test_mlp_early_stops_on_validation_and_keeps_best_epoch(parts):
    tr, va, oot = parts
    # 12k rows is ~24 optimiser steps per epoch, so a higher learning rate than the 400k-row default.
    m = C.fit_mlp(tr, tr["bad"], va, va["bad"], ATTRS, hidden=(8,), learning_rate=1e-2, max_epochs=25, patience=3)
    h = m.history
    assert m.best_epoch == int(h.loc[h.valid_log_loss.idxmin(), "epoch"])
    assert "age" in m.scaler.flagged                                # age has missing values -> flag column
    assert V.auc_gini_ks(oot["bad"], C.predict_p(m, oot), higher_is_safer=False)["gini"] > 0.2


def test_standardizer_median_impute_flags_and_train_only_fit(parts):
    tr, va, _ = parts
    s = C.Standardizer.fit(tr, ATTRS)
    assert s.median["age"] == pytest.approx(tr["age"].median()) and s.flagged == ["age"]
    X = s.transform(va)
    assert X.shape == (len(va), len(ATTRS) + 1) and not np.isnan(X).any()
    miss = va["age"].isna().to_numpy()
    assert np.allclose(X[miss, 1], (s.median["age"] - s.mean["age"]) / s.std["age"])   # median, not zero or mean
    assert np.array_equal(X[:, -1], miss.astype(float))
    shifted = va.assign(util=va["util"] + 1000)                       # statistics come from train only
    assert s.transform(shifted)[:, 0].mean() > 20
    with pytest.warns(UserWarning, match="had none in training"):
        s.transform(va.assign(inq=np.where(np.arange(len(va)) < 5, np.nan, va["inq"])))


def test_stack_weights_and_prediction(parts):
    tr, va, oot = parts
    xgb = C.fit_xgb(tr, tr["bad"], va, va["bad"], ATTRS, params={"eta": 0.2}, n_rounds=200, early_stopping=20)
    rf = C.fit_random_forest(tr, tr["bad"], ATTRS, params={"n_estimators": 40})
    base_va = {"xgb": C.predict_p(xgb, va), "rf": C.predict_p(rf, va)}
    stack = C.fit_stack(base_va, va["bad"])
    assert set(stack.weights) == {"xgb", "rf"}
    p = stack.predict({"xgb": C.predict_p(xgb, oot), "rf": C.predict_p(rf, oot)})
    assert p.shape == (len(oot),) and np.all((p > 0) & (p < 1))


def test_xgb_save_load_round_trip(parts, tmp_path):
    tr, va, oot = parts
    b = C.fit_xgb(tr, tr["bad"], va, va["bad"], ATTRS, n_rounds=100, early_stopping=10)
    C.save_xgb(b, tmp_path / "m.json")
    b2 = C.load_xgb(tmp_path / "m.json")
    assert b2.attributes_used == ATTRS and b2.best_iteration == b.best_iteration
    assert np.allclose(C.predict_p(b, oot), C.predict_p(b2, oot))


def test_optuna_tuning_runs_on_validation_only(parts):
    tr, va, _ = parts
    best, trials = T.tune_xgb(tr, va, ATTRS, n_trials=3)
    assert {"max_depth", "eta", "n_rounds"} <= set(best) and len(trials) == 3
    assert trials.valid_auc.is_monotonic_decreasing
    best_mlp, trials_mlp = T.tune_mlp(tr, va, ATTRS, n_trials=2, max_epochs=3)
    assert isinstance(best_mlp["hidden"], tuple) and len(trials_mlp) == 2


def test_model_comparison_table(parts, synth):
    _, _, oot = parts
    rng = np.random.default_rng(1)
    good = 1 / (1 + np.exp(-(-2.6 + 0.012 * oot["util"] + 0.25 * oot["inq"])))
    noise = np.clip(good + rng.normal(0, 0.2, len(oot)), 0.01, 0.99)
    tab = V.model_comparison(oot["bad"], {"good": good, "noisy": noise}, reference="good",
                             fit_seconds={"good": 1.0}, notes={"good": "additive"}, inputs={"good": "raw, 2"})
    assert list(tab.model) == ["good", "noisy"] and list(tab.columns[:2]) == ["model", "inputs"]
    assert tab.loc[0, "inputs"] == "raw, 2" and tab.loc[1, "inputs"] == ""
    assert tab.loc[0, "gini"] > tab.loc[1, "gini"] and tab.loc[1, "gini_vs_ref"] < 0
    assert tab.loc[1, "gini_vs_ref_ci_low"] < tab.loc[1, "gini_vs_ref"] < tab.loc[1, "gini_vs_ref_ci_high"]
    assert tab.loc[1, "ci_excludes_zero"] == (tab.loc[1, "gini_vs_ref_ci_high"] < 0) and tab.loc[0, "ci_excludes_zero"] is None
    assert tab.loc[0, "gini_ci_low"] < tab.loc[0, "gini"] < tab.loc[0, "gini_ci_high"]
    assert abs(tab.loc[0, "calib_slope"] - 1) < 0.3 and tab.loc[0, "explainability"] == "additive"
    assert np.isnan(tab.loc[1, "fit_seconds"])
