import numpy as np
import pandas as pd
import pytest

from scorelab import validation as V, fairness as F


def test_auc_gini_ks_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([600, 590, 580, 500, 510, 520])   # higher = safer
    m = V.auc_gini_ks(y, s)
    assert m["auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 5000); s = rng.normal(size=5000)
    assert abs(V.auc_gini_ks(y, s)["gini"]) < 0.06


def test_delong_matches_sklearn_and_ci_covers(synth):
    oot = synth[synth.split == "oot"]
    d = V.delong_ci(oot["bad"], oot["fico_mid"])
    assert d["auc"] == pytest.approx(V.auc_gini_ks(oot["bad"], oot["fico_mid"])["auc"], abs=1e-9)
    assert d["gini_ci"][0] < d["gini"] < d["gini_ci"][1] and d["gini_se"] > 0


def test_delong_paired_test_same_score_is_zero(synth):
    oot = synth[synth.split == "oot"]
    t = V.delong_test(oot["bad"], oot["fico_mid"], oot["fico_mid"])
    assert t["gini_diff"] == pytest.approx(0.0, abs=1e-12)


def test_gains_table_shape_and_monotone_capture(synth):
    oot = synth[synth.split == "oot"]
    g = V.gains_table(oot["bad"], oot["fico_mid"], n_bands=10)
    assert len(g) == 10 and g.cum_share_bads.iloc[-1] == pytest.approx(1.0)
    assert g.cum_share_bads.is_monotonic_increasing and g.bad_rate.iloc[0] > g.bad_rate.iloc[-1]


def test_swap_set_identity_and_counts(synth):
    oot = synth[synth.split == "oot"]
    same = V.swap_set(oot["bad"], oot["fico_mid"], oot["fico_mid"], 0.7)
    assert same["n_swap_in"] == 0 and same["n_swap_out"] == 0
    other = V.swap_set(oot["bad"], oot["fico_mid"], -oot["util"], 0.7)
    assert other["n_swap_in"] == other["n_swap_out"] > 0                      # matched approval rate
    tbl = other["crosstab"]; assert tbl.values.sum() == len(oot)


def test_swap_set_matches_approval_rate_with_tied_scores(synth):
    oot = synth[synth.split == "oot"]
    coarse = (oot["fico_mid"] // 20) * 20                                       # heavy ties at any cutoff
    s = V.swap_set(oot["bad"], coarse, -oot["util"], 0.7)
    k = int(round(0.7 * len(oot))); tbl = s["crosstab"]
    assert tbl.loc["incumbent_approve"].sum() == k and tbl["challenger_approve"].sum() == k
    assert s["n_swap_in"] == s["n_swap_out"]
    assert V.swap_set(oot["bad"], coarse, coarse, 0.7)["n_swap_in"] == 0          # same score, same decisions


def test_calibration_slope_recovers_truth_and_shift():
    rng = np.random.default_rng(1); eta = rng.normal(-2, 1, 40000); y = (rng.uniform(size=40000) < 1 / (1 + np.exp(-eta))).astype(int)
    c = V.calibration_slope_intercept(y, eta)
    assert c["slope"] == pytest.approx(1.0, abs=0.05) and c["intercept"] == pytest.approx(0.0, abs=0.08)
    c2 = V.calibration_slope_intercept(y, eta - 0.5)
    assert c2["intercept_only_shift"] == pytest.approx(0.5, abs=0.08)


def test_calibration_bands_and_probability_scores(synth):
    rng = np.random.default_rng(2); p = rng.uniform(0.02, 0.4, 20000); y = (rng.uniform(size=20000) < p).astype(int)
    cb = V.calibration_by_band(y, p, n_bands=5)
    assert len(cb) == 5 and cb.within_interval.mean() >= 0.6
    ps = V.probability_scores(y, p)
    assert ps["brier"] == pytest.approx(ps["brier_check"], abs=2e-3) and ps["log_loss"] < ps["null_log_loss"]


def test_psi_zero_same_and_flags_shift():
    rng = np.random.default_rng(3); dev = rng.normal(600, 40, 50000)
    r = V.psi(dev, dev); assert r["psi"] == pytest.approx(0.0, abs=1e-9) and r["verdict"] == "stable"
    r2 = V.psi(dev, dev + 40); assert r2["psi"] > 0.25 and r2["verdict"] == "action" and r2["p_value"] < 1e-6


def test_fairness_air_and_smd():
    rng = np.random.default_rng(4); n = 10000
    g = np.where(rng.uniform(size=n) < 0.3, "b", "a"); s = rng.normal(600, 40, n) - np.where(g == "b", 15, 0)
    appr = F.approval_flags(s, 0.7)
    air = F.adverse_impact_ratio(appr, g).set_index("group")
    assert air.loc["a", "reference"] and air.loc["b", "air"] < 1.0
    smd = F.standardized_mean_difference(s, g).set_index("group")
    assert smd.loc["b", "smd_vs_reference"] < 0


def test_recalibration_fixes_level_and_keeps_rank(synth):
    va, oot = synth[synth.split == "validate"], synth[synth.split == "oot"]
    eta = -2.6 + 0.012 * synth["util"] + 0.25 * synth["inq"]
    p_true = 1 / (1 + np.exp(-eta))
    p_off = 1 / (1 + np.exp(-(0.5 * eta - 1.0)))                     # wrong level and spread, same ranking
    cal = V.fit_recalibration(va["bad"], p_off.loc[va.index])
    p_new = V.apply_recalibration(p_off.loc[oot.index], cal)
    assert cal["b"] > 1.5                                              # undoes the halved spread
    assert V.auc_gini_ks(oot["bad"], p_new, higher_is_safer=False)["gini"] == pytest.approx(
        V.auc_gini_ks(oot["bad"], p_off.loc[oot.index], higher_is_safer=False)["gini"], abs=1e-12)
    c = V.calibration_slope_intercept(oot["bad"], np.log(p_new / (1 - p_new)))
    assert abs(c["slope"] - 1) < 0.25 and abs(c["intercept_only_shift"]) < 0.15
    tab = V.model_comparison(oot["bad"], {"m": p_new})
    assert {"calib_slope", "calib_intercept", "calib_in_the_large"} <= set(tab.columns)
