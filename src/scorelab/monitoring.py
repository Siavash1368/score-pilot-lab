"""Monitoring (NB09): score and attribute stability, early-performance tracking of censored
vintages, calibration drift, drift simulation, and the recalibrate / re-estimate / redevelop rule
(each period judged against its own confidence interval and against the Gini the model expects on that population).

The monitor vintages (2017Q1-2018Q4) are censored at 24 months, so performance is tracked with
the 12-month early indicator (``early_bad``) and the distribution with PSI/CSI; the 2016 OOT year
serves as the reference for both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scorelab import config
from scorelab.binning import BinSpec, csi
from scorelab.validation import psi, auc_gini_ks, delong_ci, calibration_slope_intercept


# --- Stability over time --------------------------------------------------------------------

def score_psi_by_period(dev_scores, frame: pd.DataFrame, score_col: str = "score", period_col: str = "vintage_quarter",
                        bins: int = 10) -> pd.DataFrame:
    """PSI of each period's score distribution against the development scores, with the chi-square
    critical value at the period's sample size (so small periods are not flagged by noise)."""
    rows = []
    edges = psi(dev_scores, dev_scores, bins)["edges"]
    for per, d in frame.groupby(period_col):
        r = psi(dev_scores, d[score_col], edges)
        rows.append({period_col: per, "n": len(d), "psi": round(r["psi"], 4), "critical_5pct": round(r["critical_psi_5pct"], 4),
                     "significant": r["psi"] > r["critical_psi_5pct"], "verdict": r["verdict"]})
    return pd.DataFrame(rows)


def csi_by_period(dev_specs: dict[str, BinSpec], frame: pd.DataFrame, period_col: str = "vintage_quarter") -> pd.DataFrame:
    rows = []
    for per, d in frame.groupby(period_col):
        for a, s in dev_specs.items():
            rows.append({period_col: per, "attribute": a, "csi": round(csi(s, d[a])[0], 4)})
    return pd.DataFrame(rows).pivot(index="attribute", columns=period_col, values="csi")


def attribute_drift_ranking(csi_table: pd.DataFrame) -> pd.DataFrame:
    """Worst CSI per attribute across periods, sorted: the first thing to read when score PSI trips."""
    out = pd.DataFrame({"max_csi": csi_table.max(axis=1), "last_csi": csi_table.iloc[:, -1]})
    out["verdict"] = np.where(out.max_csi > 0.25, "action", np.where(out.max_csi > 0.10, "watch", "stable"))
    return out.sort_values("max_csi", ascending=False)


# --- Performance tracking ------------------------------------------------------------------

def expected_gini(p, score, higher_is_safer: bool = True, gini_se: float | None = None, z: float = 1.96) -> dict:
    """The Gini ``score`` would have if the PDs ``p`` were the true bad probabilities, computed exactly.

    With outcomes y_i ~ Bernoulli(p_i), a pair (i, j) is a bad-good pair with probability p_i (1 - p_j); it is
    concordant when i is riskier than j, and a tie counts half. The expected AUC is expected concordant pairs over
    expected pairs, computed in O(n log n) by grouping equal scores; Gini = 2 AUC - 1. It is deterministic.

    Given ``gini_se`` (the DeLong standard error of the observed Gini on the same loans), the interval is
    expected +/- ``z`` * ``gini_se``: the range the observed Gini should fall in if the PDs were true. The expected
    value is fixed once the loans and PDs are known; the noise in the comparison is which outcomes happened, which
    the observed Gini's standard error measures, so no bootstrap is needed.

    Expected Gini depends only on the population and the model's own PDs: a population that is less spread on
    the attributes that drive the score has a lower expected Gini even when the model is right."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0); s = np.asarray(score, dtype=float)
    risk = -s if higher_is_safer else s
    _, grp = np.unique(risk, return_inverse=True)                  # groups of equal score, in ascending risk
    pb = np.bincount(grp, weights=p); pg = np.bincount(grp, weights=1 - p); pp = np.bincount(grp, weights=p * (1 - p))
    good_below = np.cumsum(pg) - pg                                 # expected goods strictly safer than each group
    concordant = float(np.sum(pb * good_below) + 0.5 * np.sum(pb * pg - pp))
    pairs = float(p.sum() * (1 - p).sum() - np.sum(p * (1 - p)))
    g = 2 * concordant / pairs - 1
    out = {"expected_gini": g}
    if gini_se is not None:
        out["expected_ci"] = (g - z * gini_se, g + z * gini_se)
    return out


def early_performance_by_period(frame: pd.DataFrame, score_col: str = "score", period_col: str = "vintage_quarter",
                                early_col: str = "early_bad", bad_col: str = "bad", eta_col: str | None = "eta") -> pd.DataFrame:
    """Per period: early (12-month) bad rate and Gini, plus the 24-month figures where observable. With
    ``eta_col`` (the model's log-odds of bad), the 24-month Gini is shown next to its expected Gini. The Gini
    columns are left unrounded because the decision rule compares them; round them for display."""
    rows = []
    for per, d in frame.groupby(period_col):
        row = {period_col: per, "n": len(d)}
        e = d[d[early_col].notna()]
        if len(e) and e[early_col].nunique() == 2:
            row.update({"n_early_observable": len(e), "early_bad_rate": round(e[early_col].mean(), 4),
                        "early_gini": round(auc_gini_ks(e[early_col], e[score_col])["gini"], 4)})
        b = d[d[bad_col].notna()]
        if len(b) and b[bad_col].nunique() == 2:
            g = delong_ci(b[bad_col].to_numpy(dtype=int), b[score_col])
            row.update({"n_24m_observable": len(b), "bad_rate_24m": round(b[bad_col].mean(), 4),
                        "gini_24m": g["gini"], "gini_24m_ci_low": g["gini_ci"][0], "gini_24m_ci_high": g["gini_ci"][1]})
            if eta_col and eta_col in b:
                x = expected_gini(1 / (1 + np.exp(-b[eta_col].to_numpy(dtype=float))), b[score_col], gini_se=g["gini_se"])
                row.update({"expected_gini": x["expected_gini"], "expected_ci_low": x["expected_ci"][0],
                            "expected_ci_high": x["expected_ci"][1]})
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_by_period(frame: pd.DataFrame, eta_col: str = "eta", period_col: str = "vintage_quarter",
                          bad_col: str = "bad") -> pd.DataFrame:
    rows = []
    for per, d in frame.groupby(period_col):
        b = d[d[bad_col].notna()]
        if len(b) < 500 or b[bad_col].nunique() < 2: continue
        c = calibration_slope_intercept(b[bad_col], b[eta_col])
        rows.append({period_col: per, "n": len(b), "mean_predicted": round(c["mean_predicted"], 4),
                     "mean_observed": round(c["mean_observed"], 4), "intercept_shift": round(c["intercept_only_shift"], 3),
                     "slope": round(c["slope"], 3), "slope_ci_low": round(c["slope_ci"][0], 3), "slope_ci_high": round(c["slope_ci"][1], 3)})
    return pd.DataFrame(rows)


# --- Drift simulation ------------------------------------------------------------------------

def simulate_drift(frame: pd.DataFrame, kind: str, attributes: list[str] | None = None, strength: float = 1.0,
                   seed: int = 42, risk_col: str | None = None) -> pd.DataFrame:
    """Return a drifted copy of ``frame`` for the monitoring exercise.

    kind = 'covariate' : draw half the loans, without replacement, with weights exp(``strength`` * sum of
                         the listed attributes' z-scores). Each loan keeps its own outcome, so P(X) moves
                         and P(Y|X) does not. (Adding a constant to the recorded attributes would instead
                         break P(Y|X).) No row is duplicated, so intervals computed on the result are
                         honest; report the realised shift with :func:`realised_shift`.
           'concept'   : keep X, raise the bad rate by flipping goods to bad with probability
                         ``strength`` * base rate, scaled by the model's own risk when ``risk_col``
                         (a log-odds column) is given so the ordering survives and only the level
                         drifts (P(Y|X) moves).
           'ordering'  : keep X and the bad rate, but add noise to the outcome-score relation by
                         shuffling a ``strength`` share of outcomes (discrimination decays).
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    if kind == "covariate":
        z = np.zeros(len(out))
        for a in attributes or []:
            v = out[a].to_numpy(dtype=float)
            z += np.nan_to_num((v - np.nanmean(v)) / np.nanstd(v))
        w = np.exp(strength * (z - z.max()))
        out = out.iloc[np.sort(rng.choice(len(out), size=len(out) // 2, replace=False, p=w / w.sum()))].reset_index(drop=True)
    elif kind == "concept":
        base = out["bad"].mean(skipna=True)
        goods = out.index[out["bad"] == 0]
        prob = np.full(len(goods), strength * base)
        if risk_col is not None:
            w = 1 / (1 + np.exp(-out.loc[goods, risk_col].to_numpy(dtype=float)))
            prob = prob * w / w.mean()
        flip = rng.uniform(size=len(goods)) < np.clip(prob, 0, 1)
        out.loc[goods[flip], "bad"] = 1
    elif kind == "ordering":
        idx = out.index[out["bad"].notna()]
        k = int(strength * len(idx))
        pick = rng.choice(idx, size=k, replace=False)
        out.loc[pick, "bad"] = rng.permutation(out.loc[pick, "bad"].to_numpy())
    else:
        raise ValueError(kind)
    return out


def realised_shift(before: pd.DataFrame, after: pd.DataFrame, attributes: list[str]) -> dict[str, float]:
    """Mean shift of each attribute, in standard deviations of ``before``."""
    return {a: round(float((after[a].mean() - before[a].mean()) / before[a].std()), 3) for a in attributes}


# --- Decision rule ----------------------------------------------------------------------------

def decide_action(psi_value: float, gini_dev: float, gini_now: float, gini_now_ci: tuple[float, float],
                  calib_slope: float, max_csi: float, slope_band: tuple[float, float] = (0.9, 1.1),
                  calib_intercept_shift: float = 0.0, max_intercept_shift: float = 0.15,
                  expected_gini_ci: tuple[float, float] | None = None, prior_breach: bool | None = None) -> dict:
    """Map the evidence of one period to an action.

    * below reference: the reference Gini lies above the period's own DeLong 95% interval (``gini_now_ci``);
    * gap to expected: the expected Gini (centre of ``expected_gini_ci``, from :func:`expected_gini`) minus the
      observed Gini, judged against half the interval's width:
        - gap <= 0: at or above expected; the model ranks as well as it says it should;
        - 0 < gap <= half width: watch; below expected, but within the noise;
        - gap > half width: below expected; the model ranks worse than it should.
      Without ``expected_gini_ci``, every drop below the reference is treated as below expected;
    * calibration off: slope outside ``slope_band`` or intercept-only shift beyond ``max_intercept_shift``.

    A breach is a period below the reference and below expected by more than half the interval width.
    Persistence: the first breach -> watch (first breach, no model action); a breach in the period right after
    another breach -> re-estimate (redevelop if a card attribute's CSI > 0.25). ``prior_breach`` says whether the
    previous period breached; None means a single period judged on its own (e.g. a simulated scenario), treated
    as if the breach persisted.

    Actions, first match wins, for a period below the reference: persistent breach -> re-estimate or redevelop;
    first breach -> watch (first breach, no model action); within the watch band -> watch (no model action); at
    or above expected -> population shift, investigate population (no model change). Each watch or population
    outcome adds "recalibrate" when calibration is off. For a period at the reference: calibration off ->
    recalibrate; PSI or max CSI > 0.25 -> investigate population; otherwise no action.
    """
    gini_drop = gini_dev - gini_now
    below_reference = gini_dev > gini_now_ci[1]
    if expected_gini_ci is None:
        expected, half, gap, versus = float("nan"), float("nan"), float("nan"), "below expected"
    else:
        expected = (expected_gini_ci[0] + expected_gini_ci[1]) / 2; half = (expected_gini_ci[1] - expected_gini_ci[0]) / 2
        gap = expected - gini_now
        versus = "at or above expected" if gap <= 0 else ("watch" if gap <= half else "below expected")
    calibration_off = (not (slope_band[0] <= calib_slope <= slope_band[1])) or abs(calib_intercept_shift) > max_intercept_shift
    population_moved = psi_value > config.PSI_THRESHOLDS["action"] or max_csi > config.PSI_THRESHOLDS["action"]
    recal = "; recalibrate" if calibration_off else ""
    breach = below_reference and versus == "below expected"
    persistent = breach and (prior_breach is None or prior_breach)
    if persistent:
        action = "redevelop" if max_csi > config.PSI_THRESHOLDS["action"] else "re-estimate"
    elif breach:
        action = "watch (first breach, no model action)" + recal
    elif below_reference and versus == "watch":
        action = "watch (no model action)" + recal
    elif below_reference:
        action = "investigate population (no model change)" + recal
    elif calibration_off:
        action = "recalibrate"
    elif population_moved:
        action = "investigate population (no model change)"
    else:
        action = "no action"
    out = {"action": action, "gini_now": round(gini_now, 4), "gini_ci_low": round(gini_now_ci[0], 4),
           "gini_ci_high": round(gini_now_ci[1], 4), "gini_drop": round(gini_drop, 4)}
    if expected_gini_ci is not None:
        out.update({"expected_gini": round(expected, 4), "gap_to_expected": round(gap, 4), "half_width": round(half, 4)})
    out.update({"below_reference": bool(below_reference), "versus_expected": versus, "breach": bool(breach),
                "persistent_breach": bool(persistent), "calibration_off": bool(calibration_off),
                "psi": round(psi_value, 4), "max_csi": round(max_csi, 4), "calib_slope": round(calib_slope, 3),
                "calib_intercept_shift": round(calib_intercept_shift, 3)})
    return out


def decide_periods(evidence: list[dict]) -> list[dict]:
    """Apply :func:`decide_action` to periods in time order, carrying the breach status forward so a breach
    must persist for two consecutive periods before it triggers re-estimation. Each item holds the keyword
    arguments of :func:`decide_action` (without ``prior_breach``); an item with ``gini_now`` = NaN (24-month
    outcome not observable) gets no performance action and resets the run of breaches."""
    out, prior = [], False
    for ev in evidence:
        if ev["gini_now"] == ev["gini_now"]:
            d = decide_action(**ev, prior_breach=prior); prior = d["breach"]
        else:
            d = {"action": "distribution checks only (24m not observable)", "psi": round(ev["psi_value"], 4),
                 "max_csi": round(ev["max_csi"], 4)}; prior = False
        out.append(d)
    return out


def monitoring_summary(dev_scores, dev_specs: dict[str, BinSpec], frame: pd.DataFrame, gini_dev: float,
                       period_col: str = "vintage_quarter") -> dict:
    """Everything the monitoring report needs for the periods in ``frame``, plus the decision per period
    (each period's Gini judged against its own DeLong interval and against its expected Gini; a breach must
    persist for two consecutive periods before the model is re-estimated)."""
    s_psi = score_psi_by_period(dev_scores, frame, period_col=period_col)
    s_csi = csi_by_period(dev_specs, frame, period_col=period_col)
    perf = early_performance_by_period(frame, period_col=period_col)
    cal = calibration_by_period(frame, period_col=period_col)
    periods, evidence = sorted(s_psi[period_col]), []
    for per in periods:
        p = float(s_psi.loc[s_psi[period_col] == per, "psi"].iloc[0])
        g_row = perf.loc[perf[period_col] == per]
        g_now = float(g_row["gini_24m"].iloc[0]) if "gini_24m" in g_row and g_row["gini_24m"].notna().any() else float("nan")
        g_ci = (float(g_row["gini_24m_ci_low"].iloc[0]), float(g_row["gini_24m_ci_high"].iloc[0])) if g_now == g_now else None
        x_ci = ((float(g_row["expected_ci_low"].iloc[0]), float(g_row["expected_ci_high"].iloc[0]))
                if g_now == g_now and "expected_ci_low" in g_row else None)
        c_row = cal.loc[cal[period_col] == per]
        slope = float(c_row["slope"].iloc[0]) if len(c_row) else 1.0
        shift = float(c_row["intercept_shift"].iloc[0]) if len(c_row) else 0.0
        mx = float(s_csi[per].max()) if per in s_csi.columns else 0.0
        evidence.append(dict(psi_value=p, gini_dev=gini_dev, gini_now=g_now, gini_now_ci=g_ci, calib_slope=slope, max_csi=mx,
                             calib_intercept_shift=shift, expected_gini_ci=x_ci))
    decisions = [{period_col: per, **d} for per, d in zip(periods, decide_periods(evidence))]
    return {"score_psi": s_psi, "csi": s_csi, "attribute_drift": attribute_drift_ranking(s_csi), "performance": perf,
            "calibration": cal, "decisions": pd.DataFrame(decisions)}
