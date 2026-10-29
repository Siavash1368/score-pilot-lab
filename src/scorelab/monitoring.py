"""Monitoring (NB09): score and attribute stability, early-performance tracking of censored
vintages, calibration drift, drift simulation, and the recalibrate / re-estimate / redevelop rule.

The monitor vintages (2017Q1-2018Q4) are censored at 24 months, so performance is tracked with
the 12-month early indicator (``early_bad``) and the distribution with PSI/CSI; the 2016 OOT year
serves as the reference for both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scorelab import config
from scorelab.binning import BinSpec, csi
from scorelab.validation import psi, auc_gini_ks, calibration_slope_intercept


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

def early_performance_by_period(frame: pd.DataFrame, score_col: str = "score", period_col: str = "vintage_quarter",
                                early_col: str = "early_bad", bad_col: str = "bad") -> pd.DataFrame:
    """Per period: early (12-month) bad rate and Gini, plus the 24-month figures where observable."""
    rows = []
    for per, d in frame.groupby(period_col):
        row = {period_col: per, "n": len(d)}
        e = d[d[early_col].notna()]
        if len(e) and e[early_col].nunique() == 2:
            row.update({"n_early_observable": len(e), "early_bad_rate": round(e[early_col].mean(), 4),
                        "early_gini": round(auc_gini_ks(e[early_col], e[score_col])["gini"], 4)})
        b = d[d[bad_col].notna()]
        if len(b) and b[bad_col].nunique() == 2:
            row.update({"n_24m_observable": len(b), "bad_rate_24m": round(b[bad_col].mean(), 4),
                        "gini_24m": round(auc_gini_ks(b[bad_col], b[score_col])["gini"], 4)})
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

    kind = 'covariate' : shift the listed numeric attributes by ``strength`` dev standard deviations
                         (P(X) moves; P(Y|X) does not), outcomes untouched.
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
        for a in attributes or []:
            sd = out[a].std(skipna=True)
            out[a] = out[a] + strength * sd
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


# --- Decision rule ----------------------------------------------------------------------------

def decide_action(psi_value: float, gini_dev: float, gini_now: float, gini_ci_halfwidth: float,
                  calib_slope: float, max_csi: float, slope_band: tuple[float, float] = (0.9, 1.1),
                  calib_intercept_shift: float = 0.0, max_intercept_shift: float = 0.15) -> dict:
    """Recalibrate / re-estimate / redevelop / no action, from what moved.

    * ordering holds (Gini drop inside its interval) and level off (slope outside band, intercept
      shift beyond ``max_intercept_shift`` log-odds, or PSI past the watch threshold)
      -> recalibrate (intercept and slope);
    * ordering decays (Gini drop beyond its interval) with attributes stable (max CSI <= 0.25)
      -> re-estimate coefficients on recent data, same bins;
    * attributes drifted (max CSI > 0.25) -> redevelop;
    * otherwise no action.
    """
    gini_drop = gini_dev - gini_now
    ordering_decayed = gini_drop > gini_ci_halfwidth
    level_off = (not (slope_band[0] <= calib_slope <= slope_band[1]) or abs(calib_intercept_shift) > max_intercept_shift
                 or psi_value > config.PSI_THRESHOLDS["watch"])
    if max_csi > config.PSI_THRESHOLDS["action"]:
        action = "redevelop"
    elif ordering_decayed:
        action = "re-estimate"
    elif level_off:
        action = "recalibrate"
    else:
        action = "no action"
    return {"action": action, "gini_drop": round(gini_drop, 4), "ordering_decayed": bool(ordering_decayed),
            "level_off": bool(level_off), "psi": round(psi_value, 4), "max_csi": round(max_csi, 4),
            "calib_slope": round(calib_slope, 3), "calib_intercept_shift": round(calib_intercept_shift, 3)}


def monitoring_summary(dev_scores, dev_specs: dict[str, BinSpec], frame: pd.DataFrame, gini_dev: float,
                       gini_ci_halfwidth: float, period_col: str = "vintage_quarter") -> dict:
    """Everything the monitoring report needs for the periods in ``frame``, plus the decision per period."""
    s_psi = score_psi_by_period(dev_scores, frame, period_col=period_col)
    s_csi = csi_by_period(dev_specs, frame, period_col=period_col)
    perf = early_performance_by_period(frame, period_col=period_col)
    cal = calibration_by_period(frame, period_col=period_col)
    decisions = []
    for per in s_psi[period_col]:
        p = float(s_psi.loc[s_psi[period_col] == per, "psi"].iloc[0])
        g_row = perf.loc[perf[period_col] == per]
        g_now = float(g_row["gini_24m"].iloc[0]) if "gini_24m" in g_row and g_row["gini_24m"].notna().any() else float("nan")
        c_row = cal.loc[cal[period_col] == per]
        slope = float(c_row["slope"].iloc[0]) if len(c_row) else 1.0
        shift = float(c_row["intercept_shift"].iloc[0]) if len(c_row) else 0.0
        mx = float(s_csi[per].max()) if per in s_csi.columns else 0.0
        if g_now == g_now:
            decisions.append({period_col: per, **decide_action(p, gini_dev, g_now, gini_ci_halfwidth, slope, mx,
                                                               calib_intercept_shift=shift)})
        else:
            decisions.append({period_col: per, "action": "distribution checks only (24m not observable)", "psi": round(p, 4), "max_csi": round(mx, 4)})
    return {"score_psi": s_psi, "csi": s_csi, "attribute_drift": attribute_drift_ranking(s_csi), "performance": perf,
            "calibration": cal, "decisions": pd.DataFrame(decisions)}
