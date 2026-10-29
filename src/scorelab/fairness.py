"""Fair-lending tests for the pilot (NB08): adverse impact ratio, standardized mean difference,
performance and calibration by group, and a less-discriminatory-alternative search.

Lending Club carries no protected-class fields. The notebook chooses a proxy (for example a
geography-based proxy from ``addr_state``) and documents it; this module only computes the tests
for whatever ``group`` column it is handed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scorelab.validation import auc_gini_ks, calibration_slope_intercept


def approval_flags(score, approval_rate: float, higher_is_safer: bool = True) -> np.ndarray:
    s = np.asarray(score, dtype=float); s = s if higher_is_safer else -s
    k = int(round(approval_rate * len(s)))
    return s >= np.sort(s)[::-1][k - 1]


def adverse_impact_ratio(approved, groups, reference: str | int | None = None) -> pd.DataFrame:
    """Approval rate per group and its ratio to the reference group (highest approval rate if None).
    The four-fifths rule flags ratios below 0.8."""
    df = pd.DataFrame({"a": np.asarray(approved, dtype=int), "g": np.asarray(groups)})
    g = df.groupby("g")["a"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "approval_rate"})
    ref = reference if reference is not None else g.approval_rate.idxmax()
    g["air"] = (g.approval_rate / g.loc[ref, "approval_rate"]).round(4)
    g["below_four_fifths"] = g.air < 0.8
    g["reference"] = g.index == ref
    return g.round(4).reset_index().rename(columns={"g": "group"})


def standardized_mean_difference(score, groups, reference=None) -> pd.DataFrame:
    df = pd.DataFrame({"s": np.asarray(score, dtype=float), "g": np.asarray(groups)})
    g = df.groupby("g")["s"].agg(["size", "mean", "std"]).rename(columns={"size": "n"})
    ref = reference if reference is not None else g["mean"].idxmax()
    pooled = np.sqrt((g["std"] ** 2 + g.loc[ref, "std"] ** 2) / 2)
    g["smd_vs_reference"] = ((g["mean"] - g.loc[ref, "mean"]) / pooled).round(4)
    g["reference"] = g.index == ref
    return g.round(4).reset_index().rename(columns={"g": "group"})


def group_performance(y, score, eta, groups, higher_is_safer: bool = True) -> pd.DataFrame:
    """Gini, KS, bad rate and calibration slope per group (equal performance and equal calibration)."""
    rows = []
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "s": np.asarray(score, dtype=float),
                       "e": np.asarray(eta, dtype=float), "g": np.asarray(groups)})
    for g, d in df.groupby("g"):
        row = {"group": g, "n": len(d), "bad_rate": round(d.y.mean(), 4), "mean_score": round(d.s.mean(), 1)}
        if d.y.nunique() == 2 and len(d) >= 200:
            m = auc_gini_ks(d.y, d.s, higher_is_safer); c = calibration_slope_intercept(d.y, d.e)
            row.update({"gini": round(m["gini"], 4), "ks": round(m["ks"], 4),
                        "calib_slope": round(c["slope"], 3), "calib_intercept": round(c["intercept"], 3)})
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_summary(y, score, eta, groups, approval_rate: float, higher_is_safer: bool = True, reference=None) -> dict:
    appr = approval_flags(score, approval_rate, higher_is_safer)
    return {"air": adverse_impact_ratio(appr, groups, reference),
            "smd": standardized_mean_difference(score, groups, reference),
            "performance": group_performance(y, score, eta, groups, higher_is_safer)}


def lda_search(fit_and_score, attributes: list[str], y, groups, approval_rate: float, reference=None,
               higher_is_safer: bool = True) -> pd.DataFrame:
    """Less-discriminatory-alternative search by leave-one-attribute-out.

    ``fit_and_score(attrs) -> (score_array, eta_array)`` refits the card on ``attrs`` and scores the
    evaluation set. Returns, per dropped attribute, the Gini and the worst AIR, so the trade-off
    between predictive power and disparate impact is explicit.
    """
    rows = []
    base_s, base_e = fit_and_score(attributes)
    air0 = adverse_impact_ratio(approval_flags(base_s, approval_rate, higher_is_safer), groups, reference)
    rows.append({"dropped": "(none)", "gini": round(auc_gini_ks(y, base_s, higher_is_safer)["gini"], 4),
                 "min_air": round(air0.air.min(), 4)})
    for a in attributes:
        rest = [b for b in attributes if b != a]
        s, e = fit_and_score(rest)
        air = adverse_impact_ratio(approval_flags(s, approval_rate, higher_is_safer), groups, reference)
        rows.append({"dropped": a, "gini": round(auc_gini_ks(y, s, higher_is_safer)["gini"], 4), "min_air": round(air.air.min(), 4)})
    df = pd.DataFrame(rows)
    df["gini_cost"] = (df.loc[0, "gini"] - df.gini).round(4)
    df["air_gain"] = (df.min_air - df.loc[0, "min_air"]).round(4)
    return df
