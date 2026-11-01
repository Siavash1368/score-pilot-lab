"""Discrimination, calibration, stability and swap-set analysis (NB08, NB09, NB07).

Every function takes arrays or Series and returns numbers or tables; nothing plots and nothing
reads the database, so the same code runs in the notebooks, the MCP server and the agent.

Conventions: ``y`` is 1 = bad. A *score* is higher = safer (300-850); a *probability* or *eta* is
higher = riskier. Functions say which they expect; ``higher_is_safer`` flips where needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, log_loss as _log_loss, brier_score_loss


def _risk(x, higher_is_safer):
    x = np.asarray(x, dtype=float)
    return -x if higher_is_safer else x


# --- Discrimination ----------------------------------------------------------------------

def auc_gini_ks(y, x, higher_is_safer: bool = True) -> dict:
    """AUC, Gini (= 2 AUC - 1) and KS of ``x`` for outcome ``y``."""
    y = np.asarray(y, dtype=int); r = _risk(x, higher_is_safer)
    auc = roc_auc_score(y, r)
    fpr, tpr, _ = roc_curve(y, r)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(tpr - fpr)), "n": int(len(y)),
            "n_bad": int(y.sum())}


def _delong_components(y, r):
    """Placement values for DeLong's variance (fast O(n log n) version)."""
    y = np.asarray(y, dtype=int); r = np.asarray(r, dtype=float)
    pos, neg = r[y == 1], r[y == 0]
    m, n = len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    ranks = stats.rankdata(allv)
    rpos = stats.rankdata(pos); rneg = stats.rankdata(neg)
    v10 = (ranks[:m] - rpos) / n            # structural components for positives
    v01 = 1 - (ranks[m:] - rneg) / m        # for negatives
    auc = v10.mean()
    return auc, v10, v01


def delong_ci(y, x, higher_is_safer: bool = True, alpha: float = 0.05) -> dict:
    """AUC and Gini with DeLong standard error and confidence interval."""
    auc, v10, v01 = _delong_components(y, _risk(x, higher_is_safer))
    var = v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01)
    se = float(np.sqrt(var)); z = stats.norm.ppf(1 - alpha / 2)
    return {"auc": float(auc), "auc_se": se, "auc_ci": (float(auc - z * se), float(auc + z * se)),
            "gini": float(2 * auc - 1), "gini_se": 2 * se, "gini_ci": (float(2 * (auc - z * se) - 1), float(2 * (auc + z * se) - 1))}


def delong_test(y, x1, x2, higher_is_safer: bool = True) -> dict:
    """Paired DeLong test that two scores on the same consumers have equal AUC."""
    a1, v10_1, v01_1 = _delong_components(y, _risk(x1, higher_is_safer))
    a2, v10_2, v01_2 = _delong_components(y, _risk(x2, higher_is_safer))
    s10 = np.cov(np.vstack([v10_1, v10_2])); s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / len(v10_1) + s01 / len(v01_1)
    diff = a1 - a2; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
    se = 2 * float(np.sqrt(var)) if var > 0 else 0.0          # SE of the Gini difference
    return {"auc_1": float(a1), "auc_2": float(a2), "gini_diff": float(2 * diff), "gini_diff_se": se,
            "gini_diff_ci": (float(2 * diff - 1.96 * se), float(2 * diff + 1.96 * se)), "z": float(z), "p_value": float(p)}


def bootstrap_gini_diff(y, x1, x2, higher_is_safer: bool = True, n_boot: int = 500, seed: int = 42) -> dict:
    """Bootstrap interval for Gini(x1) - Gini(x2) on the same rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); r1 = _risk(x1, higher_is_safer); r2 = _risk(x2, higher_is_safer)
    n = len(y); diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            diffs[b] = np.nan; continue
        diffs[b] = 2 * (roc_auc_score(y[i], r1[i]) - roc_auc_score(y[i], r2[i]))
    diffs = diffs[~np.isnan(diffs)]
    return {"gini_diff": float(2 * (roc_auc_score(y, r1) - roc_auc_score(y, r2))),
            "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), "n_boot": int(len(diffs))}


def gains_table(y, x, n_bands: int = 10, higher_is_safer: bool = True) -> pd.DataFrame:
    """Decile-style gains: bands from riskiest to safest, bad rate, cumulative bad capture, cumulative share."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "r": _risk(x, higher_is_safer)})
    df["band"] = pd.qcut(df["r"].rank(method="first", ascending=False), n_bands, labels=range(1, n_bands + 1))
    g = df.groupby("band", observed=True)["y"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "bads"})
    g["bad_rate"] = (g.bads / g.n).round(4)
    g["cum_share_population"] = (g.n.cumsum() / g.n.sum()).round(4)
    g["cum_share_bads"] = (g.bads.cumsum() / g.bads.sum()).round(4)
    g["cum_share_goods"] = ((g.n - g.bads).cumsum() / (g.n - g.bads).sum()).round(4)
    g["ks_at_band"] = (g.cum_share_bads - g.cum_share_goods).round(4)
    g["lift"] = (g.bad_rate / (g.bads.sum() / g.n.sum())).round(3)
    return g.reset_index()


# --- Swap set -------------------------------------------------------------------------------

def _approve_top_k(score: np.ndarray, k: int, tiebreak: np.ndarray) -> np.ndarray:
    """Approve exactly ``k`` loans with the highest score; ties at the cutoff are broken by ``tiebreak``."""
    a = np.zeros(len(score), dtype=bool)
    a[np.lexsort((tiebreak, -score))[:k]] = True
    return a


def swap_set(y, incumbent, challenger, approval_rate: float, higher_is_safer: bool = True, seed: int = 0) -> dict:
    """Decisions at a matched approval rate: approve the safest ``approval_rate`` share under each score.

    Both scores approve exactly the same number of loans. Coarse scores (FICO moves in 5-point steps) tie at
    the cutoff; the tied loans are split by one seeded random key shared by both scores, so identical scores
    give identical decisions. Returns the 2x2 of decisions, the bad rates of swap-ins (approved by challenger
    only) and swap-outs (approved by incumbent only), and the approved-book bad rate under each score.
    """
    y = np.asarray(y, dtype=int)
    inc = np.asarray(incumbent, dtype=float); ch = np.asarray(challenger, dtype=float)
    if not higher_is_safer:
        inc, ch = -inc, -ch
    k = int(round(approval_rate * len(y)))
    key = np.random.default_rng(seed).random(len(y))
    a_inc = _approve_top_k(inc, k, key); a_ch = _approve_top_k(ch, k, key)
    both = a_inc & a_ch; swap_in = a_ch & ~a_inc; swap_out = a_inc & ~a_ch; neither = ~a_inc & ~a_ch

    def rate(m): return float(y[m].mean()) if m.sum() else float("nan")
    tbl = pd.DataFrame({"challenger_approve": [int(both.sum()), int(swap_in.sum())],
                        "challenger_decline": [int(swap_out.sum()), int(neither.sum())]},
                       index=["incumbent_approve", "incumbent_decline"])
    return {"approval_rate": approval_rate, "n": int(len(y)), "crosstab": tbl,
            "n_swap_in": int(swap_in.sum()), "n_swap_out": int(swap_out.sum()),
            "bad_rate_swap_in": rate(swap_in), "bad_rate_swap_out": rate(swap_out), "bad_rate_both": rate(both),
            "bad_rate_approved_incumbent": rate(a_inc), "bad_rate_approved_challenger": rate(a_ch),
            "bads_avoided_per_10k_approved": float((rate(a_inc) - rate(a_ch)) * 10_000)}


def swap_curve(y, incumbent, challenger, approval_rates=(0.5, 0.6, 0.7, 0.8, 0.9), higher_is_safer: bool = True) -> pd.DataFrame:
    rows = []
    for r in approval_rates:
        s = swap_set(y, incumbent, challenger, r, higher_is_safer)
        rows.append({k: v for k, v in s.items() if k != "crosstab"})
    return pd.DataFrame(rows)


# --- Calibration -------------------------------------------------------------------------------

def calibration_by_band(y, p, n_bands: int = 10, by_score: pd.Series | None = None, band_width: int = 20) -> pd.DataFrame:
    """Observed vs predicted bad rate per band with a binomial 95% interval.

    Bands are deciles of ``p`` unless ``by_score`` is given (then fixed-width score bands)."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    if by_score is not None:
        s = np.asarray(by_score, dtype=float)
        lo = np.floor(s.min() / band_width) * band_width; hi = np.ceil(s.max() / band_width) * band_width
        df["band"] = pd.cut(s, np.arange(lo, hi + band_width, band_width), right=False)
    else:
        df["band"] = pd.qcut(df["p"].rank(method="first"), n_bands, labels=range(1, n_bands + 1))
    g = df.groupby("band", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    se = np.sqrt(g.predicted * (1 - g.predicted) / g.n)
    g["lower_95"] = (g.predicted - 1.96 * se).clip(0, 1); g["upper_95"] = (g.predicted + 1.96 * se).clip(1e-9, 1)
    g["within_interval"] = (g.observed >= g.lower_95) & (g.observed <= g.upper_95)
    g["binomial_p"] = [min(1.0, 2 * min(stats.binom.cdf(k, n, p), 1 - stats.binom.cdf(k - 1, n, p)))
                       for k, n, p in zip((g.observed * g.n).round().astype(int), g.n, g.predicted.clip(1e-9, 1 - 1e-9))]
    return g.round(4).reset_index()


def calibration_slope_intercept(y, eta) -> dict:
    """Cox recalibration: logit(y) = a + b * eta. b = 1, a = 0 means the level transfers.
    Also the intercept-only shift (calibration-in-the-large) with b fixed at 1."""
    import statsmodels.api as sm
    yv = np.asarray(y, dtype=float); e = np.asarray(eta, dtype=float)
    res = sm.GLM(yv, sm.add_constant(e), family=sm.families.Binomial()).fit()
    res0 = sm.GLM(yv, np.ones((len(yv), 1)), family=sm.families.Binomial(), offset=e).fit()
    return {"intercept": float(res.params[0]), "slope": float(res.params[1]),
            "slope_se": float(res.bse[1]), "slope_ci": (float(res.params[1] - 1.96 * res.bse[1]), float(res.params[1] + 1.96 * res.bse[1])),
            "intercept_only_shift": float(res0.params[0]),
            "mean_predicted": float(np.mean(1 / (1 + np.exp(-e)))), "mean_observed": float(yv.mean())}


def fit_recalibration(y, p) -> dict:
    """Logistic recalibration (Cox / Platt on the log-odds): logit(p_new) = a + b * logit(p).

    Fit on a vintage the model was not trained on (here the 2015 validation year), then applied unchanged
    to later years. Rank order is untouched when b > 0, so Gini and KS do not change; only the level and
    spread of the probabilities do."""
    c = calibration_slope_intercept(y, _logit(p))
    return {"a": c["intercept"], "b": c["slope"], "n": int(len(np.asarray(y))), "mean_observed": c["mean_observed"],
            "mean_predicted_before": c["mean_predicted"]}


def apply_recalibration(p, cal: dict) -> np.ndarray:
    """Recalibrated probabilities from :func:`fit_recalibration` parameters."""
    eta = cal["a"] + cal["b"] * _logit(p)
    return 1 / (1 + np.exp(-eta))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def hosmer_lemeshow(y, p, n_groups: int = 10) -> dict:
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    df["g"] = pd.qcut(df.p.rank(method="first"), n_groups, labels=False)
    g = df.groupby("g").agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"))
    chi2 = float((((g.o - g.e) ** 2) / (g.e * (1 - g.e / g.n)).replace(0, np.nan)).sum())
    return {"chi2": chi2, "df": n_groups - 2, "p_value": float(1 - stats.chi2.cdf(chi2, n_groups - 2))}


def probability_scores(y, p) -> dict:
    """Log-loss, Brier and the Murphy decomposition (reliability - resolution + uncertainty) on deciles."""
    y = np.asarray(y, dtype=int); p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    df = pd.DataFrame({"y": y, "p": p}); df["g"] = pd.qcut(df.p.rank(method="first"), 10, labels=False)
    g = df.groupby("g").agg(n=("y", "size"), o=("y", "mean"), pm=("p", "mean"))
    ybar = y.mean()
    reliability = float((g.n * (g.pm - g.o) ** 2).sum() / len(y))
    resolution = float((g.n * (g.o - ybar) ** 2).sum() / len(y))
    uncertainty = float(ybar * (1 - ybar))
    return {"log_loss": float(_log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
            "reliability": reliability, "resolution": resolution, "uncertainty": uncertainty,
            "brier_check": reliability - resolution + uncertainty,
            "null_log_loss": float(_log_loss(y, np.full(len(y), ybar)))}


# --- Model comparison ---------------------------------------------------------------------------

def model_comparison(y, preds: dict[str, np.ndarray], reference: str | None = None,
                     fit_seconds: dict[str, float] | None = None, notes: dict[str, str] | None = None,
                     inputs: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per model on the same rows: its input set, AUC and Gini with DeLong 95% CI, KS, Brier,
    log-loss, Cox calibration slope and intercept (logit(y) = a + b * eta), the intercept-only shift
    (calibration-in-the-large, b fixed at 1), fit time and an explainability note.

    ``preds`` maps model name -> probability of bad. With ``reference`` (a key of ``preds``), each
    row also carries its Gini difference from the reference with the paired DeLong 95% CI, whether
    that CI excludes zero, and the p-value."""
    y = np.asarray(y, dtype=int)
    rows = []
    for name, p in preds.items():
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        d = delong_ci(y, p, higher_is_safer=False)
        cal = calibration_slope_intercept(y, np.log(p / (1 - p)))
        row = {"model": name, "inputs": (inputs or {}).get(name, ""), "auc": d["auc"], "gini": d["gini"],
               "gini_ci_low": d["gini_ci"][0],
               "gini_ci_high": d["gini_ci"][1], "ks": auc_gini_ks(y, p, higher_is_safer=False)["ks"],
               "brier": float(brier_score_loss(y, p)), "log_loss": float(_log_loss(y, p)),
               "calib_slope": cal["slope"], "calib_intercept": cal["intercept"],
               "calib_in_the_large": cal["intercept_only_shift"]}
        if reference is not None:
            if name == reference:
                row.update({"gini_vs_ref": 0.0, "gini_vs_ref_ci_low": np.nan, "gini_vs_ref_ci_high": np.nan,
                            "ci_excludes_zero": None, "delong_p_vs_ref": np.nan})
            else:
                t = delong_test(y, p, preds[reference], higher_is_safer=False)
                lo, hi = t["gini_diff_ci"]
                row.update({"gini_vs_ref": t["gini_diff"], "gini_vs_ref_ci_low": lo, "gini_vs_ref_ci_high": hi,
                            "ci_excludes_zero": bool(lo > 0 or hi < 0), "delong_p_vs_ref": t["p_value"]})
        row["fit_seconds"] = (fit_seconds or {}).get(name, np.nan)
        row["explainability"] = (notes or {}).get(name, "")
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    return out


# --- Stability --------------------------------------------------------------------------------

def psi(dev, cur, bins: int | np.ndarray = 10) -> dict:
    """Population stability index of ``cur`` against ``dev`` on ``bins`` (int = dev quantile bins, or edges).

    Also the chi-square reference (Yurdakul 2018): under no shift PSI ~ chi2(B-1) * (1/n_dev + 1/n_cur),
    so the critical PSI at 5% is returned alongside the 0.10 / 0.25 rule of thumb."""
    dev = np.asarray(dev, dtype=float); cur = np.asarray(cur, dtype=float)
    if isinstance(bins, int):
        edges = np.unique(np.quantile(dev, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.asarray(bins, dtype=float)
    d = np.histogram(dev, edges)[0] / len(dev); c = np.histogram(cur, edges)[0] / len(cur)
    eps = 1e-4   # floor so an emptied bin reads as a large shift, not an infinite one
    parts = (c - d) * np.log((c + eps) / (d + eps))
    B = len(d)
    crit = float(stats.chi2.ppf(0.95, B - 1) * (1 / len(dev) + 1 / len(cur)))
    value = float(parts.sum())
    p_value = float(1 - stats.chi2.cdf(value / (1 / len(dev) + 1 / len(cur)), B - 1))
    table = pd.DataFrame({"lower": edges[:-1], "upper": edges[1:], "dev_share": np.round(d, 4), "cur_share": np.round(c, 4),
                          "psi_part": np.round(parts, 5)})
    return {"psi": value, "critical_psi_5pct": crit, "p_value": p_value, "n_bins": B,
            "verdict": "action" if value > 0.25 else "watch" if value > 0.10 else "stable", "table": table, "edges": edges}


def gini_by_group(y, x, groups, higher_is_safer: bool = True) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "x": np.asarray(x, dtype=float), "g": np.asarray(groups)})
    for g, d in df.groupby("g"):
        if d.y.nunique() < 2 or len(d) < 50:
            rows.append({"group": g, "n": len(d), "bad_rate": round(d.y.mean(), 4), "gini": np.nan, "ks": np.nan}); continue
        m = auc_gini_ks(d.y, d.x, higher_is_safer)
        rows.append({"group": g, "n": len(d), "bad_rate": round(d.y.mean(), 4), "gini": round(m["gini"], 4), "ks": round(m["ks"], 4)})
    return pd.DataFrame(rows)
