"""Explainability and nonlinear structure of the best challenger (NB07c).

* **Partial dependence** (and ICE curves) for any model through :func:`scorelab.challengers.predict_p`,
  on the log-odds scale so shapes compare with the card's WOE.
* **Interaction strength** two ways: XGBoost's exact pairwise SHAP interaction values
  (:func:`shap_interaction_pairs`) and Friedman and Popescu's H statistic (:func:`friedman_h`),
  which is model-agnostic.
* **Carrying an interaction back to the card**: cross two attributes at the GBM's own split
  thresholds (:func:`cross_attribute`), bin the cross like any categorical attribute, and refit the
  card with it (:func:`refit_card_with`) to measure the Gini the card recovers.
* **Reason-code agreement** between the card (points lost against each attribute's best bin) and
  the GBM (largest positive SHAP contributions) for the same applicants.

Nothing here plots; the notebook draws (``shap`` for summary and dependence plots, matplotlib for PD).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from scorelab import binning as B, scorecard as S, validation as V
from scorelab.challengers import predict_p, split_table


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --- Partial dependence -----------------------------------------------------------------------------

def partial_dependence(model, X: pd.DataFrame, attribute: str, grid: np.ndarray | None = None, n_grid: int = 20,
                       max_rows: int = 3000, include_missing: bool = True, ice: int = 0,
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Average log-odds of bad as ``attribute`` is set to each grid value for every row of (a sample of) X.

    The grid defaults to the 2.5th–97.5th percentile range in ``n_grid`` quantile steps; with
    ``include_missing`` a final point sets the attribute to missing. Returns the PD table
    (value, mean_logit, mean_p, is_missing) and, when ``ice`` > 0, the log-odds curves of ``ice`` rows."""
    Xs = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    if grid is None:
        grid = np.unique(np.nanquantile(X[attribute].to_numpy(dtype=float), np.linspace(0.025, 0.975, n_grid)))
    values = list(grid) + ([np.nan] if include_missing and X[attribute].isna().any() else [])
    curves = []
    for v in values:
        Xm = Xs.copy(); Xm[attribute] = v
        curves.append(_logit(predict_p(model, Xm)))
    curves = np.array(curves)                       # grid x rows
    pd_tab = pd.DataFrame({"value": values, "mean_logit": curves.mean(axis=1),
                           "mean_p": (1 / (1 + np.exp(-curves))).mean(axis=1),
                           "is_missing": [v != v for v in values]})
    ice_tab = None
    if ice:
        idx = np.random.default_rng(seed).choice(curves.shape[1], min(ice, curves.shape[1]), replace=False)
        ice_tab = pd.DataFrame(curves[:, idx], index=pd.Index(values, name="value"))
    return pd_tab, ice_tab


# --- Interaction strength --------------------------------------------------------------------------

def shap_interaction_pairs(booster, X: pd.DataFrame, top: int = 15, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Mean |SHAP interaction| per attribute pair from XGBoost's exact pairwise interaction values.

    ``share_of_interaction`` is the pair's share of all off-diagonal interaction mass; ``main_a`` and
    ``main_b`` are the attributes' mean |main effect| for scale."""
    import xgboost as xgb
    attrs = booster.attributes_used
    Xs = X[attrs].sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X[attrs]
    inter = booster.predict(xgb.DMatrix(Xs, missing=np.nan), pred_interactions=True,
                            iteration_range=(0, booster.best_iteration + 1))[:, :-1, :-1]
    mabs = np.abs(inter).mean(axis=0)
    rows = [{"a": attrs[i], "b": attrs[j], "mean_abs_interaction": float(2 * mabs[i, j]),
             "main_a": float(mabs[i, i]), "main_b": float(mabs[j, j])}
            for i, j in itertools.combinations(range(len(attrs)), 2)]
    df = pd.DataFrame(rows)
    df["share_of_interaction"] = df.mean_abs_interaction / df.mean_abs_interaction.sum()
    return df.sort_values("mean_abs_interaction", ascending=False).head(top).round(5).reset_index(drop=True)


def friedman_h(model, X: pd.DataFrame, a: str, b: str, n_rows: int = 300, seed: int = 42) -> float:
    """Friedman and Popescu's H^2 for the pair (a, b), on the log-odds scale.

    H^2 = sum_i [PD_ab(x_i) - PD_a(x_i) - PD_b(x_i)]^2 / sum_i PD_ab(x_i)^2 with every PD centred,
    evaluated at ``n_rows`` sampled rows (cost ~ 3 n_rows^2 predictions). 0 = purely additive in
    (a, b); the share of the joint effect that is interaction otherwise."""
    Xs = X.sample(min(n_rows, len(X)), random_state=seed).reset_index(drop=True)
    n = len(Xs)
    rep = Xs.loc[np.tile(np.arange(n), n)].reset_index(drop=True)      # every row's other attributes ...
    at = np.repeat(np.arange(n), n)                                    # ... combined with row i's (a, b)

    def pd_at(cols):
        Xm = rep.copy()
        for c in cols:
            Xm[c] = Xs[c].to_numpy()[at]
        f = _logit(predict_p(model, Xm)).reshape(n, n).mean(axis=1)
        return f - f.mean()

    pab, pa, pb = pd_at([a, b]), pd_at([a]), pd_at([b])
    den = float((pab ** 2).sum())
    return float(((pab - pa - pb) ** 2).sum() / den) if den > 0 else 0.0


def friedman_h_pairs(model, X: pd.DataFrame, pairs: list[tuple[str, str]], n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    return pd.DataFrame([{"a": a, "b": b, "h2": round(friedman_h(model, X, a, b, n_rows, seed), 4)} for a, b in pairs]
                        ).sort_values("h2", ascending=False).reset_index(drop=True)


# --- From an interaction to a card attribute ---------------------------------------------------------

def top_threshold(booster, attribute: str) -> float:
    """The split threshold with the most total gain for ``attribute`` in the booster."""
    s = split_table(booster)
    s = s[s.attribute == attribute]
    if s.empty:
        raise ValueError(f"{attribute} is never split on")
    return float(s.groupby("threshold")["gain"].sum().idxmax())


def cross_attribute(df: pd.DataFrame, a: str, b: str, thr_a: float, thr_b: float) -> pd.Series:
    """Categorical cross of two attributes at one threshold each (missing kept as its own level).

    Levels read like "a<=t | b>u"; bin the result with :func:`scorelab.binning.fit_categorical`."""
    def side(x, t, name):
        x = df[x].astype(float)
        return np.where(x.isna(), f"{name} missing", np.where(x > t, f"{name}>{t:g}", f"{name}<={t:g}"))
    return pd.Series(pd.Series(side(a, thr_a, a)).str.cat(pd.Series(side(b, thr_b, b)), sep=" | ").to_numpy(),
                     index=df.index, name=f"{a}_x_{b}")


def refit_card_with(train: pd.DataFrame, oot: pd.DataFrame, card: S.Scorecard, specs: dict, extra_specs: dict,
                    p_enter: float = 0.01, max_corr: float = 0.7, max_attributes: int = 20) -> dict:
    """Add ``extra_specs`` (new binned attributes) to the card's candidates, rerun stepwise selection on
    train and report the OOT Gini before and after with DeLong intervals and the paired test."""
    all_specs = {**specs, **extra_specs}
    W = B.transform_woe(train, all_specs)
    sel, _ = S.stepwise_select(W, train["bad"], card.attributes + list(extra_specs), p_enter=p_enter,
                               max_corr=max_corr, max_attributes=max_attributes)
    new = S.fit_logistic(W, train["bad"], sel)
    s_old, s_new = S.score(oot, card, specs)["score"], S.score(oot, new, all_specs)["score"]
    g_old, g_new = V.delong_ci(oot["bad"], s_old), V.delong_ci(oot["bad"], s_new)
    t = V.delong_test(oot["bad"], s_new, s_old)
    return {"card": new, "specs": all_specs, "selected": sel, "entered": [a for a in sel if a in extra_specs],
            "gini_before": g_old["gini"], "gini_after": g_new["gini"], "gini_after_ci": g_new["gini_ci"],
            "gini_gain": t["gini_diff"], "p_value": t["p_value"]}


# --- Reason-code agreement ----------------------------------------------------------------------------

def _strip(code: str) -> str:
    return code.split(" [")[0].strip()


def reason_code_agreement(card_codes: pd.DataFrame, gbm_codes: pd.DataFrame, n_codes: int = 4) -> tuple[dict, pd.DataFrame]:
    """Compare the top-``n_codes`` reasons of the card and the GBM for the same applicants.

    Card codes carry " [bin label]"; that suffix is ignored. Returns a summary (mean overlap, share
    with the same first reason, share with no reason in common, rank correlation of shared reasons)
    and a per-row table. Both sides should be restricted to applicants the card would decline, since
    reasons are only ever given for declines."""
    cols = [f"reason_{i + 1}" for i in range(n_codes)]
    rows = []
    for (_, rc), (_, rg) in zip(card_codes[cols].iterrows(), gbm_codes[cols].iterrows()):
        c = [_strip(x) for x in rc if x]; g = [x for x in rg if x]
        shared = set(c) & set(g)
        rank_gap = np.mean([abs(c.index(s) - g.index(s)) for s in shared]) if shared else np.nan
        rows.append({"n_card": len(c), "n_gbm": len(g), "overlap": len(shared),
                     "same_first": bool(c and g and c[0] == g[0]), "mean_rank_gap_shared": rank_gap})
    per_row = pd.DataFrame(rows, index=card_codes.index)
    summary = {"n": int(len(per_row)), "mean_overlap": float(per_row.overlap.mean()),
               "share_same_first_reason": float(per_row.same_first.mean()),
               "share_no_reason_in_common": float((per_row.overlap == 0).mean()),
               "share_full_agreement": float((per_row.overlap == np.minimum(per_row.n_card, per_row.n_gbm)).mean()),
               "mean_rank_gap_shared": float(per_row.mean_rank_gap_shared.mean())}
    return summary, per_row


def first_reason_crosstab(card_codes: pd.DataFrame, gbm_codes: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Crosstab of the first reason under each model (most frequent card reasons as rows)."""
    c = card_codes["reason_1"].map(_strip); g = gbm_codes["reason_1"]
    keep = c.value_counts().head(top).index
    return pd.crosstab(c.where(c.isin(keep), "other"), g.where(g.isin(keep), "other"))
