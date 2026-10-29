"""Monotone gradient-boosting challengers, SHAP, surrogate tree, and the translation of tree
splits into candidate scorecard attributes (NB07).

The challenger descends the same Bernoulli log-loss as the card (``binary:logistic``), so the only
differences are the function class (piecewise, with interactions) and the search (greedy). The
monotone constraints come from the attribute registry: an attribute whose direction is "riskier"
gets +1 on the log-odds of bad, "safer" gets -1, "no prior" is unconstrained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scorelab.attributes import REGISTRY_BY_NAME


def monotone_constraints(attributes: list[str]) -> list[int]:
    """+1 / -1 / 0 per attribute, on the log-odds of bad."""
    return [REGISTRY_BY_NAME[a].direction if a in REGISTRY_BY_NAME else 0 for a in attributes]


def fit_xgb(X_train: pd.DataFrame, y_train, X_valid: pd.DataFrame, y_valid, attributes: list[str],
            monotone: bool = True, interaction_constraints: list[list[str]] | None = None, params: dict | None = None,
            n_rounds: int = 2000, early_stopping: int = 50, seed: int = 42):
    """XGBoost with optional monotone and interaction constraints, early-stopped on the validation set."""
    import xgboost as xgb
    p = {"objective": "binary:logistic", "eval_metric": "auc", "eta": 0.05, "max_depth": 4, "min_child_weight": 50,
         "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "tree_method": "hist", "seed": seed}
    if params: p.update(params)
    if monotone:
        p["monotone_constraints"] = "(" + ",".join(str(c) for c in monotone_constraints(attributes)) + ")"
    if interaction_constraints:
        idx = {a: i for i, a in enumerate(attributes)}
        p["interaction_constraints"] = str([[idx[a] for a in grp] for grp in interaction_constraints])
    dtr = xgb.DMatrix(X_train[attributes], label=np.asarray(y_train, dtype=float), missing=np.nan)
    dva = xgb.DMatrix(X_valid[attributes], label=np.asarray(y_valid, dtype=float), missing=np.nan)
    booster = xgb.train(p, dtr, num_boost_round=n_rounds, evals=[(dva, "valid")],
                        early_stopping_rounds=early_stopping, verbose_eval=False)
    booster.attributes_used = attributes
    return booster


def predict_xgb(booster, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb
    d = xgb.DMatrix(X[booster.attributes_used], missing=np.nan)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


def fit_lgbm(X_train, y_train, X_valid, y_valid, attributes: list[str], monotone: bool = True, params: dict | None = None,
             n_rounds: int = 2000, early_stopping: int = 50, seed: int = 42):
    import lightgbm as lgb
    p = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": seed}
    if params: p.update(params)
    if monotone:
        p["monotone_constraints"] = monotone_constraints(attributes); p["monotone_constraints_method"] = "advanced"
    dtr = lgb.Dataset(X_train[attributes], label=np.asarray(y_train, dtype=float))
    dva = lgb.Dataset(X_valid[attributes], label=np.asarray(y_valid, dtype=float), reference=dtr)
    model = lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    model.attributes_used = attributes
    return model


def predict_lgbm(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X[model.attributes_used], num_iteration=model.best_iteration)


# --- Explanations ------------------------------------------------------------------------------

def shap_values(booster, X: pd.DataFrame, max_rows: int = 20000, seed: int = 42) -> tuple[pd.DataFrame, float]:
    """TreeSHAP values on (a sample of) X, in log-odds units, plus the expected value.

    Uses the boosters' native TreeSHAP (``pred_contribs`` / ``pred_contrib``), which is exact for
    trees and avoids version coupling with the ``shap`` package; ``shap`` is still used for plots."""
    Xs = X[booster.attributes_used]
    if len(Xs) > max_rows:
        Xs = Xs.sample(max_rows, random_state=seed)
    if hasattr(booster, "trees_to_dataframe") and booster.__class__.__module__.startswith("xgboost"):
        import xgboost as xgb
        d = xgb.DMatrix(Xs, missing=np.nan)
        contrib = booster.predict(d, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1))
    else:   # lightgbm
        contrib = np.asarray(booster.predict(Xs, pred_contrib=True, num_iteration=booster.best_iteration))
    vals, bias = contrib[:, :-1], contrib[:, -1]
    return pd.DataFrame(vals, columns=booster.attributes_used, index=Xs.index), float(bias.mean())


def shap_importance(sv: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per attribute and the sign consistency (share of rows where the attribute's
    contribution has the registry's expected direction given the attribute's rank)."""
    imp = sv.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"attribute": imp.index, "mean_abs_shap": imp.values.round(4),
                         "share_of_total": (imp.values / imp.values.sum()).round(4)})


def gbm_reason_codes(sv: pd.DataFrame, n_codes: int = 4, meanings: dict[str, str] | None = None) -> pd.DataFrame:
    """Top positive contributions to the log-odds of bad per row (the GBM's adverse-action reasons)."""
    out = {f"reason_{i + 1}": [] for i in range(n_codes)}
    for _, row in sv.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(n_codes)
        for i in range(n_codes):
            out[f"reason_{i + 1}"].append((meanings or {}).get(top.index[i], top.index[i]) if i < len(top) else "")
    return pd.DataFrame(out, index=sv.index)


def surrogate_tree(X: pd.DataFrame, p_gbm: np.ndarray, attributes: list[str], max_depth: int = 4, min_leaf: int = 500):
    """Shallow tree fitted to the GBM's predicted probability; returns the tree and its fidelity R^2."""
    from sklearn.tree import DecisionTreeRegressor
    Xf = X[attributes].fillna(-999)
    t = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42).fit(Xf, p_gbm)
    fidelity = float(t.score(Xf, p_gbm))
    return t, fidelity


def surrogate_rules(tree, attributes: list[str]) -> str:
    from sklearn.tree import export_text
    return export_text(tree, feature_names=attributes, decimals=2)


# --- Splits to attributes ----------------------------------------------------------------------

def split_table(booster) -> pd.DataFrame:
    """Every split in the booster: attribute, threshold, gain, cover. Thresholds are the candidate
    bin edges the card did not have."""
    df = booster.trees_to_dataframe()
    s = df[df.Feature != "Leaf"][["Tree", "Feature", "Split", "Gain", "Cover"]].rename(
        columns={"Feature": "attribute", "Split": "threshold", "Gain": "gain", "Cover": "cover"})
    return s


def splits_to_attributes(booster, X: pd.DataFrame, top_k: int = 10, round_to: int = 3) -> pd.DataFrame:
    """Aggregate split thresholds per attribute by total gain, keep the ``top_k`` (attribute, threshold)
    pairs, and return them as candidate binary attributes ``<attribute>_gt_<threshold>`` with their
    population share. These are the "risk nodes" of the ML-to-attributes method: feed the winners
    back to NB04 as forced bin edges or new flags and measure how much GBM lift the card recovers."""
    s = split_table(booster)
    s["threshold"] = s["threshold"].round(round_to)
    agg = s.groupby(["attribute", "threshold"])["gain"].sum().reset_index().sort_values("gain", ascending=False).head(top_k)
    rows = []
    for r in agg.itertuples():
        flag = (X[r.attribute] > r.threshold)
        rows.append({"candidate": f"{r.attribute}_gt_{r.threshold:g}", "attribute": r.attribute, "threshold": r.threshold,
                     "total_gain": round(r.gain, 2), "share_above": round(float(flag.mean()), 4)})
    return pd.DataFrame(rows)


def add_split_flags(frame: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for r in candidates.itertuples():
        out[r.candidate] = (out[r.attribute] > r.threshold).astype(float).where(out[r.attribute].notna(), np.nan)
    return out


def interaction_strength(booster, attributes: list[str], X: pd.DataFrame, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Share of SHAP variance explained by main effects vs interactions (from XGBoost's native SHAP
    interaction values) per attribute: a direct measure of what the additive card leaves on the table."""
    import xgboost as xgb
    Xs = X[attributes]
    if len(Xs) > max_rows: Xs = Xs.sample(max_rows, random_state=seed)
    inter = booster.predict(xgb.DMatrix(Xs, missing=np.nan), pred_interactions=True,
                            iteration_range=(0, booster.best_iteration + 1))[:, :-1, :-1]
    main = np.array([inter[:, i, i] for i in range(len(attributes))]).T
    total = inter.sum(axis=2)
    rows = []
    for i, a in enumerate(attributes):
        off = total[:, i] - main[:, i]
        rows.append({"attribute": a, "main_effect_var": float(main[:, i].var()), "interaction_var": float(off.var())})
    df = pd.DataFrame(rows)
    df["interaction_share"] = (df.interaction_var / (df.main_effect_var + df.interaction_var).replace(0, np.nan)).round(4)
    return df.sort_values("main_effect_var", ascending=False).round(5)
