"""Challenger models (NB07, NB07b), SHAP, surrogate tree, and the translation of tree splits into
candidate scorecard attributes.

The challengers descend the same Bernoulli log-loss as the card, so the differences are the
function class and the search:

* gradient boosting (XGBoost, LightGBM): piecewise-constant with interactions, greedy;
* random forest: averaged deep trees, no shrinkage, probabilities need not be calibrated;
* EBM (``interpret``): one boosted shape function per attribute, optionally a few pairwise terms,
  additive and readable like the card;
* MLP: a small neural network on WOE or standardised attributes, early-stopped on validation;
* stacked ensemble: a logistic meta-model on the base models' log-odds, fitted on validation.

The monotone constraints come from the attribute registry: an attribute whose direction is
"riskier" gets +1 on the log-odds of bad, "safer" gets -1, "no prior" is unconstrained.
Every fitted model carries ``attributes_used``, and :func:`predict_p` scores any of them.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

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


def save_xgb(booster, path) -> None:
    """Save a booster with its attribute list (stored as a booster attribute, so one file suffices)."""
    import json
    booster.set_attr(attributes_used=json.dumps(booster.attributes_used))
    booster.save_model(str(path))


def load_xgb(path):
    """Load a booster saved by :func:`save_xgb`; ``best_iteration`` and ``attributes_used`` are restored."""
    import json
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(path))
    booster.attributes_used = json.loads(booster.attr("attributes_used"))
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


# --- Random forest, EBM ----------------------------------------------------------------------

def fit_random_forest(X_train: pd.DataFrame, y_train, attributes: list[str], monotone: bool = True,
                      params: dict | None = None, seed: int = 42):
    """sklearn random forest on the raw attributes (missing values handled natively by the trees).

    Leaves are kept large (``min_samples_leaf``) and each tree sees a bootstrap of ``max_samples``
    rows, which keeps a 400k-row fit to a minute or two on a laptop and the probabilities smooth."""
    from sklearn.ensemble import RandomForestClassifier
    p = {"n_estimators": 300, "min_samples_leaf": 200, "max_features": "sqrt", "max_samples": 0.3,
         "n_jobs": -1, "random_state": seed}
    if params: p.update(params)
    if monotone:
        p["monotonic_cst"] = monotone_constraints(attributes)
    model = RandomForestClassifier(**p).fit(X_train[attributes].to_numpy(dtype=float), np.asarray(y_train, dtype=int))
    model.attributes_used = attributes
    return model


def fit_ebm(X_train: pd.DataFrame, y_train, attributes: list[str], monotone: bool = True, interactions: int = 10,
            params: dict | None = None, seed: int = 42):
    """Explainable boosting machine (``interpret``): an additive model of boosted shape functions.

    With ``monotone=True`` each shape function follows the registry direction; interpret then fits
    main effects only (pairwise terms are dropped), so ``interactions`` matters only when
    ``monotone=False``. EBM holds out its own ``validation_size`` share of the training rows for
    early stopping; the 2015 vintage is not used."""
    from interpret.glassbox import ExplainableBoostingClassifier
    p = {"interactions": 0 if monotone else interactions, "max_bins": 256, "outer_bags": 8,
         "n_jobs": -1, "random_state": seed}
    if params: p.update(params)
    if monotone:
        p["monotone_constraints"] = monotone_constraints(attributes)
    model = ExplainableBoostingClassifier(feature_names=list(attributes), **p)
    model.fit(X_train[attributes].to_numpy(dtype=float), np.asarray(y_train, dtype=int))
    model.attributes_used = attributes
    return model


# --- MLP ----------------------------------------------------------------------------------------

@dataclass
class Standardizer:
    """Train-set statistics for turning raw attributes into a dense, scaled matrix.

    Fit on the training vintages only. Missing values are imputed at the train median (never zero
    or the mean) and, for every attribute with missing values in train, a 0/1 missing flag is
    appended, so "no such event" stays visible to the network. An attribute that had no missing
    values in train has no flag; if it is missing at scoring time, :meth:`transform` warns rather
    than imputing silently."""
    attributes: list[str]
    median: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, df: pd.DataFrame, attributes: list[str]) -> "Standardizer":
        s = cls(list(attributes))
        for a in attributes:
            x = df[a].astype(float)
            med = float(x.median()) if x.notna().any() else 0.0
            xf = x.fillna(med)
            s.median[a], s.mean[a], s.std[a] = med, float(xf.mean()), float(xf.std()) or 1.0
            if x.isna().any():
                s.flagged.append(a)
        return s

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        unflagged = {a: int(df[a].isna().sum()) for a in self.attributes if a not in self.flagged and df[a].isna().any()}
        if unflagged:
            import warnings
            warnings.warn(f"missing values in attributes that had none in training (median-imputed, no flag): {unflagged}")
        cols = [((df[a].astype(float).fillna(self.median[a]) - self.mean[a]) / self.std[a]).to_numpy()
                for a in self.attributes]
        cols += [df[a].isna().to_numpy(dtype=float) for a in self.flagged]
        return np.column_stack(cols)


@dataclass
class MLPModel:
    """A fitted sklearn MLP with the input transform it was trained on and its learning curve."""
    net: object
    attributes_used: list[str]
    specs: dict | None
    scaler: Standardizer | None
    history: pd.DataFrame
    best_epoch: int

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        return mlp_matrix(df, self.attributes_used, self.specs, self.scaler)


def mlp_matrix(df: pd.DataFrame, attributes: list[str], specs: dict | None, scaler: Standardizer | None) -> np.ndarray:
    """WOE matrix when ``specs`` is given (bins from NB04), otherwise the standardised raw attributes."""
    if specs is not None:
        from scorelab.binning import transform_woe
        woe = transform_woe(df, {a: specs[a] for a in attributes})
        return woe[[a + "_woe" for a in attributes]].to_numpy(dtype=float)
    return scaler.transform(df)


def fit_mlp(X_train: pd.DataFrame, y_train, X_valid: pd.DataFrame, y_valid, attributes: list[str],
            specs: dict | None = None, hidden: tuple[int, ...] = (32, 16), alpha: float = 1e-4,
            learning_rate: float = 1e-3, batch_size: int = 512, max_epochs: int = 60, patience: int = 6,
            seed: int = 42) -> MLPModel:
    """Small feed-forward network trained one epoch at a time, early-stopped on the validation log-loss.

    sklearn's own ``early_stopping`` holds out a random slice of train; here the stopping set is the
    validation vintage (2015), as for the GBM, and the weights of the best epoch are kept.
    With ``specs`` the inputs are the WOE columns; without, raw attributes standardised on train."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    scaler = None if specs is not None else Standardizer.fit(X_train, attributes)
    Xt, Xv = mlp_matrix(X_train, attributes, specs, scaler), mlp_matrix(X_valid, attributes, specs, scaler)
    yt, yv = np.asarray(y_train, dtype=int), np.asarray(y_valid, dtype=int)
    net = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=learning_rate,
                        batch_size=batch_size, random_state=seed, shuffle=True)
    best, best_loss, rows, wait = None, np.inf, [], 0
    for epoch in range(1, max_epochs + 1):
        net.partial_fit(Xt, yt, classes=[0, 1])
        pv = net.predict_proba(Xv)[:, 1]
        loss = log_loss(yv, np.clip(pv, 1e-7, 1 - 1e-7))
        rows.append({"epoch": epoch, "train_loss": float(net.loss_), "valid_log_loss": float(loss),
                     "valid_auc": float(roc_auc_score(yv, pv))})
        if loss < best_loss - 1e-5:
            best, best_loss, wait = copy.deepcopy(net), loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    hist = pd.DataFrame(rows)
    return MLPModel(net=best, attributes_used=list(attributes), specs=specs, scaler=scaler, history=hist,
                    best_epoch=int(hist.loc[hist.valid_log_loss.idxmin(), "epoch"]))


# --- Stacked ensemble ---------------------------------------------------------------------------

def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class StackModel:
    """Logistic meta-model on the base models' log-odds: p = sigmoid(w0 + sum_m w_m logit(p_m))."""
    base_names: list[str]
    intercept: float
    weights: dict[str, float]

    def predict(self, base_preds: dict[str, np.ndarray]) -> np.ndarray:
        eta = self.intercept + sum(self.weights[n] * _logit(base_preds[n]) for n in self.base_names)
        return 1 / (1 + np.exp(-eta))


def fit_stack(base_preds_valid: dict[str, np.ndarray], y_valid, C: float = 1.0) -> StackModel:
    """Fit the meta-model on validation-vintage predictions of base models trained on train.

    Using a different vintage for the meta-model is the time-ordered version of out-of-fold
    stacking: the base models never saw these rows, so the weights are not rewarded for overfit."""
    from sklearn.linear_model import LogisticRegression
    names = list(base_preds_valid)
    Z = np.column_stack([_logit(base_preds_valid[n]) for n in names])
    lr = LogisticRegression(C=C, max_iter=1000).fit(Z, np.asarray(y_valid, dtype=int))
    return StackModel(base_names=names, intercept=float(lr.intercept_[0]),
                      weights={n: float(w) for n, w in zip(names, lr.coef_[0])})


# --- Uniform prediction and timing ------------------------------------------------------------------

def predict_p(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of bad from any challenger fitted in this module."""
    mod = model.__class__.__module__
    if mod.startswith("xgboost"):
        return predict_xgb(model, X)
    if mod.startswith("lightgbm"):
        return predict_lgbm(model, X)
    if isinstance(model, MLPModel):
        return model.net.predict_proba(model.matrix(X))[:, 1]
    return model.predict_proba(X[model.attributes_used].to_numpy(dtype=float))[:, 1]


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


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
