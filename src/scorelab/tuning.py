"""Hyper-parameter search with Optuna for the GBM and the MLP (NB07b).

Every trial trains on the training vintages and is scored on the validation vintage (2015) only;
the out-of-time vintage (2016) is never seen during the search, so the comparison table in NB07b
stays an honest out-of-time test. The objective is validation AUC (rank order is what a score
sells); the log-loss of each trial is stored alongside so calibration can be checked.

Optuna's TPE sampler is seeded, so a rerun with the same data and trial budget gives the same search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scorelab import challengers as C


def _study(seed: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))


def _trials_table(study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    df["duration_s"] = df.pop("duration").dt.total_seconds().round(1)
    return df.rename(columns={"value": "valid_auc"}).sort_values("valid_auc", ascending=False).reset_index(drop=True)


def tune_xgb(train: pd.DataFrame, valid: pd.DataFrame, attributes: list[str], n_trials: int = 30,
             timeout: float | None = None, monotone: bool = True, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Search depth, child weight, learning rate, sampling and regularisation; early-stopped on ``valid``.

    Returns the best parameters (plus the number of boosting rounds it stopped at) and the trials table."""
    yt, yv = train["bad"], valid["bad"]

    def objective(trial):
        params = {"max_depth": trial.suggest_int("max_depth", 2, 6),
                  "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000, log=True),
                  "eta": trial.suggest_float("eta", 0.02, 0.2, log=True),
                  "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                  "lambda": trial.suggest_float("lambda", 0.1, 30, log=True),
                  "gamma": trial.suggest_float("gamma", 0.0, 5.0)}
        booster = C.fit_xgb(train, yt, valid, yv, attributes, monotone=monotone, params=params, seed=seed)
        p = C.predict_xgb(booster, valid)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        trial.set_user_attr("valid_log_loss", float(log_loss(yv, p)))
        return float(roc_auc_score(yv, p))

    study = _study(seed)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = dict(study.best_params)
    best["n_rounds"] = int(study.best_trial.user_attrs["best_iteration"]) + 1
    return best, _trials_table(study)


def tune_mlp(train: pd.DataFrame, valid: pd.DataFrame, attributes: list[str], specs: dict | None = None,
             n_trials: int = 20, timeout: float | None = None, max_epochs: int = 40,
             seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Search architecture, L2 penalty, learning rate and batch size; each trial early-stops on ``valid``."""
    yt, yv = train["bad"], valid["bad"]
    shapes = {"16": (16,), "32": (32,), "32-16": (32, 16), "64-32": (64, 32), "64-32-16": (64, 32, 16)}

    def objective(trial):
        hidden = shapes[trial.suggest_categorical("hidden", list(shapes))]
        model = C.fit_mlp(train, yt, valid, yv, attributes, specs=specs, hidden=hidden,
                          alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                          learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                          batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
                          max_epochs=max_epochs, seed=seed)
        best_row = model.history.loc[model.history.valid_log_loss.idxmin()]
        trial.set_user_attr("best_epoch", int(best_row.epoch))
        trial.set_user_attr("valid_log_loss", float(best_row.valid_log_loss))
        return float(roc_auc_score(yv, C.predict_p(model, valid)))

    study = _study(seed)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = dict(study.best_params)
    best["hidden"] = shapes[best["hidden"]]
    return best, _trials_table(study)


def best_trial_gap(trials: pd.DataFrame, top: int = 5) -> float:
    """Spread of validation AUC among the ``top`` trials: a small spread means the search has
    flattened out and more trials are unlikely to change the out-of-time picture."""
    v = trials["valid_auc"].head(top).to_numpy()
    return float(np.max(v) - np.min(v)) if len(v) else float("nan")
