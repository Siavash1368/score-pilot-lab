"""Double machine learning for an illustrative treatment effect (NB07e).

Question: does a 60-month term, rather than 36, change the probability of going bad within 24
months, for applicants with the same application attributes?

**This is illustrative, not a causal claim for policy.** The borrower chooses the term and the
lender prices it (rate and grade depend on term), so selection on unobservables (why this
borrower wanted lower payments) is almost certain and no set of application attributes rules it
out. Rate and grade are *consequences* of the term choice (mediators) and are therefore not
controlled for; controlling for them would remove part of the effect being estimated.

Two estimators, both with K-fold cross-fitting (Chernozhukov et al. 2018):

* **Partially linear model (PLR):** Y = theta D + g(X) + e, D = m(X) + v. Residualise Y and D on X
  with flexible learners, then theta = sum(v_hat * u_hat) / sum(v_hat^2), with the
  heteroskedasticity-robust standard error from the score.
* **Interactive model (IRM / AIPW):** the average treatment effect from doubly robust scores
  psi = mu1(X) - mu0(X) + D (Y - mu1(X)) / e(X) - (1 - D)(Y - mu0(X)) / (1 - e(X)), with the
  propensity e(X) trimmed to [trim, 1 - trim]. Also the effect on the treated (ATT).

Learners default to sklearn's histogram gradient boosting, which handles missing values natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scorelab.target import _sql_list, sample_predicate

# Application-time fields from ``accepted`` used only as controls here (never as score inputs).
# ``purpose`` is already in v_accepted_asof (key/context), so only home_ownership comes from accepted.
APPLICATION_CONTROLS = ["loan_amnt", "annual_inc", "dti", "emp_length"]
APPLICATION_CATEGORICALS = ["purpose", "home_ownership"]
_FROM_ACCEPTED = APPLICATION_CONTROLS + ["home_ownership"]


def causal_frame(con, splits: tuple[str, ...] = ("train", "validate", "oot"), sample_frac: float | None = None,
                 seed: int = 42) -> pd.DataFrame:
    """Observable, labelled loans of 36 or 60 months with the bureau attributes (``v_accepted_asof``),
    application controls from ``accepted``, ``treated`` = 60-month term and ``bad``.

    Interest rate, grade and instalment are deliberately absent: they are set after the term is chosen."""
    sample = sample_predicate(sample_frac, seed, id_col="s.id")
    ctrl = ", ".join(f"acc.{c}" for c in _FROM_ACCEPTED)
    return con.sql(f"""
        SELECT s.split, s.vintage_year, s.term_months,
               CAST(s.term_months = 60 AS INTEGER) AS treated,
               CASE WHEN s.label = 'bad' THEN 1 ELSE 0 END AS bad,
               {ctrl}, a.*
        FROM v_split s
        JOIN v_accepted_asof a USING (id)
        JOIN accepted acc USING (id)
        WHERE s.split IN ({_sql_list(splits)}) AND s.observable = 1 AND s.exclusion IS NULL
          AND s.label IN ('bad', 'good') AND s.term_months IN (36, 60)
          {sample}
        ORDER BY s.id   -- fixed row order: the cross-fitting folds depend on it
    """).df()


def control_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str] | None = None) -> pd.DataFrame:
    """Numeric controls as floats (missing kept) plus one-hot categoricals with the first level dropped,
    so the regression benchmark with a constant is full rank (tree learners are unaffected)."""
    X = df[numeric].astype(float)
    if categorical:
        X = pd.concat([X, pd.get_dummies(df[categorical].astype(str), dtype=float, drop_first=True)], axis=1)
    return X


def _default_learner(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200,
          "early_stopping": True, "validation_fraction": 0.1, "random_state": seed}
    return HistGradientBoostingClassifier(**kw) if kind == "clf" else HistGradientBoostingRegressor(**kw)


def _folds(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % k


def _fit_predict_p(make, X, y, Xp):
    m = make().fit(X, y)
    return m.predict_proba(Xp)[:, 1]


def dml_plr(y, d, X: pd.DataFrame, n_folds: int = 5, seed: int = 42, learner_y=None, learner_d=None) -> dict:
    """Partially linear DML: theta, robust SE, 95% CI, p-value, and nuisance fit diagnostics.

    With binary Y and D, theta is the effect on the probability scale (risk difference)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=float); d = np.asarray(d, dtype=float)
    Xv = X.to_numpy(dtype=float)
    ly = learner_y or (lambda: _default_learner("clf", seed))
    ld = learner_d or (lambda: _default_learner("clf", seed))
    folds = _folds(len(y), n_folds, seed)
    ell, m = np.zeros(len(y)), np.zeros(len(y))
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        ell[te] = _fit_predict_p(ly, Xv[tr], y[tr], Xv[te])
        m[te] = _fit_predict_p(ld, Xv[tr], d[tr], Xv[te])
    u, v = y - ell, d - m
    theta = float((v * u).sum() / (v * v).sum())
    psi = (u - theta * v) * v
    se = float(np.sqrt(np.mean(psi ** 2) / np.mean(v * v) ** 2 / len(y)))
    z = theta / se
    return {"estimator": "PLR", "theta": theta, "se": se, "ci": (theta - 1.96 * se, theta + 1.96 * se),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))), "n": int(len(y)), "n_folds": n_folds,
            "outcome_auc": float(roc_auc_score(y, ell)), "propensity_auc": float(roc_auc_score(d, m)),
            "propensity": m}


def dml_irm(y, d, X: pd.DataFrame, n_folds: int = 5, trim: float = 0.01, seed: int = 42,
            learner_y=None, learner_d=None) -> dict:
    """Interactive-model DML (AIPW): ATE and ATT with robust SEs, cross-fitted nuisances, trimmed propensity."""
    y = np.asarray(y, dtype=float); d = np.asarray(d, dtype=int)
    Xv = X.to_numpy(dtype=float)
    ly = learner_y or (lambda: _default_learner("clf", seed))
    ld = learner_d or (lambda: _default_learner("clf", seed))
    folds = _folds(len(y), n_folds, seed)
    mu0, mu1, e = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        e[te] = _fit_predict_p(ld, Xv[tr], d[tr], Xv[te])
        t1, t0 = tr & (d == 1), tr & (d == 0)
        mu1[te] = _fit_predict_p(ly, Xv[t1], y[t1], Xv[te])
        mu0[te] = _fit_predict_p(ly, Xv[t0], y[t0], Xv[te])
    n_trimmed = int(((e < trim) | (e > 1 - trim)).sum())
    e = np.clip(e, trim, 1 - trim)
    psi_ate = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    ate = float(psi_ate.mean()); se_ate = float(psi_ate.std(ddof=1) / np.sqrt(len(y)))
    p1 = d.mean()
    psi_att = (d * (y - mu0) - e * (1 - d) * (y - mu0) / (1 - e)) / p1
    att = float(psi_att.mean()); se_att = float((psi_att - att * d / p1).std(ddof=1) / np.sqrt(len(y)))

    def ci(est, se):
        return (est - 1.96 * se, est + 1.96 * se)
    return {"estimator": "IRM (AIPW)", "ate": ate, "ate_se": se_ate, "ate_ci": ci(ate, se_ate),
            "ate_p_value": float(2 * (1 - stats.norm.cdf(abs(ate / se_ate)))),
            "att": att, "att_se": se_att, "att_ci": ci(att, se_att), "n": int(len(y)), "n_treated": int(d.sum()),
            "n_propensity_trimmed": n_trimmed, "trim": trim, "propensity": e}


def naive_and_adjusted(y, d, X: pd.DataFrame) -> pd.DataFrame:
    """Benchmarks: the raw difference in bad rates, and a main-effects logistic regression's average
    marginal effect of D (missing controls median-imputed with flags).

    Bureau fields tend to be missing as a block, so many missing flags are identical; duplicates are
    dropped, then any column still linearly dependent (pivoted QR) is dropped, so the fit is full rank.
    The constant and D are always kept."""
    import statsmodels.api as sm
    from scipy.linalg import qr
    y = np.asarray(y, dtype=float); d = np.asarray(d, dtype=float)
    raw = float(y[d == 1].mean() - y[d == 0].mean())
    Xi = X.copy()
    for c in X.columns[X.isna().any()]:
        Xi[c + "_missing"] = X[c].isna().astype(float); Xi[c] = X[c].fillna(X[c].median())
    Xi = Xi.loc[:, Xi.std() > 0]
    Xi = Xi.loc[:, ~Xi.T.duplicated().to_numpy()]
    Z = sm.add_constant(np.column_stack([d, Xi.to_numpy(dtype=float)]))
    R, piv = qr(Z, mode="r", pivoting=True)
    diag = np.abs(np.diag(R))
    keep = np.sort(piv[: int((diag > 1e-9 * diag[0]).sum())])
    if not {0, 1} <= set(keep):
        raise ValueError("treatment is collinear with the controls")
    Z = Z[:, keep]
    res = sm.GLM(y, Z, family=sm.families.Binomial()).fit()
    Z1, Z0 = Z.copy(), Z.copy(); Z1[:, 1] = 1; Z0[:, 1] = 0
    ame = float(np.mean(res.predict(Z1) - res.predict(Z0)))
    return pd.DataFrame([{"estimator": "raw difference in bad rate", "effect": raw},
                         {"estimator": "logistic regression, average marginal effect", "effect": ame}])


def overlap_table(propensity: np.ndarray, d, bins: int = 10) -> pd.DataFrame:
    """Treated and control counts by propensity decile: where the two groups are comparable."""
    df = pd.DataFrame({"e": propensity, "d": np.asarray(d, dtype=int)})
    df["band"] = pd.cut(df.e, np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("band", observed=False).agg(n=("d", "size"), treated=("d", "sum"))
    g["control"] = g.n - g.treated
    return g.reset_index()


CAVEAT = ("Illustrative only. Term is chosen by the borrower and priced by the lender, so borrowers who "
          "pick 60 months differ in ways the application attributes do not record (payment pressure, "
          "purpose, expected income path). The estimate is the association that remains after flexible "
          "adjustment for observed application attributes, not the effect of a policy that moved "
          "borrowers from 36 to 60 months.")
