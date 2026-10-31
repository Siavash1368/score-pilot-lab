"""Discrete-time survival model of default by months on book (NB07d).

Each loan contributes one row per month on book until it leaves the risk set:

    default    the month 90+ DPD is reached (the same reconstruction as the label: last payment
               + 1 month = first miss, + 3 months = 90 DPD), when that month is inside the window
    prepay     a Fully Paid loan leaves at its last payment month (inside the window)
    censored   still open at the end of the window (or delinquent only after it)

The hazard is a multinomial logit on the person-month rows with three outcomes (continue, default,
prepay): log[h_k(t | x) / h_0(t | x)] = gamma_k(t) + x' beta_k, with gamma_k(t) a piecewise-constant
baseline in months on book. Treating prepayment as a *competing* risk, not as censoring, makes the
cumulative incidence of default at month 24,

    CIF_d(24 | x) = sum_{t <= 24} S(t - 1 | x) h_d(t | x),   S(t) = prod_{s <= t} [1 - h_d(s) - h_p(s)],

the probability that the loan is bad under the label definition, so it compares directly with
the card's PD. A cause-specific curve (prepayment as censoring) is also available: it answers
"default probability if nobody prepaid" and is always higher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scorelab import config

DEFAULT, PREPAY, CONTINUE = 1, 2, 0


def event_frame(con, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``months_to_last_payment`` from ``v_split`` to a modelling frame (matched on id)."""
    m = con.sql("SELECT id, months_to_last_payment FROM v_split WHERE observable = 1").df()
    return frame.merge(m, on="id", how="left")


def event_times(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Add ``exit_month`` (1..window) and ``exit_type`` (default / prepay / censored) per loan.

    Needs ``loan_status`` and ``months_to_last_payment``. A loan has ``exit_type == 'default'``
    exactly when its label is bad, which the tests check."""
    w = window or config.PERFORMANCE_WINDOW_MONTHS
    out = df.copy()
    mlp = out["months_to_last_payment"].fillna(0).astype(float)
    delinquent = out["loan_status"].isin(config.BAD_STATUSES + config.DELINQUENT_STATUSES).to_numpy()
    dpd90 = (mlp + config.MONTHS_LAST_PAYMENT_TO_FIRST_MISS + config.MONTHS_FIRST_MISS_TO_90DPD).to_numpy()
    paid = out["loan_status"].str.endswith("Fully Paid").to_numpy()
    is_default = delinquent & (dpd90 <= w)
    is_prepay = ~is_default & paid & (mlp.to_numpy() < w)
    out["exit_type"] = np.where(is_default, "default", np.where(is_prepay, "prepay", "censored"))
    out["exit_month"] = np.where(is_default, dpd90, np.where(is_prepay, np.maximum(mlp.to_numpy(), 1), w)).astype(int)
    return out


def person_period(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Expand to one row per loan-month: ``month`` 1..exit_month, ``outcome`` 0 until the exit month,
    then 1 (default) or 2 (prepay); censored loans end with 0."""
    n = df["exit_month"].to_numpy()
    idx = np.repeat(np.arange(len(df)), n)
    month = np.concatenate([np.arange(1, k + 1) for k in n]) if len(n) else np.array([], dtype=int)
    last = np.cumsum(n) - 1
    outcome = np.zeros(len(idx), dtype=int)
    code = df["exit_type"].map({"default": DEFAULT, "prepay": PREPAY, "censored": CONTINUE}).to_numpy()
    outcome[last] = code
    pp = df[covariates].iloc[idx].reset_index(drop=True)
    pp.insert(0, "month", month); pp.insert(1, "outcome", outcome)
    pp.insert(0, "loan_row", idx)
    return pp


def month_bands(edges: tuple[int, ...] = (3, 6, 9, 12, 15, 18, 21, 24)) -> list[tuple[int, int]]:
    """Piecewise-constant baseline bands, default 3-month steps up to 24."""
    lo, out = 1, []
    for e in edges:
        out.append((lo, e)); lo = e + 1
    return out


class HazardModel:
    """Multinomial discrete-time hazard: baseline dummies per month band plus covariate effects."""

    def __init__(self, covariates: list[str], bands: list[tuple[int, int]] | None = None, C: float = 1e4):
        self.covariates, self.bands, self.C = list(covariates), bands or month_bands(), C
        self.model = None

    def _design(self, month: np.ndarray, X: np.ndarray) -> np.ndarray:
        D = np.column_stack([((month >= lo) & (month <= hi)).astype(float) for lo, hi in self.bands])
        return np.hstack([D, X])

    def fit(self, pp: pd.DataFrame, sample_weight=None) -> "HazardModel":
        from sklearn.linear_model import LogisticRegression
        Z = self._design(pp["month"].to_numpy(), pp[self.covariates].to_numpy(dtype=float))
        self.model = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
        self.model.fit(Z, pp["outcome"].to_numpy(), sample_weight=sample_weight)
        self.classes_ = list(self.model.classes_)
        return self

    def hazards(self, X: pd.DataFrame, horizon: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(h_default, h_prepay), each n_loans x horizon."""
        H = horizon or self.bands[-1][1]
        Xv = X[self.covariates].to_numpy(dtype=float)
        hd, hp = np.zeros((len(Xv), H)), np.zeros((len(Xv), H))
        for t in range(1, H + 1):
            pr = self.model.predict_proba(self._design(np.full(len(Xv), t), Xv))
            hd[:, t - 1] = pr[:, self.classes_.index(DEFAULT)]
            hp[:, t - 1] = pr[:, self.classes_.index(PREPAY)] if PREPAY in self.classes_ else 0.0
        return hd, hp

    def cumulative_incidence(self, X: pd.DataFrame, horizon: int | None = None,
                             competing: bool = True) -> np.ndarray:
        """CIF of default by month (n_loans x horizon). ``competing=False`` treats prepayment as
        censoring: 1 - prod(1 - h_d), the default probability if nobody prepaid."""
        hd, hp = self.hazards(X, horizon)
        if not competing:
            return 1 - np.cumprod(1 - hd, axis=1)
        surv_prev = np.hstack([np.ones((len(hd), 1)), np.cumprod(1 - hd - hp, axis=1)[:, :-1]])
        return np.cumsum(surv_prev * hd, axis=1)

    def baseline_table(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(baseline, covariates): band and covariate coefficients for default and prepay, as log
        relative hazards against continuing (the baseline is at x = 0, i.e. all WOE at zero)."""
        k = len(self.bands)
        # sklearn's multinomial fit is the symmetric softmax (one coefficient row per class), so the
        # log relative hazard against continuing is the difference from the CONTINUE row.
        raw = dict(zip(self.classes_, self.model.coef_))
        coef = {c: raw[c] - raw[CONTINUE] for c in raw}
        rows = []
        for i, (lo, hi) in enumerate(self.bands):
            rows.append({"months": f"{lo}-{hi}", "default_coef": coef[DEFAULT][i],
                         "prepay_coef": coef[PREPAY][i] if PREPAY in coef else np.nan})
        out = pd.DataFrame(rows)
        cov = pd.DataFrame({"covariate": self.covariates, "default_coef": coef[DEFAULT][k:],
                            "prepay_coef": coef[PREPAY][k:] if PREPAY in coef else np.nan})
        return out.round(4), cov.round(4)


def empirical_cif(df: pd.DataFrame, group: str | None = None, horizon: int | None = None) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence of default (and prepay) by month, overall or per ``group``.

    At each month: at_risk = loans with exit_month >= t; hazard = exits of that type at t / at_risk."""
    H = horizon or config.PERFORMANCE_WINDOW_MONTHS
    groups = [(None, df)] if group is None else list(df.groupby(group, observed=True))
    rows = []
    for g, d in groups:
        em, et = d["exit_month"].to_numpy(), d["exit_type"].to_numpy()
        surv, cif_d, cif_p = 1.0, 0.0, 0.0
        for t in range(1, H + 1):
            at_risk = int((em >= t).sum())
            if at_risk == 0:
                break
            nd = int(((em == t) & (et == "default")).sum()); npp = int(((em == t) & (et == "prepay")).sum())
            hd, hp = nd / at_risk, npp / at_risk
            cif_d += surv * hd; cif_p += surv * hp; surv *= (1 - hd - hp)
            rows.append({**({group: g} if group else {}), "month": t, "at_risk": at_risk, "defaults": nd, "prepays": npp,
                         "cif_default": cif_d, "cif_prepay": cif_p})
    return pd.DataFrame(rows)


def predicted_cif_by_group(cif: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Mean predicted CIF of default per month within each group (long format)."""
    df = pd.DataFrame(cif, index=groups.index, columns=range(1, cif.shape[1] + 1))
    df["_g"] = groups.to_numpy()
    long = df.groupby("_g", observed=True).mean().stack().rename("cif_default_pred").reset_index()
    return long.rename(columns={"_g": groups.name or "group", "level_1": "month"})
