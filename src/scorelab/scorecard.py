"""Scorecard fit, scaling to points, scoring and reason codes (NB05).

The card is a logistic regression of ``bad`` on WOE-coded attributes. With WOE = ln(good share /
bad share), a correctly behaving attribute has a *negative* coefficient (higher WOE = safer), and a
naive-Bayes card would have every coefficient at -1; the fitted coefficients re-weight attributes
for their dependence. Scaling follows the PDO convention in :mod:`scorelab.config`:

    odds = good:bad = exp(-eta),  eta = alpha + sum_k beta_k WOE_k
    Score = Offset + Factor * ln(odds) = Offset - Factor * eta
    Factor = PDO / ln 2,  Offset = BASE_SCORE - Factor * ln(BASE_ODDS)

Points for bin j of attribute k:  -Factor * beta_k * WOE_kj  -  (Factor * alpha - Offset) / n_attributes,
so the score is the sum of the points of the bins a consumer falls in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scorelab import config
from scorelab.binning import BinSpec, bin_index, transform_woe, load_specs, save_specs

WOE_SUFFIX = "_woe"


@dataclass
class Scorecard:
    attributes: list[str]
    intercept: float
    coef: dict[str, float]                    # attribute -> coefficient on its WOE
    std_err: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    base_score: int = config.BASE_SCORE
    base_odds: float = config.BASE_ODDS
    pdo: int = config.PDO
    n_train: int = 0
    bad_rate_train: float = float("nan")
    deviance: float = float("nan")
    null_deviance: float = float("nan")
    penalty: float | None = None

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    @property
    def pseudo_r2(self) -> float:
        return 1 - self.deviance / self.null_deviance if self.null_deviance == self.null_deviance else float("nan")

    def logit(self, woe: pd.DataFrame) -> np.ndarray:
        """eta = log-odds of bad, from WOE-coded columns ``<attr>_woe``."""
        eta = np.full(len(woe), self.intercept, dtype=float)
        for a in self.attributes:
            eta += self.coef[a] * woe[a + WOE_SUFFIX].to_numpy(dtype=float)
        return eta

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, float) or v == v else None) for k, v in self.__dict__.items()}


# --- Fitting -----------------------------------------------------------------------------

def fit_logistic(woe: pd.DataFrame, y: pd.Series, attributes: list[str], penalty: float | None = None,
                 sample_weight: pd.Series | None = None) -> Scorecard:
    """Fit the card on WOE columns. ``penalty`` > 0 adds an L2 ridge (no standard errors then)."""
    X = sm.add_constant(woe[[a + WOE_SUFFIX for a in attributes]].astype(float), has_constant="add")
    yv = y.astype(float)
    if penalty:
        model = sm.GLM(yv, X, family=sm.families.Binomial(), freq_weights=sample_weight)
        alpha = np.full(X.shape[1], float(penalty)); alpha[0] = 0.0
        res = model.fit_regularized(alpha=alpha, L1_wt=0.0)
        params = pd.Series(res.params, index=X.columns)
        se, pv = {}, {}
        dev = _deviance(yv.to_numpy(), model.predict(params.to_numpy()))
    else:
        model = sm.GLM(yv, X, family=sm.families.Binomial(), freq_weights=sample_weight)
        res = model.fit()
        params = res.params
        se = {a: float(res.bse[a + WOE_SUFFIX]) for a in attributes}
        pv = {a: float(res.pvalues[a + WOE_SUFFIX]) for a in attributes}
        dev = float(res.deviance)
    p0 = yv.mean()
    null_dev = _deviance(yv.to_numpy(), np.full(len(yv), p0))
    return Scorecard(attributes=list(attributes), intercept=float(params["const"]),
                     coef={a: float(params[a + WOE_SUFFIX]) for a in attributes}, std_err=se, p_values=pv,
                     n_train=int(len(yv)), bad_rate_train=float(p0), deviance=float(dev), null_deviance=float(null_dev),
                     penalty=penalty)


def _deviance(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def stepwise_select(woe: pd.DataFrame, y: pd.Series, candidates: list[str], p_enter: float = 0.01,
                    max_corr: float = 0.70, max_attributes: int = 20, verbose: bool = False) -> tuple[list[str], pd.DataFrame]:
    """Forward selection in ``candidates`` order (pass them sorted by IV, strongest first).

    A candidate enters if its WOE is not correlated above ``max_corr`` with any attribute already in,
    its coefficient is negative (right sign) and significant at ``p_enter``, and every attribute already
    in keeps the right sign. Returns the selected list and a log of every decision.
    """
    selected, log = [], []
    W = woe
    for cand in candidates:
        if len(selected) >= max_attributes:
            log.append({"attribute": cand, "decision": "skipped", "reason": "max_attributes reached"}); continue
        col = cand + WOE_SUFFIX
        if W[col].std() == 0:
            log.append({"attribute": cand, "decision": "rejected", "reason": "constant WOE"}); continue
        if selected:
            corr = W[[s + WOE_SUFFIX for s in selected]].corrwith(W[col]).abs()
            if corr.max() > max_corr:
                log.append({"attribute": cand, "decision": "rejected", "reason": f"corr {corr.max():.2f} with {corr.idxmax().replace(WOE_SUFFIX, '')}"}); continue
        trial = selected + [cand]
        card = fit_logistic(W, y, trial)
        wrong = [a for a in trial if card.coef[a] >= 0]
        if wrong:
            log.append({"attribute": cand, "decision": "rejected", "reason": f"wrong sign on {wrong}"}); continue
        if card.p_values[cand] > p_enter:
            log.append({"attribute": cand, "decision": "rejected", "reason": f"p = {card.p_values[cand]:.3f}"}); continue
        selected = trial
        log.append({"attribute": cand, "decision": "entered", "reason": f"coef {card.coef[cand]:.3f}, p = {card.p_values[cand]:.2g}"})
        if verbose: print(log[-1])
    return selected, pd.DataFrame(log)


def vif_table(woe: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = sm.add_constant(woe[[a + WOE_SUFFIX for a in attributes]].astype(float))
    return pd.DataFrame({"attribute": attributes,
                         "vif": [round(variance_inflation_factor(X.values, i + 1), 2) for i in range(len(attributes))]})


def coefficient_table(card: Scorecard) -> pd.DataFrame:
    rows = [{"attribute": a, "coef": round(card.coef[a], 4), "std_err": round(card.std_err.get(a, float("nan")), 4),
             "p_value": card.p_values.get(a, float("nan")), "right_sign": card.coef[a] < 0} for a in card.attributes]
    return pd.DataFrame(rows)


# --- Scaling and scoring -------------------------------------------------------------------

def points_table(card: Scorecard, specs: dict[str, BinSpec]) -> pd.DataFrame:
    """One row per attribute bin with its points; the sum over a consumer's bins is the score."""
    n = len(card.attributes)
    base_share = (card.factor * card.intercept - card.offset) / n
    rows = []
    for a in card.attributes:
        s = specs[a]
        for b, (label, w, cnt, nb) in enumerate(zip(s.labels, s.woe, s.n, s.n_bad)):
            pts = -card.factor * card.coef[a] * w - base_share
            rows.append({"attribute": a, "bin": b, "label": label, "woe": round(w, 4), "points": round(pts, 1),
                         "dev_share": round(cnt / sum(s.n), 4), "dev_bad_rate": round(nb / cnt if cnt else 0, 4)})
    df = pd.DataFrame(rows)
    mx = df.groupby("attribute")["points"].transform("max")
    df["points_lost_vs_best"] = (mx - df["points"]).round(1)
    return df


def score(df: pd.DataFrame, card: Scorecard, specs: dict[str, BinSpec], clip: bool = True) -> pd.DataFrame:
    """Return eta (log-odds of bad), probability of bad, and the scaled score for every row."""
    woe = transform_woe(df, {a: specs[a] for a in card.attributes})
    eta = card.logit(woe)
    pb = 1 / (1 + np.exp(-eta))
    sc = card.offset - card.factor * eta
    if clip:
        sc = np.clip(sc, config.SCORE_MIN, config.SCORE_MAX)
    return pd.DataFrame({"eta": eta, "p_bad": pb, "score": np.round(sc).astype(int)}, index=df.index)


def reason_codes(df: pd.DataFrame, card: Scorecard, specs: dict[str, BinSpec], n_codes: int = config.N_REASON_CODES,
                 meanings: dict[str, str] | None = None) -> pd.DataFrame:
    """Top ``n_codes`` attributes per row by points lost against that attribute's best bin
    (the max-points method; exact for an additive card). ``meanings`` maps attribute -> plain words."""
    pts = points_table(card, specs)
    lookup = {(r.attribute, r.bin): (r.points_lost_vs_best, r.label) for r in pts.itertuples()}
    out = {f"reason_{i + 1}": [] for i in range(n_codes)}
    out.update({f"reason_{i + 1}_points_lost": [] for i in range(n_codes)})
    idx = {a: bin_index(df[a], specs[a]) for a in card.attributes}
    for r in range(len(df)):
        losses = []
        for a in card.attributes:
            lost, label = lookup[(a, int(idx[a][r]))]
            if lost > 0:
                losses.append((lost, a, label))
        losses.sort(reverse=True)
        for i in range(n_codes):
            if i < len(losses):
                lost, a, label = losses[i]
                text = (meanings or {}).get(a, a)
                out[f"reason_{i + 1}"].append(f"{text} [{label}]")
                out[f"reason_{i + 1}_points_lost"].append(round(lost, 1))
            else:
                out[f"reason_{i + 1}"].append(""); out[f"reason_{i + 1}_points_lost"].append(0.0)
    return pd.DataFrame(out, index=df.index)


def odds_table(scores: pd.Series, y: pd.Series, card: Scorecard, band: int = 20) -> pd.DataFrame:
    """Observed goods, bads and odds by score band against the odds the scaling implies."""
    lo = int(np.floor(scores.min() / band) * band); hi = int(np.ceil(scores.max() / band) * band)
    edges = np.arange(lo, hi + band, band)
    b = pd.cut(scores, edges, right=False)
    g = pd.DataFrame({"band": b, "y": y.astype(int)}).groupby("band", observed=True)["y"].agg(["count", "sum"])
    g.columns = ["n", "bads"]; g["goods"] = g.n - g.bads
    g["observed_bad_rate"] = (g.bads / g.n).round(4)
    g["observed_odds"] = (g.goods / g.bads.replace(0, np.nan)).round(2)
    mids = np.array([iv.left + band / 2 for iv in g.index])
    g["scaled_odds"] = np.round(np.exp((mids - card.offset) / card.factor), 2)
    g["scaled_bad_rate"] = np.round(1 / (1 + g["scaled_odds"]), 4)
    return g.reset_index()


# --- Persistence ---------------------------------------------------------------------------

def save_card(card: Scorecard, specs: dict[str, BinSpec], directory: str | Path) -> None:
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    (d / "scorecard.json").write_text(json.dumps(card.to_dict(), indent=1), encoding="utf-8")
    save_specs({a: specs[a] for a in card.attributes}, d / "bins.json")


def load_card(directory: str | Path) -> tuple[Scorecard, dict[str, BinSpec]]:
    d = Path(directory)
    payload = json.loads((d / "scorecard.json").read_text(encoding="utf-8"))
    for k in ("bad_rate_train", "deviance", "null_deviance"):
        if payload.get(k) is None: payload[k] = float("nan")
    return Scorecard(**payload), load_specs(d / "bins.json")
