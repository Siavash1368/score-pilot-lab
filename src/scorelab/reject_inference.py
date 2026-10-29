"""Reject inference (NB06): hard cutoff, parceling, fuzzy augmentation, and the comparison against
hidden truth.

Lending Club's rejected-applications file carries only amount, date, title, FICO (``risk_score``),
DTI, zip, state and employment length, none of the bureau attributes the card uses, so the rejects
cannot be scored by the card. Two things are still possible and both are implemented here:

1. **Simulated reject region with hidden truth.** Treat the riskiest ``reject_rate`` share of the
   accepted population *by the incumbent score* as "declined", hide their outcomes, build the
   known-good-bad (KGB) card on the "approved" remainder, infer the declines by each method, refit,
   and compare every variant on the full through-the-door population where the truth is known.
   This is the only setting in which reject inference can be *validated* rather than assumed.
2. **Through-the-door profile.** Compare the FICO distribution of the real rejected applications
   with the accepted ones to show how far the booked population is from the applicant population,
   and to produce weights for a sensitivity run.

The caller supplies ``fit_fn(frame) -> scorer`` and ``score_fn(scorer, frame) -> p_bad`` so the
same code works with the logistic card or a GBM.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from scorelab.validation import auc_gini_ks


def simulate_reject_region(frame: pd.DataFrame, incumbent: str = "fico_mid", reject_rate: float = 0.20,
                           seed: int = 42) -> pd.DataFrame:
    """Mark the lowest ``reject_rate`` share by ``incumbent`` (ties broken at random) as declined.

    Returns a copy with ``declined`` (bool) and ``bad_hidden`` (the true outcome, kept for evaluation
    only) and ``bad`` set to NaN on declined rows."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    jitter = rng.uniform(0, 1e-6, len(out))
    rank = (out[incumbent] + jitter).rank(method="first")
    out["declined"] = rank <= reject_rate * len(out)
    out["bad_hidden"] = out["bad"]
    out.loc[out["declined"], "bad"] = np.nan
    return out


# --- Methods: each returns an augmented frame with ``bad`` filled and a ``weight`` column ----

def hard_cutoff(frame: pd.DataFrame, p_bad_declined: np.ndarray, cutoff: float) -> pd.DataFrame:
    """Declines with KGB-predicted p_bad above ``cutoff`` are labelled bad, the rest good."""
    out = frame.copy(); out["weight"] = 1.0
    d = out["declined"].to_numpy()
    out.loc[d, "bad"] = (p_bad_declined > cutoff).astype(float)
    out["method"] = "hard_cutoff"
    return out


def parceling(frame: pd.DataFrame, p_bad_declined: np.ndarray, n_bands: int = 10, seed: int = 42,
              bad_rate_multiplier: float = 1.0) -> pd.DataFrame:
    """Within score bands of the KGB prediction, label declines bad at random in proportion to the
    band's predicted bad rate (times ``bad_rate_multiplier``, Siddiqi's adjustment for declines
    being worse than the approved at the same score)."""
    rng = np.random.default_rng(seed)
    out = frame.copy(); out["weight"] = 1.0
    d = out["declined"].to_numpy()
    p = np.asarray(p_bad_declined, dtype=float)
    bands = pd.qcut(pd.Series(p).rank(method="first"), n_bands, labels=False).to_numpy()
    lab = np.zeros(len(p))
    for b in range(n_bands):
        m = bands == b
        rate = min(1.0, p[m].mean() * bad_rate_multiplier)
        lab[m] = (rng.uniform(size=m.sum()) < rate).astype(float)
    out.loc[d, "bad"] = lab
    out["method"] = "parceling"
    return out


def fuzzy_augmentation(frame: pd.DataFrame, p_bad_declined: np.ndarray) -> pd.DataFrame:
    """Each decline enters twice: as a bad with weight p and as a good with weight 1 - p."""
    approved = frame[~frame["declined"]].copy(); approved["weight"] = 1.0
    dec = frame[frame["declined"]].copy()
    p = np.asarray(p_bad_declined, dtype=float)
    as_bad = dec.copy(); as_bad["bad"] = 1.0; as_bad["weight"] = p
    as_good = dec.copy(); as_good["bad"] = 0.0; as_good["weight"] = 1 - p
    out = pd.concat([approved, as_bad, as_good], ignore_index=True)
    out["method"] = "fuzzy_augmentation"
    return out


# --- Evaluation against hidden truth -------------------------------------------------------

def compare_methods(sim: pd.DataFrame, fit_fn: Callable[[pd.DataFrame], object],
                    score_fn: Callable[[object, pd.DataFrame], np.ndarray], eval_frame: pd.DataFrame,
                    cutoff: float | None = None, multipliers=(1.0, 1.5)) -> pd.DataFrame:
    """Fit KGB on approved only, then each augmentation; score ``eval_frame`` (truth known) and the
    simulated through-the-door population; report Gini on all, on the approved, on the declined.

    ``fit_fn`` must accept a frame with columns ``bad`` and ``weight`` (weight may be absent)."""
    approved = sim[~sim["declined"]].copy(); approved["weight"] = 1.0
    kgb = fit_fn(approved)
    p_dec = score_fn(kgb, sim[sim["declined"]])
    cutoff = cutoff if cutoff is not None else float(np.quantile(score_fn(kgb, approved), 0.90))
    variants = {"kgb_only": approved, "hard_cutoff": hard_cutoff(sim, p_dec, cutoff)}
    for m in multipliers:
        variants[f"parceling_x{m:g}"] = parceling(sim, p_dec, bad_rate_multiplier=m)
    variants["fuzzy_augmentation"] = fuzzy_augmentation(sim, p_dec)
    rows = []
    truth = sim["bad_hidden"].to_numpy(dtype=float)
    for name, aug in variants.items():
        model = kgb if name == "kgb_only" else fit_fn(aug)
        p_all = score_fn(model, sim); p_eval = score_fn(model, eval_frame)
        d = sim["declined"].to_numpy()
        rows.append({"method": name,
                     "inferred_bad_rate_declined": float(np.nan) if name == "kgb_only" else
                     float(np.average(aug.loc[aug["declined"], "bad"], weights=aug.loc[aug["declined"], "weight"])),
                     "true_bad_rate_declined": float(truth[d].mean()),
                     "gini_through_the_door": auc_gini_ks(truth, p_all, higher_is_safer=False)["gini"],
                     "gini_approved_only": auc_gini_ks(truth[~d], p_all[~d], higher_is_safer=False)["gini"],
                     "gini_declined_only": auc_gini_ks(truth[d], p_all[d], higher_is_safer=False)["gini"],
                     "gini_eval_set": auc_gini_ks(eval_frame["bad"], p_eval, higher_is_safer=False)["gini"]})
    return pd.DataFrame(rows).round(4)


def swap_vs_kgb(sim: pd.DataFrame, p_kgb: np.ndarray, p_aug: np.ndarray, approval_rate: float = 0.7) -> pd.DataFrame:
    """How many through-the-door applicants change decision between the KGB and the augmented card."""
    k = int(round(approval_rate * len(sim)))
    a1 = p_kgb <= np.sort(p_kgb)[k - 1]; a2 = p_aug <= np.sort(p_aug)[k - 1]
    truth = sim["bad_hidden"].to_numpy(dtype=float)
    return pd.DataFrame({"n": [int((a1 & a2).sum()), int((a2 & ~a1).sum()), int((a1 & ~a2).sum()), int((~a1 & ~a2).sum())],
                         "true_bad_rate": [truth[a1 & a2].mean() if (a1 & a2).any() else np.nan,
                                           truth[a2 & ~a1].mean() if (a2 & ~a1).any() else np.nan,
                                           truth[a1 & ~a2].mean() if (a1 & ~a2).any() else np.nan,
                                           truth[~a1 & ~a2].mean() if (~a1 & ~a2).any() else np.nan]},
                        index=["approved_by_both", "swap_in_augmented", "swap_out_augmented", "declined_by_both"]).round(4)


# --- The real rejected file --------------------------------------------------------------

def through_the_door_profile(con, years: tuple[int, ...]) -> pd.DataFrame:
    """FICO (risk_score) distribution of real rejected applications vs the FICO midpoint of accepted
    loans, by 20-point band, for the given application years."""
    yrs = ", ".join(str(y) for y in years)
    return con.sql(f"""
        WITH r AS (
            SELECT 'rejected' AS pop, CAST(floor(risk_score / 20) * 20 AS INTEGER) AS fico_band
            FROM rejected WHERE year(application_date) IN ({yrs}) AND risk_score IS NOT NULL AND risk_score > 0
        ), a AS (
            SELECT 'accepted' AS pop, CAST(floor((fico_range_low + fico_range_high) / 2 / 20) * 20 AS INTEGER) AS fico_band
            FROM accepted WHERE year(issue_d) IN ({yrs})
        ), u AS (SELECT * FROM r UNION ALL SELECT * FROM a)
        SELECT fico_band,
               SUM(CASE WHEN pop = 'rejected' THEN 1 ELSE 0 END) AS n_rejected,
               SUM(CASE WHEN pop = 'accepted' THEN 1 ELSE 0 END) AS n_accepted
        FROM u GROUP BY 1 ORDER BY 1
    """).df().assign(share_rejected=lambda d: (d.n_rejected / d.n_rejected.sum()).round(4),
                     share_accepted=lambda d: (d.n_accepted / d.n_accepted.sum()).round(4),
                     acceptance_rate=lambda d: (d.n_accepted / (d.n_accepted + d.n_rejected)).round(4))
