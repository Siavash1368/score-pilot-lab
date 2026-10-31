"""Attribute library (NB03): cleaning, derived ratios, missing-value semantics, leakage control.

Inputs are the as-of-application bureau fields in ``v_accepted_asof`` (the keep-list in
``data/reference/as_of_application_fields.csv``). This module turns them into the candidate
attributes the scorecard may use, with one registry row per attribute: family, business meaning,
the expected direction of risk (used later as the monotone constraint) and how a missing value
is read.

Rules enforced here:

* only keep-list columns with role ``input`` may feed an attribute (:func:`leakage_check`);
* every derived attribute is a deterministic function of those columns and ``issue_d``;
* missing is kept as missing (its own bin in NB04, native handling in the GBM); the registry
  says what the absence means so the bin can be read.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scorelab import config

# Expected direction of risk as the attribute increases: +1 riskier, -1 safer, 0 no prior.
RISKIER, SAFER, NO_PRIOR = 1, -1, 0


@dataclass(frozen=True)
class Attribute:
    name: str
    family: str
    meaning: str
    direction: int
    missing_means: str
    sources: tuple[str, ...]
    derived: bool = False
    available_from: int = 2012     # first vintage year where the field is populated
    notes: str = ""


def _a(name, family, meaning, direction, missing_means, sources=None, derived=False,
       available_from=2012, notes=""):
    return Attribute(name, family, meaning, direction, missing_means,
                     tuple(sources or (name,)), derived, available_from, notes)


# --- Registry -------------------------------------------------------------------
# Raw bureau fields kept as attributes, plus derived ones at the end. Direction follows
# standard bureau logic (Siddiqi App. A); NB04 tests it and NB07 enforces it.
REGISTRY: list[Attribute] = [
    # inquiries
    _a("inq_last_6mths", "inquiries", "Hard inquiries in the last 6 months", RISKIER, "no inquiries recorded"),
    _a("inq_last_12m", "inquiries", "Hard inquiries in the last 12 months", RISKIER, "field not reported", available_from=2016,
       notes="populated only for late vintages; check CSI before use"),
    _a("mths_since_recent_inq", "inquiries", "Months since the most recent inquiry", SAFER, "never inquired"),
    # delinquency history
    _a("delinq_2yrs", "delinquency", "30+ DPD incidences in the last 2 years", RISKIER, "none"),
    _a("mths_since_last_delinq", "delinquency", "Months since the last delinquency", SAFER, "never delinquent"),
    _a("mths_since_last_major_derog", "delinquency", "Months since the last 90+ DPD or worse", SAFER, "never 90+"),
    _a("mths_since_recent_bc_dlq", "delinquency", "Months since the last bankcard delinquency", SAFER, "never delinquent on a bankcard"),
    _a("mths_since_recent_revol_delinq", "delinquency", "Months since the last revolving delinquency", SAFER, "never delinquent on revolving"),
    _a("num_tl_90g_dpd_24m", "delinquency", "Trades 90+ DPD in the last 24 months", RISKIER, "none"),
    _a("num_tl_30dpd", "delinquency", "Trades currently 30 DPD", RISKIER, "none"),
    _a("num_tl_120dpd_2m", "delinquency", "Trades 120+ DPD in the last 2 months", RISKIER, "none"),
    _a("acc_now_delinq", "delinquency", "Accounts currently delinquent", RISKIER, "none"),
    _a("pct_tl_nvr_dlq", "delinquency", "Percent of trades never delinquent", SAFER, "no trades"),
    # derogatory / public records
    _a("pub_rec", "derogatory", "Derogatory public records", RISKIER, "none"),
    _a("pub_rec_bankruptcies", "derogatory", "Public-record bankruptcies", RISKIER, "none"),
    _a("tax_liens", "derogatory", "Tax liens", RISKIER, "none"),
    _a("mths_since_last_record", "derogatory", "Months since the last public record", SAFER, "no public record"),
    _a("collections_12_mths_ex_med", "derogatory", "Collections in 12 months excluding medical", RISKIER, "none"),
    _a("chargeoff_within_12_mths", "derogatory", "Charge-offs in the last 12 months", RISKIER, "none"),
    _a("tot_coll_amt", "derogatory", "Total collection amount ever owed", RISKIER, "none"),
    # tradelines and file depth
    _a("open_acc", "tradelines", "Open credit lines", NO_PRIOR, "none"),
    _a("total_acc", "tradelines", "Total credit lines ever", SAFER, "none"),
    _a("acc_open_past_24mths", "tradelines", "Trades opened in the last 24 months", RISKIER, "none"),
    _a("num_tl_op_past_12m", "tradelines", "Trades opened in the last 12 months", RISKIER, "none"),
    _a("num_actv_bc_tl", "tradelines", "Active bankcard trades", NO_PRIOR, "none"),
    _a("num_bc_sats", "tradelines", "Satisfactory bankcard accounts", SAFER, "none"),
    _a("num_bc_tl", "tradelines", "Bankcard trades ever", SAFER, "none"),
    _a("num_il_tl", "tradelines", "Installment trades ever", NO_PRIOR, "none"),
    _a("num_rev_accts", "tradelines", "Revolving accounts ever", SAFER, "none"),
    _a("num_rev_tl_bal_gt_0", "tradelines", "Revolving trades with a balance", RISKIER, "none"),
    _a("num_sats", "tradelines", "Satisfactory accounts", SAFER, "none"),
    _a("mort_acc", "tradelines", "Mortgage accounts", SAFER, "none"),
    _a("open_rv_24m", "tradelines", "Revolving trades opened in 24 months", RISKIER, "field not reported", available_from=2016),
    _a("open_il_24m", "tradelines", "Installment trades opened in 24 months", RISKIER, "field not reported", available_from=2016),
    # balances and utilisation
    _a("revol_util", "balances_utilisation", "Revolving utilisation, percent", RISKIER, "no revolving limit"),
    _a("bc_util", "balances_utilisation", "Bankcard utilisation, percent", RISKIER, "no bankcard limit"),
    _a("all_util", "balances_utilisation", "Utilisation across all trades, percent", RISKIER, "field not reported", available_from=2016),
    _a("il_util", "balances_utilisation", "Installment utilisation, percent", RISKIER, "no installment limit / not reported", available_from=2016),
    _a("percent_bc_gt_75", "balances_utilisation", "Percent of bankcards above 75% utilised", RISKIER, "no bankcards"),
    _a("bc_open_to_buy", "balances_utilisation", "Open-to-buy on bankcards, dollars", SAFER, "no bankcards"),
    _a("revol_bal", "balances_utilisation", "Revolving balance, dollars", NO_PRIOR, "none"),
    _a("total_bal_ex_mort", "balances_utilisation", "Total balance excluding mortgage", NO_PRIOR, "none"),
    _a("tot_cur_bal", "balances_utilisation", "Total current balance", NO_PRIOR, "none"),
    _a("avg_cur_bal", "balances_utilisation", "Average current balance per account", SAFER, "no accounts"),
    _a("max_bal_bc", "balances_utilisation", "Maximum bankcard balance", NO_PRIOR, "no bankcards"),
    _a("total_bc_limit", "balances_utilisation", "Total bankcard limit", SAFER, "no bankcards"),
    _a("total_rev_hi_lim", "balances_utilisation", "Total revolving high credit / limit", SAFER, "none"),
    _a("tot_hi_cred_lim", "balances_utilisation", "Total high credit / limit", SAFER, "none"),
    # credit age
    _a("mo_sin_old_rev_tl_op", "credit_age", "Months since the oldest revolving trade opened", SAFER, "no revolving trade"),
    _a("mo_sin_old_il_acct", "credit_age", "Months since the oldest installment account opened", SAFER, "no installment account"),
    _a("mo_sin_rcnt_rev_tl_op", "credit_age", "Months since the most recent revolving trade opened", SAFER, "no revolving trade"),
    _a("mo_sin_rcnt_tl", "credit_age", "Months since the most recent trade opened", SAFER, "no trades"),
    _a("mths_since_rcnt_il", "credit_age", "Months since the most recent installment account", SAFER, "no installment account"),
    # derived
    _a("credit_age_months", "credit_age", "Months from the earliest credit line to issue", SAFER, "no credit line date",
       sources=("earliest_cr_line", "issue_d"), derived=True),
    _a("new_trades_share_24m", "tradelines", "Trades opened in 24 months as a share of all trades", RISKIER, "no trades",
       sources=("acc_open_past_24mths", "total_acc"), derived=True),
    _a("revol_bal_to_limit", "balances_utilisation", "Revolving balance over revolving limit (own utilisation)", RISKIER,
       "no revolving limit", sources=("revol_bal", "total_rev_hi_lim"), derived=True),
    _a("bal_ex_mort_to_limit", "balances_utilisation", "Non-mortgage balance over total limit", RISKIER, "no limit",
       sources=("total_bal_ex_mort", "tot_hi_cred_lim"), derived=True),
    _a("any_derog_flag", "derogatory", "Any public record, lien, bankruptcy or 12-month collection", RISKIER, "none",
       sources=("pub_rec", "tax_liens", "pub_rec_bankruptcies", "collections_12_mths_ex_med", "chargeoff_within_12_mths"),
       derived=True),
    _a("ever_delinquent_flag", "delinquency", "Any delinquency ever recorded", RISKIER, "none",
       sources=("mths_since_last_delinq", "delinq_2yrs", "num_tl_90g_dpd_24m"), derived=True),
]

REGISTRY_BY_NAME = {a.name: a for a in REGISTRY}
PERCENT_FIELDS = ("revol_util", "bc_util", "all_util", "il_util", "percent_bc_gt_75", "pct_tl_nvr_dlq")
PERCENT_CAP = 150.0     # utilisation above this is a reporting artefact (max in data ~900%)
# Adverse-action text when an attribute is missing for a reason of its own. A missing bankcard
# open-to-buy or bankcard-utilisation share means no active bankcard (NB04, 2026-10-31): the
# applicant must hear that, not a utilisation reason. Both map to one reason code.
MISSING_REASONS = {"bc_open_to_buy": "No active bankcard on file",
                   "percent_bc_gt_75": "No active bankcard on file"}
# bc_util is not mapped: it is also missing for a few hundred applicants who do have a bankcard limit,
# so "no active bankcard" would be untrue for them. It is not on the card.
# "No active bankcard" is defined by a zero (or missing) total bankcard limit. For those applicants the
# bankcard fields describe nothing, so build_attributes sets them to missing even where Lending Club
# recorded a 0 (about 1 in 30 of them had bc_open_to_buy = 0 instead of missing).
NO_BANKCARD_FIELDS = ("bc_open_to_buy", "percent_bc_gt_75", "bc_util")


def registry_table() -> pd.DataFrame:
    """The registry as a table for the notebook and the model document."""
    rows = [{"attribute": a.name, "family": a.family, "meaning": a.meaning,
             "direction": {1: "riskier", -1: "safer", 0: "no prior"}[a.direction],
             "missing_means": a.missing_means, "derived": a.derived,
             "available_from": a.available_from, "sources": ", ".join(a.sources)} for a in REGISTRY]
    return pd.DataFrame(rows)


# --- Leakage control --------------------------------------------------------------

def keep_list(path=None) -> pd.DataFrame:
    with open(path or config.AS_OF_FIELDS_CSV, newline="", encoding="utf-8") as f:
        return pd.DataFrame(list(csv.DictReader(f)))


def leakage_check(columns: list[str] | None = None, path=None) -> pd.DataFrame:
    """Assert that every source column of every attribute is a keep-list ``input`` field.

    Returns the offending rows (empty when clean) and raises if any attribute reads a field that
    is not on the keep-list or whose role is ``input_excluded`` (FICO) or ``not_input``.
    ``issue_d`` is allowed as the clock for derived attributes.
    """
    kl = keep_list(path).set_index("field")["role"].to_dict()
    cols = columns or sorted({c for a in REGISTRY for c in a.sources})
    rows = []
    for c in cols:
        role = kl.get(c, "NOT ON KEEP-LIST")
        ok = role == "input" or c == "issue_d"
        rows.append({"column": c, "role": role, "ok": ok})
    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    if len(bad):
        raise ValueError("Attributes read columns that are not permitted inputs:\n" + bad.to_string(index=False))
    return bad


# --- Building ----------------------------------------------------------------------

def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def build_attributes(frame: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Return ``frame`` with the attributes in ``names`` (default: all of the registry) appended.

    Cleaning applied: percent fields capped at ``PERCENT_CAP``; negative counts set to missing;
    derived attributes computed from their sources. Columns that are not attributes are left as
    they are, so the result can be passed straight to binning with ``attribute_names()`` as the list.
    """
    out = frame.copy()
    names = names or [a.name for a in REGISTRY]
    for f in PERCENT_FIELDS:
        if f in out:
            out[f] = out[f].clip(upper=PERCENT_CAP)       # caps the tail; missing stays missing
    if "total_bc_limit" in out:
        no_bankcard = out["total_bc_limit"].fillna(0) <= 0
        for f in NO_BANKCARD_FIELDS:
            if f in out:
                out.loc[no_bankcard, f] = np.nan
    count_like = [a.name for a in REGISTRY if not a.derived and a.family in ("inquiries", "delinquency", "derogatory", "tradelines")]
    for c in count_like:
        if c in out:
            out[c] = out[c].where(out[c] >= 0, np.nan)
    if "credit_age_months" in names:
        issue = pd.to_datetime(out["issue_d"])
        ecl = pd.to_datetime(out["earliest_cr_line"])
        out["credit_age_months"] = ((issue.dt.year - ecl.dt.year) * 12 + (issue.dt.month - ecl.dt.month)).astype("float")
        out.loc[out["credit_age_months"] < 0, "credit_age_months"] = np.nan
    if "new_trades_share_24m" in names:
        out["new_trades_share_24m"] = _ratio(out["acc_open_past_24mths"], out["total_acc"]).clip(0, 1)
    if "revol_bal_to_limit" in names:
        out["revol_bal_to_limit"] = _ratio(out["revol_bal"], out["total_rev_hi_lim"]).clip(0, PERCENT_CAP / 100)
    if "bal_ex_mort_to_limit" in names:
        out["bal_ex_mort_to_limit"] = _ratio(out["total_bal_ex_mort"], out["tot_hi_cred_lim"]).clip(0, PERCENT_CAP / 100)
    if "any_derog_flag" in names:
        parts = out[["pub_rec", "tax_liens", "pub_rec_bankruptcies", "collections_12_mths_ex_med", "chargeoff_within_12_mths"]]
        out["any_derog_flag"] = (parts.fillna(0) > 0).any(axis=1).astype(float)
    if "ever_delinquent_flag" in names:
        out["ever_delinquent_flag"] = ((out["mths_since_last_delinq"].notna()) | (out["delinq_2yrs"].fillna(0) > 0)
                                      | (out["num_tl_90g_dpd_24m"].fillna(0) > 0)).astype(float)
    return out


def attribute_names(min_available_from: int | None = None, families: list[str] | None = None) -> list[str]:
    """Registry names, optionally only those populated from ``min_available_from`` and in ``families``."""
    names = []
    for a in REGISTRY:
        if min_available_from is not None and a.available_from > min_available_from:
            continue
        if families and a.family not in families:
            continue
        names.append(a.name)
    return names


# --- Screening -----------------------------------------------------------------------

def availability_by_vintage(frame: pd.DataFrame, names: list[str], vintage_col: str = "vintage_year") -> pd.DataFrame:
    """Share populated (non-missing) per attribute and vintage: finds fields that only exist late."""
    g = frame.groupby(vintage_col)[names].apply(lambda d: d.notna().mean())
    return g.T.round(3)


def bucket_table(df: pd.DataFrame, name: str, edges: list[float], target: str = "bad") -> pd.DataFrame:
    """Count, share, bads and bad rate of ``target`` by bucket of ``name``, missing as its own row.

    ``edges`` are inclusive upper bounds of the finite buckets; values above the last edge form the
    final bucket. A plain readout for checking that an attribute is built and signed as expected."""
    x = df[name].astype(float)
    bounds = [-np.inf] + list(edges) + [np.inf]
    labels = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        labels.append(f"<= {hi:g}" if lo == -np.inf else f"> {lo:g}" if hi == np.inf else f"({lo:g}, {hi:g}]")
    b = pd.cut(x, bounds, labels=labels).astype(object).where(x.notna(), "Missing")
    g = df.groupby(b, sort=False)[target].agg(n="size", bads="sum")
    g = g.reindex([l for l in labels + ["Missing"] if l in g.index])
    g["share"] = (g.n / g.n.sum()).round(4)
    g["bad_rate"] = (g.bads / g.n).round(4)
    return g.rename_axis(name).reset_index()


def univariate_screen(frame: pd.DataFrame, names: list[str], target: str = "bad") -> pd.DataFrame:
    """Per attribute: missing rate, number of distinct values, and rank power as a single-variable
    AUC oriented so that > 0.5 means higher value = riskier (sign tells the empirical direction).
    Missing rows are excluded from the AUC, so read it with the missing rate.
    """
    from sklearn.metrics import roc_auc_score
    y = frame[target]
    rows = []
    for n in names:
        x = frame[n]
        m = x.notna() & y.notna()
        auc = np.nan
        if m.sum() > 100 and y[m].nunique() == 2 and x[m].nunique() > 1:
            auc = roc_auc_score(y[m], x[m])
        reg = REGISTRY_BY_NAME.get(n)
        rows.append({"attribute": n, "family": reg.family if reg else "", "missing_rate": round(1 - m.mean(), 4),
                     "n_distinct": int(x.nunique()), "auc_higher_is_riskier": round(auc, 4) if auc == auc else np.nan,
                     "gini_abs": round(abs(2 * auc - 1), 4) if auc == auc else np.nan,
                     "expected_direction": {1: "riskier", -1: "safer", 0: "no prior"}[reg.direction] if reg else "",
                     "empirical_direction": ("riskier" if auc > 0.5 else "safer") if auc == auc else ""})
    return pd.DataFrame(rows).sort_values("gini_abs", ascending=False).reset_index(drop=True)
