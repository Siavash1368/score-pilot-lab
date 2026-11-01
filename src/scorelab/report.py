"""The lender-pilot report (NB08): one function that computes every exhibit from scored frames, and
one that writes them as markdown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scorelab import config
from scorelab.validation import (auc_gini_ks, delong_ci, delong_test, bootstrap_gini_diff, gains_table, swap_set,
                                 swap_curve, calibration_by_band, calibration_slope_intercept, hosmer_lemeshow,
                                 probability_scores, psi, gini_by_group)
from scorelab.fairness import fairness_summary


def pilot_exhibits(frame: pd.DataFrame, dev_scores, score_col: str = "score", eta_col: str = "eta", p_col: str = "p_bad",
                   incumbent_col: str = config.INCUMBENT_SCORE_NAME, bad_col: str = "bad",
                   approval_rate: float = config.APPROVAL_RATE_FOR_SWAP, segment_cols: tuple[str, ...] = ("term_months",),
                   group_col: str | None = None, n_boot: int = 300, level: str = "out-of-time") -> dict:
    """Every exhibit of the pilot on ``frame`` (already scored; outcomes known).

    ``dev_scores`` are the development-sample scores for the PSI. ``level`` is written on every
    exhibit so a number never travels without its validation level."""
    y = frame[bad_col].to_numpy(dtype=int); s = frame[score_col]; inc = frame[incumbent_col]
    out = {"level": level, "n": int(len(frame)), "n_bad": int(y.sum()), "bad_rate": float(y.mean()),
           "approval_rate": approval_rate, "incumbent": incumbent_col}
    out["discrimination"] = pd.DataFrame([
        {"score": "challenger", **{k: v for k, v in delong_ci(y, s).items() if k in ("gini", "gini_se")}, "ks": auc_gini_ks(y, s)["ks"]},
        {"score": "incumbent", **{k: v for k, v in delong_ci(y, inc).items() if k in ("gini", "gini_se")}, "ks": auc_gini_ks(y, inc)["ks"]}]).round(4)
    out["gini_lift"] = {**delong_test(y, s, inc), **{"bootstrap_ci": bootstrap_gini_diff(y, s, inc, n_boot=n_boot)["ci"]}}
    out["gains_challenger"] = gains_table(y, s); out["gains_incumbent"] = gains_table(y, inc)
    sw = swap_set(y, inc, s, approval_rate); out["swap_set"] = sw; out["swap_curve"] = swap_curve(y, inc, s)
    out["calibration_bands"] = calibration_by_band(y, frame[p_col], by_score=s, band_width=20)
    out["calibration_fit"] = calibration_slope_intercept(y, frame[eta_col])
    out["hosmer_lemeshow"] = hosmer_lemeshow(y, frame[p_col]); out["probability_scores"] = probability_scores(y, frame[p_col])
    ps = psi(dev_scores, s); out["psi"] = {k: v for k, v in ps.items() if k not in ("table", "edges")}; out["psi_table"] = ps["table"]
    out["segments"] = {c: gini_by_group(y, s, frame[c]).merge(gini_by_group(y, inc, frame[c]), on=["group", "n", "bad_rate"],
                                                                suffixes=("_challenger", "_incumbent")) for c in segment_cols if c in frame}
    if group_col and group_col in frame:
        out["fairness"] = fairness_summary(y, s, frame[eta_col], frame[group_col], approval_rate)
    return out


def indeterminate_sensitivity(labelled: pd.DataFrame, indeterminate: pd.DataFrame, score_cols: dict[str, str],
                              bad_col: str = "bad") -> pd.DataFrame:
    """Bad rate and ranking under the three treatments of indeterminate loans: excluded (the rule in force),
    counted as good, counted as bad. ``score_cols`` maps a display name -> a column where higher = safer.
    Returns one row per treatment with n, bads, bad rate, and Gini and KS for each score."""
    rows = []
    for name, ind in (("excluded (rule in force)", None), ("counted as good", 0), ("counted as bad", 1)):
        d = labelled if ind is None else pd.concat([labelled, indeterminate.assign(**{bad_col: ind})])
        y = d[bad_col].to_numpy(dtype=int)
        row = {"treatment": name, "n": int(len(d)), "bads": int(y.sum()), "bad_rate": round(float(y.mean()), 4)}
        for label, col in score_cols.items():
            m = auc_gini_ks(y, d[col])
            row[f"gini_{label}"] = round(m["gini"], 4); row[f"ks_{label}"] = round(m["ks"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def assemble_pilot_report(ex: dict, title: str, lender: str, interpretation: dict[str, str] | None = None,
                          caveats: list[str] | None = None, path: str | Path | None = None,
                          incumbent_caveat: str | None = None, appendix: str | None = None,
                          extra_sections: dict[str, str] | None = None, swap_set_note: str | None = None) -> str:
    """Markdown pilot report from ``pilot_exhibits``. ``interpretation`` maps exhibit name -> your
    paragraph (written by a person, not generated). Every section states the validation level.

    ``incumbent_caveat`` is printed under every exhibit that compares the challenger with the incumbent
    (discrimination, gains, swap set, segments). ``extra_sections`` maps a heading -> markdown, placed
    after the calibration section; ``appendix`` is added at the end, before the caveats. ``swap_set_note`` is
    printed at the top of the swap-set section (e.g. which population the cutoff is applied to)."""
    i = interpretation or {}
    lv = ex["level"]; sw = ex["swap_set"]; gl = ex["gini_lift"]; cf = ex["calibration_fit"]
    note = [f"*Note on the incumbent: {incumbent_caveat}*", ""] if incumbent_caveat else []
    parts = [f"# {title}", "", f"*Lender: {lender} · Validation level: {lv} · n = {ex['n']:,} · bad rate = {ex['bad_rate']:.2%} · "
             f"incumbent = {ex['incumbent']} · approval rate for swap set = {ex['approval_rate']:.0%}*", ""]
    parts += ["## 1. Discrimination", "", _md_table(ex["discrimination"]), "",
              f"Gini lift challenger - incumbent: **{gl['gini_diff']:.4f}** (DeLong z = {gl['z']:.2f}, p = {gl['p_value']:.3g}; "
              f"bootstrap 95% CI {gl['bootstrap_ci'][0]:.4f} to {gl['bootstrap_ci'][1]:.4f}). Level: {lv}.", ""] + note + [i.get("discrimination", ""), ""]
    parts += ["## 2. Gains tables", "", "Challenger:", "", _md_table(ex["gains_challenger"]), "", "Incumbent:", "", _md_table(ex["gains_incumbent"]), ""] + note + [i.get("gains", ""), ""]
    parts += ["## 3. Swap set at matched approval rate", ""] + ([swap_set_note, ""] if swap_set_note else []) + [_md_table(sw["crosstab"].reset_index().rename(columns={"index": ""})), "",
              f"Swap-ins: {sw['n_swap_in']:,} loans, bad rate {sw['bad_rate_swap_in']:.2%}. Swap-outs: {sw['n_swap_out']:,} loans, bad rate "
              f"{sw['bad_rate_swap_out']:.2%}. Approved-book bad rate: incumbent {sw['bad_rate_approved_incumbent']:.2%}, challenger "
              f"{sw['bad_rate_approved_challenger']:.2%} ({sw['bads_avoided_per_10k_approved']:.0f} bads avoided per 10,000 approvals).", "",
              _md_table(ex["swap_curve"].drop(columns=["n"])), ""] + note + [i.get("swap_set", ""), ""]
    parts += ["## 4. Calibration", "", _md_table(ex["calibration_bands"]), "",
              f"Calibration slope {cf['slope']:.3f} (95% CI {cf['slope_ci'][0]:.3f} to {cf['slope_ci'][1]:.3f}), intercept {cf['intercept']:.3f}; "
              f"intercept-only shift {cf['intercept_only_shift']:.3f}; mean predicted {cf['mean_predicted']:.2%} vs observed {cf['mean_observed']:.2%}. "
              f"Hosmer-Lemeshow chi2 {ex['hosmer_lemeshow']['chi2']:.1f} (p = {ex['hosmer_lemeshow']['p_value']:.3g}). "
              f"Log-loss {ex['probability_scores']['log_loss']:.4f} vs null {ex['probability_scores']['null_log_loss']:.4f}; Brier {ex['probability_scores']['brier']:.4f} "
              f"(reliability {ex['probability_scores']['reliability']:.5f}, resolution {ex['probability_scores']['resolution']:.5f}).", "", i.get("calibration", ""), ""]
    for heading, body in (extra_sections or {}).items():
        parts += [f"## {heading}", "", body, ""]
    parts += ["## 5. Segment performance", ""]
    for c, t in ex["segments"].items():
        parts += [f"By `{c}`:", "", _md_table(t), ""]
    parts += note + [i.get("segments", ""), ""]
    parts += ["## 6. Stability", "", f"PSI of the pilot population's scores against development: **{ex['psi']['psi']:.4f}** "
              f"(verdict {ex['psi']['verdict']}; chi-square critical value at this n {ex['psi']['critical_psi_5pct']:.4f}).", "", _md_table(ex["psi_table"]), "", i.get("stability", ""), ""]
    if "fairness" in ex:
        f = ex["fairness"]
        parts += ["## 7. Fair lending", "", "Adverse impact ratio at the approval cutoff:", "", _md_table(f["air"]), "",
                  "Standardized mean difference of scores:", "", _md_table(f["smd"]), "", "Performance and calibration by group:", "", _md_table(f["performance"]), "", i.get("fairness", ""), ""]
    if appendix:
        parts += [appendix, ""]
    parts += ["## Caveats and assumptions", ""] + [f"- {c}" for c in (caveats or [])] + [""]
    md = "\n".join(parts)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(md, encoding="utf-8")
    return md


def challenger_appendix(name: str, ex: dict, note: str = "", incumbent_caveat: str | None = None,
                        swap_set_note: str | None = None) -> str:
    """A compact report section for a second challenger (e.g. a recalibrated GBM) computed with
    :func:`pilot_exhibits`: discrimination, swap set at the same approval rate, calibration."""
    sw = ex["swap_set"]; gl = ex["gini_lift"]; cf = ex["calibration_fit"]
    parts = [f"## Appendix: {name}", "", note, "", _md_table(ex["discrimination"]), "",
             f"Gini lift over the incumbent: **{gl['gini_diff']:.4f}** (DeLong p = {gl['p_value']:.3g}; bootstrap 95% CI "
             f"{gl['bootstrap_ci'][0]:.4f} to {gl['bootstrap_ci'][1]:.4f}). Level: {ex['level']}.", "",
             *([swap_set_note, ""] if swap_set_note else []),
             _md_table(sw["crosstab"].reset_index().rename(columns={"index": ""})), "",
             f"Swap-ins {sw['n_swap_in']:,} (bad rate {sw['bad_rate_swap_in']:.2%}); swap-outs {sw['n_swap_out']:,} "
             f"(bad rate {sw['bad_rate_swap_out']:.2%}); approved-book bad rate {sw['bad_rate_approved_challenger']:.2%} against "
             f"{sw['bad_rate_approved_incumbent']:.2%} for the incumbent.", "",
             _md_table(ex["calibration_bands"]), "",
             f"Calibration slope {cf['slope']:.3f}, intercept {cf['intercept']:.3f}, intercept-only shift {cf['intercept_only_shift']:.3f}; "
             f"mean predicted {cf['mean_predicted']:.2%} against observed {cf['mean_observed']:.2%}."]
    if incumbent_caveat:
        parts += ["", f"*Note on the incumbent: {incumbent_caveat}*"]
    return "\n".join(parts)


md_table = _md_table          # public name for notebooks that add their own sections
