# Decisions log

Dated log of design decisions.

## 2026-10-25 — Exclusive vintage split

`v_split` labels are exclusive: train 2012–2014, validate 2015, oot = all of 2016, monitor
2017Q1–2018Q4, pre-2012 vintages = NULL (unused). The spec's monitor range began at 2016Q3, which
overlapped the OOT year; `config.MONITOR_VINTAGES` is now `("2017Q1", "2018Q4")`.

## 2026-10-25 — Raw data checksums (first download)

Source: Kaggle `wordsforthewise/lending-club`.

| File | Bytes | SHA-256 |
|---|---:|---|
| accepted_2007_to_2018Q4.csv | 1,675,133,810 | `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a` |
| rejected_2007_to_2018Q4.csv | 1,782,281,620 | `07eb8468d55340d8ca4145c3e3c2e2d3e25ff83c44e432a825729ee6c99c4d45` |

## 2026-10-27 — Target definition, monitoring indicator and incumbent (decided by Siavash, implemented in `scorelab.target`)

- **Performance snapshot is 2019-03-01**, not the last issue month: the newest `last_pymnt_d` in the
  accepted file is 2019-03-01. `config.PERFORMANCE_SNAPSHOT` drives `observable`; `config.DATA_CUT`
  stays the last issue month.
- **Bad definition, 90+ style with indeterminates.** Lending Club publishes no payment history, so
  the month a loan reached 90+ DPD is reconstructed from `last_pymnt_d`: first missed due date =
  last payment + 1 month, 90 DPD = first miss + 3 (60 DPD = first miss + 2). Bad = 90+ DPD inside the
  24-month window for `Charged Off` / `Default` and for open delinquencies at the snapshot;
  indeterminate = first miss inside the window but 90+ after it (excluded from fitting, reported as
  a sensitivity in NB08); good = everything else observable, including loans that only became
  delinquent after the window. `Late (31-120 days)` is no longer a bad status by itself.
  On the 2012–2016 vintages this gives a 11.05% bad rate (excluding indeterminates), a 1.5%
  indeterminate share, and captures 64.8% of the charge-offs observed at the snapshot.
- **Monitor vintages (2017Q1–2018Q4) are censored at 24 months.** They are monitored with PSI/CSI
  and an early-performance indicator, 60+ DPD within 12 months (`early_bad`, `early_observable`),
  rather than by shortening the window.
- **Incumbent score for the pilot is the FICO midpoint**, `(fico_range_low + fico_range_high) / 2`,
  exposed as `fico_mid` by `target.modeling_frame`. FICO fields remain excluded from model inputs.
- **Proxy-target caveat for the pilot report:** VantageScore's definition observes 90+ DPD on any
  tradeline over 24 months from a bureau archive; this project observes a single Lending Club loan
  and reconstructs the delinquency date from the last payment. The report states the mismatch.
- Both views are created by `scorelab.target.create_views`; `scripts/refresh_views.py` recreates
  them on an existing database after a config change.

## 2026-10-27 — Part 1 library and notebooks built (Claude); decisions left to Siavash

- `scorelab` modules `target, attributes, binning, scorecard, reject_inference, challengers, validation, fairness,
  report, monitoring` implemented and tested (33 tests, synthetic data; every notebook executed on a 5% sample).
- Binning is a self-contained monotone classer (quantile pre-bins, pooled adjacent violators, population and
  bad-count minimums, least-IV-loss merging, missing bin when it earns one); `optbinning` stays optional.
- SHAP for the GBM uses the boosters' native TreeSHAP (`pred_contribs`), avoiding the `shap`/`xgboost` version
  coupling; the `shap` package is used for plots only.
- Open decisions are marked in the notebooks' decision cells; defaults are placeholders, not recommendations:
  scope and the 2012 reporting-start block (NB01), bad event and indeterminates (NB02, implemented per the
  2026-10-27 entry above), candidate attributes (NB03), bins and the CSI screen (NB04), selection rules (NB05),
  reject-inference method (NB06), thresholds carried back from the GBM (NB07), approval rate, segments, fairness
  proxy, interpretations and caveats (NB08), monitoring thresholds and plan (NB09).

## 2026-10-30 — Extended modelling (NB07b–07e, NB06 self-training)

- **Scope.** Built: challenger zoo (RF, EBM, MLP, stacked ensemble alongside XGBoost/LightGBM) with Optuna
  tuning, explainability and interaction carry-back, a discrete-time survival model, self-training reject
  inference, and an illustrative DML estimate. Not attempted, because Lending Club cannot support them: an
  account-management (behaviour) model, which needs monthly account performance, and a real fair-lending LDA
  search, which needs protected-class data or a validated proxy. Both are covered in a separate module.
- **No leakage into 2016.** Every Optuna trial, the MLP's early stopping and the stack's meta-model use the
  2015 validation vintage only. 2016 is used once, in the NB07b comparison table.
- **Packages.** `optuna` and `interpret-core` approved 2026-10-30 and installed in the project `.venv` (optuna
  5.0.0, interpret 0.7.8), which is the environment JupyterLab uses. No PyTorch: the MLP is sklearn's
  `MLPClassifier`. Both packages are listed under the `extended` extra in `pyproject.toml`.
- **Input sets are held fixed so model effects and input effects can be told apart.** The GBMs, random
  forest, EBMs and one MLP all use the same 29 NB03 candidates. A second MLP uses the 10 NB04 WOE attributes.
  The comparison table has an `inputs` column. Raw MLP inputs are median-imputed, with one missing-indicator
  column per attribute that has missing values, then standardised. All of this is fitted on 2012–14 only
  (no zero-fill or mean-fill). An attribute missing at scoring time but never missing in training raises a
  warning instead of being imputed silently.
- **EBM.** With monotone constraints, interpret fits main effects only (pairwise terms are dropped), so
  NB07b fits two EBMs: monotone with main effects, and unconstrained with 10 pairs.
- **Survival.** A multinomial discrete-time hazard with three outcomes (continue, default, prepay), with
  continue as the reference; default and prepay log-hazards are reported against continue. Prepayment is a
  competing risk, not censoring, so that CIF(24) is
  the probability of the label event and compares directly with the card's PD. Event timing uses the same
  last-payment reconstruction as the label; `exit_type == 'default'` exactly when `label == 'bad'`.
- **Self-training.** Each round pseudo-labels a quarter of the declines from the confident tails of the
  current model. The bad share of each round equals the model's mean predicted PD, not a 0.5 threshold,
  which at a ~10% bad rate would label every decline good.
- **DML controls.** Bureau candidates plus amount, income, DTI, employment length, purpose and home
  ownership. Rate, grade and instalment are excluded as post-treatment (the lender prices the chosen term).

## 2026-10-30 — Reproducible samples and row order

- **Sampling by id hash.** `SAMPLE_FRAC` and the 07e causal sample used DuckDB's `USING SAMPLE (bernoulli, seed)`,
  which is only repeatable on a single thread. After a join, it gave a different set of rows on different
  runs. Rows are now kept when md5("<id>:<seed>") mod 10^6 < frac × 10^6 (`target.sample_predicate`). The result
  depends only on the id, and a smaller sample is always nested in a larger one with the same seed.
- **Fixed row order.** `modeling_frame` and `causal_frame` now `ORDER BY id`. Without it, multi-threaded queries
  returned the same rows in a different order, which changed every model that subsamples, bootstraps or
  shuffles rows (XGBoost, LightGBM, random forest, EBM, MLP, DML folds). The GLM scorecard was unaffected.
  Two consecutive NB07b runs now give byte-identical tuned models and identical comparison tables.
- **Consequence.** The 5% smoke sample changed with this fix, so smoke numbers from before 2026-10-30 evening
  are not comparable with later ones.
