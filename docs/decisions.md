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

## 2026-10-31 — Scope, structural fields and exclusions (NB01, NB02; decided by Siavash)

- **Training vintages 2013–2014** (`config.TRAIN_VINTAGES`). A block of 21 bureau fields is 100% missing for
  loans issued January–July 2012 and 83% missing in August 2012, against 0% from 2013. Keeping 2012 would
  turn the missing bin into a marker of issue date. Cost: 53,367 loans dropped from training. Starting in
  September 2012 was considered and not taken, because it would need a month-level start date.
- **Structural fields for NB03:** the 2016-only block (`all_util`, `il_util`, `inq_last_12m`, `max_bal_bc`,
  `mths_since_rcnt_il`, `open_il_24m`, `open_rv_24m`). `mths_since_last_record` stays a candidate: it is 81–88%
  missing in every year from 2013, so missing means "no public record", not a reporting gap.
- **Joint applications excluded** from every split (`config.EXCLUDED_APPLICATION_TYPES`, exclusion
  `joint_application`). There are none in train, 511 in validate and 8,789 in 2016, and the joint-application
  fields are out of scope for v1.
- **60-month loans kept,** and reported as a segment. **24-month window kept.** **Indeterminates excluded** from
  fitting, with NB08 to show the sensitivity with them counted as bad. **Full data**, with no prior correction.

## 2026-10-31 — Fix: missing utilisation was coded as 150%

`attributes.build_attributes` capped the percent fields with `where(x <= 150, 150)`, which also replaced
missing values with 150%, the riskiest value. Applicants with no bankcard (about 1% of train) were binned as
maxed-out on `bc_util` and `percent_bc_gt_75`, and `all_util` and `il_util` looked complete where they are
absent. The cap is now `clip(upper=150)`, which leaves missing values missing, and
`tests/test_attributes.py` covers it. This affected every earlier run, including the 5% smoke test. NB03–NB05
were re-run on full data after the fix.

## 2026-10-31 — No active bankcard: own bin and own reason code (NB04, NB05)

- **Definition.** "No active bankcard" means a total bankcard limit of 0 (or missing). `build_attributes` then sets
  `bc_open_to_buy`, `percent_bc_gt_75` and `bc_util` to missing. Lending Club had recorded
  `bc_open_to_buy = 0` instead of missing for 114 such loans in train, 273 in validate and 29 in 2016.
- **Size and risk.** 3,505 train loans (396 bads, 11.30% bad, against 9.58% for everyone else); 4,166 in validate
  (13.03%); 4,470 in 2016 (12.42%).
- **Own bin.** `bc_open_to_buy`, `percent_bc_gt_75` and `bc_util` always get a separate missing bin
  (`binning.fit_numeric(force_missing_bin=True)`). Before, the 5% share floor merged this group into the low
  open-to-buy and >75% bins. The 396 bads make the bin reliable, and standard WOE smoothing applies.
- **Reason code.** A missing value in `bc_open_to_buy` or `percent_bc_gt_75` gives "No active bankcard on file",
  never a utilisation reason, and both together count as one reason with their points lost added together
  (`attributes.MISSING_REASONS`, `scorecard.reason_codes`). On 2016, all 4,470 such applicants get it exactly
  once and none gets a bankcard-utilisation reason. `bc_util` is not mapped, because it is also missing for 225
  train applicants who have a bankcard limit.
- **Effect on the card.** None of substance: the same 8 attributes, coefficients within 0.0001, OOT Gini
  0.2838 unchanged.

## 2026-11-01 — Stage 3 follow-ups (NB07b, NB08, NB09); decision cells still DRAFT

- **Swap set at a matched approval rate.** `validation.swap_set` approves exactly k loans under each score; ties at
  the cutoff (FICO moves in 5-point steps) are broken by one seeded random key shared by both scores. Before, FICO
  approved 72.9% at a "70%" cutoff. The report states that the cutoff is applied to loans Lending Club already
  approved, so the swap set is a reordering within the booked population.
- **Deployed card.** The card's intercept is recalibrated on 2015 by prior correction
  (`target.prior_correct_intercept`, +0.198 log-odds). NB08 shows raw and recalibrated calibration for the card and
  the tuned GBM side by side; NB09 monitors the recalibrated card and reports the development-to-deployment shift
  once (2016 calibration-in-the-large 0.262 as developed, 0.064 deployed).
- **Monitoring rule** (`monitoring.decide_action`, `decide_periods`). Each period's Gini is judged against its own
  DeLong interval and against the Gini the card expects on that period's population if its PDs were true
  (`monitoring.expected_gini`: expected concordant pairs over expected pairs, deterministic), with interval
  expected ± 1.96 × the observed Gini's DeLong SE. A breach (below reference and below expected by more than half
  that width) gives watch on its first quarter and re-estimate (redevelop with CSI > 0.25) on the second
  consecutive one; at or above expected is a population shift (investigate, no model change). Result: 2016Q3
  watch, 2016Q4 and 2017Q1 re-estimate.
- **Covariate-shift simulation** now draws half the loans without replacement, tilted towards the attributes, so
  P(Y|X) is unchanged; the old version shifted recorded attributes while keeping outcomes.
- **NB07b reproducibility.** The XGBoost search is pinned to the 23 trials the first full run completed and the
  wall-clock caps are off; the re-run reproduced the tuned model byte for byte. Paired DeLong, tuned XGBoost
  minus monotone EBM: +0.0064 Gini on 2015 (95% CI 0.0055 to 0.0074), +0.0058 on 2016 (0.0048 to 0.0068).
