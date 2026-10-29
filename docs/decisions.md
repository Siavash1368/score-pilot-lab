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
