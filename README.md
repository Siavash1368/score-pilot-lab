# score-pilot-lab

A bureau-attribute PD scorecard on Lending Club 2007–2018, built the way a score vendor develops and
pilots a score: inputs limited to bureau attributes known at application, a 24-month bad definition,
a vintage split with a true out-of-time test, a WOE logistic scorecard scaled to 300–850 with
adverse-action reason codes, reject inference, monotonic GBM challengers, a vendor-style pilot report
with swap sets against FICO, and PSI/CSI monitoring. The analytics are exposed as read-only MCP tools
driven by a Claude agent and checked by an evaluation harness.

## Quick start

```bash
pip install -e ".[dev]"
make data        # Kaggle download -> data/raw/, data dictionary -> data/reference/
make db          # CSVs -> data/lc.duckdb with typed tables and views
make views       # recreate v_target / v_split after a target change (no table rebuild)
make test
make dashboard   # Streamlit PSI/CSI (once notebook 09 exists)
```

Requirements: Python 3.11 or 3.12, a Kaggle API token (`~/.kaggle/kaggle.json` or
`~/.kaggle/access_token`), and roughly 8 GB of free disk for the raw CSVs plus the DuckDB file.
On Windows without GNU make, `mingw32-make` works with the same targets, or run the scripts directly
(`python scripts/download_data.py`, `python scripts/build_db.py`).

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` before using the agent. `RAW_DATA_DIR`
and `DUCKDB_PATH` in `.env` move the raw CSVs and the database out of the repo. Use them when the repo
sits in a synced folder (OneDrive, Dropbox). This checkout keeps them in
`C:/Users/siava/score-pilot-lab-data/` because the repo is on the OneDrive Desktop.
`data/reference/` always stays in the repo.

## Layout

| Path | What it holds |
|---|---|
| `data/raw/` | Kaggle CSVs `accepted_2007_to_2018Q4.csv`, `rejected_2007_to_2018Q4.csv` (git-ignored) |
| `data/lc.duckdb` | Built by `scripts/build_db.py` (git-ignored) |
| `data/reference/as_of_application_fields.csv` | Leakage-control keep-list: the only accepted-loan columns modelling may read |
| `data/reference/lc_data_dictionary.csv` | CSV copy of Kaggle's `LCDataDictionary.xlsx` |
| `docs/model_card.md` | Fixed card the agent reads: bad definition, window, thresholds, rules |
| `docs/model_document.md` | SR 11-7 model document, filled in as notebooks complete |
| `docs/pilot_report/` | Pilot white paper written by notebook 08 |
| `docs/decisions.md` | Dated log of design decisions (raw-file checksums are recorded here) |
| `notebooks/01`–`14` | One notebook per step; each starts with the same three setup cells |
| `scripts/` | `download_data.py` (Kaggle CLI + checksums), `build_db.py` (CSV → DuckDB) |
| `src/scorelab/` | All analytics (see **Modules** below). `config.py` holds paths and constants; `data.py` is the only DuckDB entry point |
| `models/` | Artifacts the notebooks hand to each other: `candidate_attributes.json` (NB03), `bins.json` (NB04), `champion/` (NB05), `challenger_xgb.json` (NB07). `_smoke_test_5pct/` holds an example run on a 5% sample, not a result |
| `mcp_server/server.py` | Six read-only MCP tools over `lc.duckdb` and saved models |
| `agent/` | Pilot analyst tool-use loop, validator (stretch), system prompts |
| `evals/` | JSON cases and `run_evals.py` (numeric faithfulness, reason codes, refusals) |
| `finetune/` | Reason-code → plain-language dataset and LoRA training |
| `dashboard/app.py` | Streamlit PSI/CSI dashboard |
| `tests/` | pytest suite |

**Rule:** notebooks stay thin and call `src/scorelab`. The MCP server and the agent import the same
functions, so every number in a pilot report and in the agent's output comes from one code path.

### DuckDB objects

| Object | Kind | Definition |
|---|---|---|
| `accepted` | table | All columns of the accepted file, explicitly typed. `Mon-YYYY` columns are `DATE` (first of month); `int_rate`, `revol_util` are `DOUBLE` percent; `emp_length` is `INTEGER` years |
| `rejected` | table | `amount_requested, application_date, loan_title, risk_score, dti, zip_code, state, emp_length, policy_code` |
| `v_accepted_asof` | view | `accepted` restricted to the keep-list in `data/reference/as_of_application_fields.csv`. Every notebook after 01 reads this, never `accepted` |
| `v_target` | view | One row per accepted loan: `vintage_year/quarter, term_months, loan_status, months_to_last_payment, exclusion, observable, bad, indeterminate, label, early_observable, early_bad`. 24-month 90+ DPD label reconstructed from `last_pymnt_d` (see `docs/decisions.md`, 2026-10-27); labels NULL where the window is not observable at the 2019-03-01 snapshot. Defined in `scorelab.target`; refresh with `python scripts/refresh_views.py` |
| `v_split` | view | `v_target` plus `split` ∈ {train 2012–14, validate 2015, oot 2016, monitor 2017Q1–2018Q4}; NULL for pre-2012 vintages |


## Modules (`src/scorelab`)

| Module | What it does | Notebook | Wrapped later as an MCP tool |
|---|---|---|---|
| `target` | `v_target` / `v_split` SQL (24-month 90+ DPD label, indeterminates, early indicator), vintage curves, window sensitivity, `modeling_frame`, prior correction | 02 | — |
| `data` | DuckDB connection and read-only loaders; profiling helpers (missingness by vintage, numeric and categorical profiles, incumbent coverage) | 01 | `sql_readonly` |
| `attributes` | Attribute registry (family, meaning, expected direction, missing semantics), `build_attributes`, `leakage_check`, availability and univariate screens | 03 | — |
| `binning` | Monotone coarse classing with population and bad-count minimums, WOE/IV, fixed-edge overrides, categorical grouping, CSI, save/load | 04 | — |
| `scorecard` | Logistic fit with SEs and deviance, stepwise selection with sign and correlation guards, PDO scaling, points table, scoring, reason codes, odds table, save/load | 05 | `score_applicant`, `reason_codes` |
| `reject_inference` | Simulated reject region with hidden truth; hard cutoff, parceling, fuzzy augmentation; comparison against truth; through-the-door profile from the rejected file | 06 | — |
| `challengers` | XGBoost / LightGBM with monotone and interaction constraints, native TreeSHAP, GBM reason codes, surrogate tree, split-to-attribute translation, interaction strength | 07 | — |
| `validation` | AUC/Gini/KS, DeLong CI and paired test, bootstrap, gains tables, swap sets, calibration by band and slope/intercept, Hosmer–Lemeshow, log-loss/Brier decomposition, PSI with chi-square reference | 08, 09 | `run_pilot` |
| `fairness` | Adverse impact ratio, standardized mean difference, performance and calibration by group, less-discriminatory-alternative search | 08 | `fairness_report` |
| `report` | `pilot_exhibits` (every pilot number from one call) and `assemble_pilot_report` (markdown) | 08 | `run_pilot` |
| `monitoring` | PSI/CSI by period with the n-aware test, early-indicator tracking, calibration by period, drift simulation, the recalibrate / re-estimate / redevelop rule | 09 | `psi_monitor` |

## Working through the notebooks

Each notebook (01–09) states what it settles, shows the evidence, and leaves **decision cells** (`_(write here)_`) for the modelling decisions and their reasoning. Run them in order; every notebook reads the artifacts of the previous ones from `models/`. Set `SAMPLE_FRAC` in cell 3 for a quick pass; leave it `None` for the fit that goes into the report.

| Notebook | Decisions you make | Writes |
|---|---|---|
| 01 Data audit | scope (years, terms, application types); fields with structural gaps | `models/audit_decisions.json`, `docs/audit/` |
| 02 Sample design | window, bad event and indeterminates, exclusions, split, sampling | `docs/decisions.md` entry (pasted) |
| 03 Attribute library | families and exclusions, derived attributes, candidate list | `models/candidate_attributes.json`, `docs/exhibits/` |
| 04 Classing & WOE | binning parameters, edge overrides, IV cut and CSI screen | `models/bins.json`, `models/binning_decisions.json` |
| 05 Scorecard | selection rules, forced in/out, penalty | `models/champion/`, `docs/exhibits/points_table.csv` |
| 06 Reject inference | simulation design, which method the pilot reports | `docs/exhibits/reject_inference_comparison.csv` |
| 07 Challengers | constraints and hyper-parameters, thresholds carried back to the card, champion memo | `models/challenger_xgb.json` |
| 08 Pilot report | approval rate and segments, fairness proxy, interpretations and caveats | `docs/pilot_report/pilot_report.md` |
| 09 Monitoring | thresholds and cadence, action per scenario, monitoring plan | `docs/monitoring/` |

The first time through, NB01 will show that a block of bureau fields is unreported for the first part of 2012 and that another block exists only from 2016; the scope decision in NB01 (train from the month reporting starts, or from 2013, via `config.TRAIN_VINTAGES` + `make views`) is what makes the later CSI screen in NB04 clean.

## Data status

Produced 2026-10-25 20:02 by `make data`, `make db` and a headless run of notebook 01's starter cells
(fresh kernel). On 2026-10-27 the `scorelab` modules, notebooks 01–09 and the tests were written and every notebook was executed headlessly on a 5% sample against this database (outputs in `models/_smoke_test_5pct/` and `docs/_smoke_test_5pct/`). Raw-file sizes and SHA-256 are in `docs/decisions.md`.

`make db` (row and column counts per object):

```
object            kind    rows         columns
accepted          table      2,260,668     151
rejected          table     27,648,741       9
v_accepted_asof   view       2,260,668      71
v_split           view       2,260,668       7   (stub at build time; 15 after `make views`)
v_target          view       2,260,668       6   (stub at build time; 14 after `make views`)

accepted issue_d range: 2007-06-01 .. 2018-12-01
```

The accepted CSV has 2,260,701 data rows. The 33 rows with a non-numeric `id` (the trailing
"Total amount funded in policy code ..." summary lines) are dropped.

Notebook 01, cell 2:

```
┌─────────────────┐
│      name       │
├─────────────────┤
│ accepted        │
│ rejected        │
│ v_accepted_asof │
│ v_split         │
│ v_target        │
└─────────────────┘
┌─────────┬──────────────┬──────────────┐
│    n    │ min(issue_d) │ max(issue_d) │
├─────────┼──────────────┼──────────────┤
│ 2260668 │ 2007-06-01   │ 2018-12-01   │
└─────────┴──────────────┴──────────────┘
```

`v_split` rows: train 423,810 · validate 421,095 · oot 434,407 · monitor 938,821 · pre-2012 (NULL)
42,535.

Target views (`make views`, 2026-10-29). Labels by split, with bad rate = bad / (bad + good):

| split | bad | good | indeterminate | bad rate |
|---|---:|---:|---:|---:|
| train 2012–14 | 40,187 | 377,210 | 6,413 | 9.63% |
| validate 2015 | 47,469 | 366,822 | 6,804 | 11.46% |
| oot 2016 | 51,546 | 376,693 | 6,168 | 12.04% |
| monitor (2017Q1 observable at 24 mo) | 10,214 | 85,005 | 1,560 | 10.73% |

Not yet available: `data/reference/lc_data_dictionary.csv`. The `wordsforthewise/lending-club`
archive does not ship `LCDataDictionary.xlsx`.
