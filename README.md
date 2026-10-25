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
make test
make dashboard   # Streamlit PSI/CSI (once notebook 09 exists)
```

Requirements: Python 3.11 or 3.12, a Kaggle API token (`~/.kaggle/kaggle.json` or
`~/.kaggle/access_token`), and roughly 8 GB of free disk for the raw CSVs plus the DuckDB file.
On Windows without GNU make, `mingw32-make` works with the same targets, or run the scripts directly
(`python scripts/download_data.py`, `python scripts/build_db.py`).

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` before using the agent.

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
| `src/scorelab/` | All analytics. `config.py` holds paths and constants; `data.py` is the only DuckDB entry point |
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
| `v_target` | view (stub) | `id, issue_d, vintage_year, vintage_quarter, bad, observable`. `bad` is NULL until notebook 02 implements `scorelab.target` |
| `v_split` | view (stub) | `split` ∈ {train, validate, oot, monitor} by vintage |

## Data status

_Not yet built._
