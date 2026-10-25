"""Paths and project-wide constants (spec section 5.2).

Every module, the MCP server and the agent read these values from here, so a
change made once applies to every number the project produces.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

def _env_path(name: str, default: Path) -> Path:
    """Path from an environment variable (relative values resolve against ROOT), else ``default``."""
    p = Path(os.environ.get(name) or default)
    return p if p.is_absolute() else ROOT / p


# RAW_DATA_DIR and DUCKDB_PATH can point outside the repo, e.g. to keep large files out of a
# synced folder. data/reference/ always stays in the repo and is tracked.
DATA_DIR = ROOT / "data"
RAW_DIR = _env_path("RAW_DATA_DIR", DATA_DIR / "raw")
REFERENCE_DIR = DATA_DIR / "reference"
DUCKDB_PATH = _env_path("DUCKDB_PATH", DATA_DIR / "lc.duckdb")

ACCEPTED_CSV = RAW_DIR / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = RAW_DIR / "rejected_2007_to_2018Q4.csv"
AS_OF_FIELDS_CSV = REFERENCE_DIR / "as_of_application_fields.csv"
DATA_DICTIONARY_CSV = REFERENCE_DIR / "lc_data_dictionary.csv"

DOCS_DIR = ROOT / "docs"
MODEL_CARD = DOCS_DIR / "model_card.md"
MODELS_DIR = ROOT / "models"

# --- Score scaling -----------------------------------------------------------
# Score = BASE_SCORE at odds BASE_ODDS (good:bad); every PDO points doubles the odds.
# Alternative from the spec: 650 / 50:1 / 20.
BASE_SCORE = 600
BASE_ODDS = 30.0
PDO = 40
SCORE_MIN = 300
SCORE_MAX = 850

# --- Sample design -----------------------------------------------------------
DATA_CUT = "2018-12-31"  # last month of the 2018Q4 data cut
TRAIN_VINTAGES = (2012, 2013, 2014)
VALIDATE_VINTAGE = 2015
OOT_VINTAGE = 2016
# Inclusive range; outcomes partially censored. Starts after the full-year 2016 OOT vintage so
# v_split labels are exclusive (spec section 5.2 said 2016Q3; see docs/decisions.md).
MONITOR_VINTAGES = ("2017Q1", "2018Q4")

# --- Target ------------------------------------------------------------------
PERFORMANCE_WINDOW_MONTHS = 24
BAD_STATUSES = ("Charged Off", "Default", "Late (31-120 days)")
CREDIT_POLICY_PREFIX = "Does not meet the credit policy. Status:"

# --- Monitoring --------------------------------------------------------------
PSI_THRESHOLDS = {"watch": 0.10, "action": 0.25}  # Siddiqi Ch 14 convention

# --- Pilot / adverse action --------------------------------------------------
APPROVAL_RATE_FOR_SWAP = 0.70
N_REASON_CODES = 4
