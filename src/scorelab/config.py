"""Paths and project-wide constants (spec section 5.2).

Every module and notebook reads these values from here, so a
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
# 2012 is out: a block of 21 bureau fields is unreported for January–August 2012 (NB01, decided
# 2026-10-31, see docs/decisions.md).
TRAIN_VINTAGES = (2013, 2014)
VALIDATE_VINTAGE = 2015
OOT_VINTAGE = 2016
# Inclusive range; outcomes partially censored. Starts after the full-year 2016 OOT vintage so
# v_split labels are exclusive (spec section 5.2 said 2016Q3; see docs/decisions.md).
MONITOR_VINTAGES = ("2017Q1", "2018Q4")

# --- Target ------------------------------------------------------------------
# Performance is observed at the data snapshot, not at the last issue month: the newest
# last_pymnt_d in the accepted file is 2019-03-01. A vintage is observable at horizon h months
# when issue_d + h months <= PERFORMANCE_SNAPSHOT.
PERFORMANCE_SNAPSHOT = "2019-03-01"
PERFORMANCE_WINDOW_MONTHS = 24
# Terminal statuses that mean the loan reached serious delinquency.
BAD_STATUSES = ("Charged Off", "Default")
# Open delinquency at the snapshot: not (yet) 90+ DPD, treated by the same timing arithmetic.
DELINQUENT_STATUSES = ("Late (31-120 days)", "Late (16-30 days)", "In Grace Period")
# Lending Club publishes no monthly payment history, so the month a loan reached 90+ DPD is
# approximated from last_pymnt_d: the first missed due date is one month after the last payment,
# 60 DPD two months after that, 90 DPD three months after that. A loan that never paid has its
# first miss in month 1.
MONTHS_LAST_PAYMENT_TO_FIRST_MISS = 1
MONTHS_FIRST_MISS_TO_60DPD = 2
MONTHS_FIRST_MISS_TO_90DPD = 3
# Label rule (decided 2026-10-27, see docs/decisions.md):
#   bad           = delinquent status and 90+ DPD reached within the window
#   indeterminate = delinquent status, first miss inside the window, 90+ DPD after it
#   good          = everything else that is observable (Fully Paid, Current, or delinquent only
#                   after the window closed)
# Early-performance indicator for the censored monitor vintages (NB09): 60+ DPD within 12 months.
EARLY_WINDOW_MONTHS = 12
CREDIT_POLICY_PREFIX = "Does not meet the credit policy. Status:"
# Application types excluded from every split: joint applications first appear in the validation
# year (none in train), and the joint-application fields are out of scope for v1 (NB02, 2026-10-31).
EXCLUDED_APPLICATION_TYPES = ("Joint App",)

# --- Incumbent score -----------------------------------------------------------
# The pilot benchmarks the scorecard against the lender's current score: the FICO range at
# origination, summarised by its midpoint. FICO fields are never model inputs (see keep-list).
INCUMBENT_SCORE_SQL = "(fico_range_low + fico_range_high) / 2.0"
INCUMBENT_SCORE_NAME = "fico_mid"

# --- Monitoring --------------------------------------------------------------
PSI_THRESHOLDS = {"watch": 0.10, "action": 0.25}  # Siddiqi Ch 14 convention

# --- Pilot / adverse action --------------------------------------------------
APPROVAL_RATE_FOR_SWAP = 0.70
N_REASON_CODES = 4
