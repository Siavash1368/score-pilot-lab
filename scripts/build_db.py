"""Build data/lc.duckdb from the raw Kaggle CSVs.

Each CSV is read with ``read_csv(..., all_varchar=True)`` into a staging table,
then cast column by column into a typed table:

* ``accepted``  all 151 columns. ``Mon-YYYY`` columns -> DATE (first of month),
  percent strings -> DOUBLE, ``emp_length`` -> INTEGER years. Rows whose id is
  not numeric (the trailing "Total amount funded..." summary rows) are dropped.
* ``rejected``  the nine columns, snake-cased, ``application_date`` -> DATE,
  ``dti`` percent -> DOUBLE.

Then the views ``v_accepted_asof`` (leakage-control keep-list), ``v_target`` and
``v_split`` (stubs until notebook 02) are created and row/column counts printed.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scorelab import config  # noqa: E402

# --- accepted: expected header and explicit types ---------------------------

ACCEPTED_COLUMNS = [
    "id", "member_id", "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate",
    "installment", "grade", "sub_grade", "emp_title", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "pymnt_plan", "url", "desc", "purpose", "title",
    "zip_code", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "initial_list_status", "out_prncp",
    "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code", "application_type",
    "annual_inc_joint", "dti_joint", "verification_status_joint", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit", "revol_bal_joint",
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
    "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog", "hardship_flag",
    "hardship_type", "hardship_reason", "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount", "disbursement_method",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
]

TEXT_COLUMNS = {
    "term", "grade", "sub_grade", "emp_title", "home_ownership", "verification_status",
    "loan_status", "pymnt_plan", "url", "desc", "purpose", "title", "zip_code", "addr_state",
    "initial_list_status", "application_type", "verification_status_joint", "hardship_flag",
    "hardship_type", "hardship_reason", "hardship_status", "hardship_loan_status",
    "disbursement_method", "debt_settlement_flag", "settlement_status",
}
DATE_COLUMNS = {
    "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    "sec_app_earliest_cr_line", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "debt_settlement_flag_date", "settlement_date",
}
PERCENT_COLUMNS = {"int_rate", "revol_util"}
BIGINT_COLUMNS = {"id", "member_id"}
# Everything else is numeric and stored as DOUBLE.

# --- rejected: header -> (snake_case name, type) -----------------------------

REJECTED_COLUMNS = {
    "Amount Requested": ("amount_requested", "DOUBLE"),
    "Application Date": ("application_date", "DATE"),
    "Loan Title": ("loan_title", "VARCHAR"),
    "Risk_Score": ("risk_score", "DOUBLE"),
    "Debt-To-Income Ratio": ("dti", "PERCENT"),
    "Zip Code": ("zip_code", "VARCHAR"),
    "State": ("state", "VARCHAR"),
    "Employment Length": ("emp_length", "EMP_LENGTH"),
    "Policy Code": ("policy_code", "INTEGER"),
}

# --- as-of-application keep-list (spec section 4.1) -------------------------

AS_OF_FIELDS = [
    ("incumbent_score", "input_excluded", ["fico_range_low", "fico_range_high"]),
    ("inquiries", "input", ["inq_last_6mths", "inq_last_12m", "inq_fi", "mths_since_recent_inq"]),
    ("delinquency", "input", [
        "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record",
        "mths_since_last_major_derog", "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq"]),
    ("derogatory", "input", [
        "pub_rec", "pub_rec_bankruptcies", "tax_liens", "collections_12_mths_ex_med",
        "chargeoff_within_12_mths", "acc_now_delinq", "delinq_amnt"]),
    ("tradelines", "input", [
        "open_acc", "total_acc", "acc_open_past_24mths", "num_actv_bc_tl", "num_actv_rev_tl",
        "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
        "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m", "mort_acc", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m",
        "open_rv_24m"]),
    ("balances_utilisation", "input", [
        "revol_bal", "revol_util", "bc_util", "bc_open_to_buy", "all_util", "il_util", "total_bal_il",
        "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit", "total_rev_hi_lim",
        "tot_hi_cred_lim", "tot_cur_bal", "tot_coll_amt", "max_bal_bc", "avg_cur_bal",
        "percent_bc_gt_75", "pct_tl_nvr_dlq"]),
    ("credit_age", "input", [
        "earliest_cr_line", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl", "mths_since_rcnt_il"]),
    ("key_context", "not_input", ["id", "issue_d", "loan_status", "addr_state", "purpose",
                                  "application_type"]),
]
# Always present in v_accepted_asof regardless of the CSV (spec section 3.2).
AS_OF_REQUIRED = ["id", "issue_d", "loan_status", "fico_range_low", "fico_range_high"]


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def pct(col: str) -> str:
    return f"CAST(NULLIF(rtrim(trim({q(col)}), '%'), '') AS DOUBLE)"


def emp_length(col: str) -> str:
    c = f"trim({q(col)})"
    return (f"CASE WHEN {c} IS NULL OR {c} IN ('', 'n/a') THEN NULL "
            f"WHEN {c} LIKE '<%' THEN 0 "
            f"ELSE CAST(regexp_extract({c}, '(\\d+)', 1) AS INTEGER) END")


def accepted_expr(col: str) -> str:
    if col in DATE_COLUMNS:
        return f"strptime({q(col)}, '%b-%Y')::DATE AS {q(col)}"
    if col in PERCENT_COLUMNS:
        return f"{pct(col)} AS {q(col)}"
    if col == "emp_length":
        return f"{emp_length(col)} AS {q(col)}"
    if col in TEXT_COLUMNS:
        return f"{q(col)}::VARCHAR AS {q(col)}"
    if col in BIGINT_COLUMNS:
        return f"CAST({q(col)} AS BIGINT) AS {q(col)}"
    return f"CAST({q(col)} AS DOUBLE) AS {q(col)}"


def rejected_expr(header: str, name: str, kind: str) -> str:
    if kind == "PERCENT":
        return f"{pct(header)} AS {name}"
    if kind == "EMP_LENGTH":
        return f"{emp_length(header)} AS {name}"
    return f"CAST({q(header)} AS {kind}) AS {name}"


def read_varchar(path: Path, columns: list[str] | None = None) -> str:
    cols = ", ".join(q(c) for c in columns) if columns else "*"
    return (f"SELECT {cols} FROM read_csv('{path.as_posix()}', header=true, all_varchar=true, "
            f"sample_size=-1)")


def staging_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def check_header(actual: list[str], expected: list[str], label: str) -> None:
    if actual == expected:
        return
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    raise SystemExit(f"{label} header mismatch.\n  missing: {missing}\n  unexpected: {extra}\n"
                     f"  (order differs: {not missing and not extra})")


def write_as_of_fields(path: Path) -> None:
    """Write the starting keep-list; NB01 owns the file afterwards, so never overwrite it."""
    if path.exists():
        print(f"Using existing {path.relative_to(config.ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "group", "role"])
        for group, role, fields in AS_OF_FIELDS:
            for field in fields:
                w.writerow([field, group, role])
    print(f"Wrote {path.relative_to(config.ROOT)}")


def step(label: str, con: duckdb.DuckDBPyConnection, sql: str) -> None:
    t0 = time.time()
    print(f"{label} ...", end=" ", flush=True)
    con.execute(sql)
    print(f"{time.time() - t0:.0f}s", flush=True)


def build_accepted(con: duckdb.DuckDBPyConnection) -> None:
    step("Staging accepted (all VARCHAR)", con,
         f"CREATE TABLE stg_accepted AS {read_varchar(config.ACCEPTED_CSV)}")
    check_header(staging_columns(con, "stg_accepted"), ACCEPTED_COLUMNS, "accepted")
    n_all = con.execute("SELECT COUNT(*) FROM stg_accepted").fetchone()[0]
    n_bad = con.execute("SELECT COUNT(*) FROM stg_accepted "
                        "WHERE TRY_CAST(id AS BIGINT) IS NULL").fetchone()[0]
    print(f"  staging rows {n_all:,}; dropping {n_bad:,} rows with non-numeric id")
    exprs = ",\n  ".join(accepted_expr(c) for c in ACCEPTED_COLUMNS)
    step("Casting accepted", con,
         f"CREATE TABLE accepted AS SELECT\n  {exprs}\nFROM stg_accepted\n"
         f"WHERE TRY_CAST(id AS BIGINT) IS NOT NULL")
    con.execute("DROP TABLE stg_accepted")


def build_rejected(con: duckdb.DuckDBPyConnection) -> None:
    headers = list(REJECTED_COLUMNS)
    step("Staging rejected (all VARCHAR, projected)", con,
         f"CREATE TABLE stg_rejected AS {read_varchar(config.REJECTED_CSV, headers)}")
    exprs = ",\n  ".join(rejected_expr(h, n, k) for h, (n, k) in REJECTED_COLUMNS.items())
    step("Casting rejected", con, f"CREATE TABLE rejected AS SELECT\n  {exprs}\nFROM stg_rejected")
    con.execute("DROP TABLE stg_rejected")


def build_views(con: duckdb.DuckDBPyConnection) -> None:
    write_as_of_fields(config.AS_OF_FIELDS_CSV)
    with open(config.AS_OF_FIELDS_CSV, newline="", encoding="utf-8") as f:
        listed = [row["field"] for row in csv.DictReader(f)]
    fields = list(dict.fromkeys(AS_OF_REQUIRED + listed))
    present = set(staging_columns(con, "accepted"))
    missing = [c for c in fields if c not in present]
    if missing:
        raise SystemExit(f"as_of_application_fields.csv lists columns missing from accepted: {missing}")
    con.execute("CREATE VIEW v_accepted_asof AS SELECT "
                + ", ".join(q(c) for c in fields) + " FROM accepted")

    # Stub: vintage and censoring flag only. bad stays NULL until NB02 implements scorelab.target.
    cut = f"DATE '{config.DATA_CUT}'"
    con.execute(f"""
        CREATE VIEW v_target AS
        SELECT id,
               issue_d,
               year(issue_d) AS vintage_year,
               year(issue_d) || 'Q' || quarter(issue_d) AS vintage_quarter,
               CAST(NULL AS INTEGER) AS bad,  -- TODO(NB02): 24-month bad definition
               CAST(issue_d + INTERVAL {config.PERFORMANCE_WINDOW_MONTHS} MONTH <= {cut} AS INTEGER)
                   AS observable
        FROM v_accepted_asof
    """)

    # Stub split by vintage: OOT is the whole of 2016 and monitor is MONITOR_VINTAGES (2017Q1 on).
    train = ", ".join(str(y) for y in config.TRAIN_VINTAGES)
    mon_lo, mon_hi = config.MONITOR_VINTAGES
    con.execute(f"""
        CREATE VIEW v_split AS
        SELECT *,
               CASE WHEN vintage_year IN ({train}) THEN 'train'
                    WHEN vintage_year = {config.VALIDATE_VINTAGE} THEN 'validate'
                    WHEN vintage_year = {config.OOT_VINTAGE} THEN 'oot'
                    WHEN vintage_quarter BETWEEN '{mon_lo}' AND '{mon_hi}' THEN 'monitor'
               END AS split  -- NULL = pre-{min(config.TRAIN_VINTAGES)} vintages, not used
        FROM v_target
    """)


def print_counts(con: duckdb.DuckDBPyConnection) -> None:
    print("\nobject            kind    rows         columns")
    for name, kind in con.execute("SELECT table_name, table_type FROM information_schema.tables "
                                  "ORDER BY table_type, table_name").fetchall():
        rows = con.execute(f"SELECT COUNT(*) FROM {q(name)}").fetchone()[0]
        cols = len(staging_columns(con, q(name)))
        print(f"{name:<17} {'table' if kind == 'BASE TABLE' else 'view':<7} {rows:>12,}  {cols:>6}")
    lo, hi = con.execute("SELECT MIN(issue_d), MAX(issue_d) FROM accepted").fetchone()
    print(f"\naccepted issue_d range: {lo} .. {hi}")


def main() -> int:
    for p in (config.ACCEPTED_CSV, config.REJECTED_CSV):
        if not p.is_file():
            print(f"ERROR: {p} not found; run `make data` first.", file=sys.stderr)
            return 1
    db = config.DUCKDB_PATH
    tmp = db.with_suffix(".building.duckdb")
    for p in (tmp, Path(str(tmp) + ".wal")):
        p.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        build_accepted(con)
        build_rejected(con)
        build_views(con)
        print_counts(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    db.unlink(missing_ok=True)
    tmp.rename(db)
    print(f"\nWrote {db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
