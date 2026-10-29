"""DuckDB access: one read-only entry point for notebooks, the MCP server and the agent."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import duckdb
import pandas as pd

from scorelab import config

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def connect(read_only: bool = True, path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open data/lc.duckdb (or ``path``). Read-only by default; only build_db.py writes."""
    db = Path(path) if path is not None else config.DUCKDB_PATH
    if read_only and not db.exists():
        raise FileNotFoundError(f"{db} does not exist; run `make db` first.")
    return duckdb.connect(str(db), read_only=read_only)


def load_table(name: str, con: duckdb.DuckDBPyConnection | None = None,
               columns: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    """Return a table or view as a DataFrame, optionally projecting columns and limiting rows."""
    for ident in [name, *(columns or [])]:
        if not _IDENTIFIER.match(ident):
            raise ValueError(f"Invalid identifier: {ident!r}")
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {name}" + (f" LIMIT {int(limit)}" if limit is not None else "")
    own = con is None
    con = con or connect()
    try:
        return con.sql(sql).df()
    finally:
        if own:
            con.close()


def as_of_application_fields(path: str | Path | None = None) -> list[str]:
    """Columns known at application time (the leakage-control keep-list, spec section 4)."""
    with open(path or config.AS_OF_FIELDS_CSV, newline="", encoding="utf-8") as f:
        return [row["field"] for row in csv.DictReader(f)]


def as_of_application(con: duckdb.DuckDBPyConnection | None = None,
                      limit: int | None = None) -> pd.DataFrame:
    """Accepted loans restricted to as-of-application fields (the v_accepted_asof view)."""
    return load_table("v_accepted_asof", con=con, limit=limit)


# --- Profiling for the data audit (NB01) ------------------------------------------------------

def row_counts(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Rows per table/view and the issue-date range of the accepted file."""
    own = con is None; con = con or connect()
    try:
        rows = []
        for name, kind in con.execute("SELECT table_name, table_type FROM information_schema.tables ORDER BY 2, 1").fetchall():
            rows.append({"object": name, "kind": "table" if kind == "BASE TABLE" else "view",
                         "rows": con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]})
        return pd.DataFrame(rows)
    finally:
        if own: con.close()


def missingness_by_vintage(columns: list[str], con: duckdb.DuckDBPyConnection | None = None,
                           source: str = "v_accepted_asof", min_year: int = 2012) -> pd.DataFrame:
    """Share missing per column and issue year (finds fields that only exist for late vintages)."""
    for c in columns + [source]:
        if not _IDENTIFIER.match(c): raise ValueError(c)
    exprs = ", ".join(f"ROUND(AVG(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END), 4) AS {c}" for c in columns)
    own = con is None; con = con or connect()
    try:
        df = con.sql(f"SELECT year(issue_d) AS vintage_year, COUNT(*) AS n, {exprs} FROM {source} "
                     f"WHERE year(issue_d) >= {int(min_year)} GROUP BY 1 ORDER BY 1").df()
        return df.set_index("vintage_year").T
    finally:
        if own: con.close()


def numeric_profile(columns: list[str], con: duckdb.DuckDBPyConnection | None = None, source: str = "v_accepted_asof",
                    years: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Count, missing share, min, p1, median, p99, max per numeric column (SQL, no data pulled)."""
    for c in columns + [source]:
        if not _IDENTIFIER.match(c): raise ValueError(c)
    where = f"WHERE year(issue_d) IN ({', '.join(str(y) for y in years)})" if years else ""
    own = con is None; con = con or connect()
    try:
        rows = []
        for c in columns:
            r = con.execute(f"""SELECT COUNT({c}), AVG(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END), MIN({c}),
                               quantile_cont({c}, 0.01), median({c}), quantile_cont({c}, 0.99), MAX({c})
                               FROM {source} {where}""").fetchone()
            rows.append({"column": c, "n_nonmissing": r[0], "missing_share": round(r[1], 4), "min": r[2], "p1": r[3],
                         "median": r[4], "p99": r[5], "max": r[6]})
        return pd.DataFrame(rows)
    finally:
        if own: con.close()


def categorical_profile(column: str, con: duckdb.DuckDBPyConnection | None = None, source: str = "v_accepted_asof",
                        top: int = 20) -> pd.DataFrame:
    if not _IDENTIFIER.match(column) or not _IDENTIFIER.match(source): raise ValueError(column)
    own = con is None; con = con or connect()
    try:
        return con.sql(f"SELECT {column} AS level, COUNT(*) AS n, ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4) AS share "
                       f"FROM {source} GROUP BY 1 ORDER BY 2 DESC LIMIT {int(top)}").df()
    finally:
        if own: con.close()


def incumbent_coverage(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """FICO range availability and distribution of the midpoint by vintage year."""
    own = con is None; con = con or connect()
    try:
        return con.sql("""
            SELECT year(issue_d) AS vintage_year, COUNT(*) AS n,
                   ROUND(AVG(CASE WHEN fico_range_low IS NULL THEN 1 ELSE 0 END), 4) AS fico_missing,
                   MIN((fico_range_low + fico_range_high) / 2) AS fico_min,
                   ROUND(AVG((fico_range_low + fico_range_high) / 2), 1) AS fico_mean,
                   MAX((fico_range_low + fico_range_high) / 2) AS fico_max,
                   ROUND(AVG(fico_range_high - fico_range_low), 1) AS range_width
            FROM v_accepted_asof WHERE year(issue_d) >= 2012 GROUP BY 1 ORDER BY 1""").df()
    finally:
        if own: con.close()
