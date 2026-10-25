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
