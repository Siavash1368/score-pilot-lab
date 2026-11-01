"""Recreate v_target and v_split on an existing lc.duckdb without rebuilding the tables.

Run after changing the target definition in scorelab.config or scorelab.target:

    python scripts/refresh_views.py

Opens the database in write mode (close notebook kernels first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scorelab import config, target  # noqa: E402


def main() -> None:
    # DuckDB's table printout uses box-drawing characters that a cp1252 Windows console rejects.
    sys.stdout.reconfigure(encoding="utf-8")
    db = config.DUCKDB_PATH
    if not db.exists():
        raise SystemExit(f"{db} does not exist; run `make db` first.")
    con = duckdb.connect(str(db), read_only=False)
    try:
        target.create_views(con)
        for name in ("v_target", "v_split"):
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchall()[0][0]
            cols = len(con.execute(f"DESCRIBE {name}").fetchall())
            print(f"{name:<9} rows {n:>12,}  columns {cols:>3}")
        print(con.sql("""
            SELECT split, label, COUNT(*) AS n
            FROM v_split WHERE exclusion IS NULL
            GROUP BY 1, 2 ORDER BY 1, 2
        """))
    finally:
        con.close()
    print(f"Refreshed views in {db}")


if __name__ == "__main__":
    main()
