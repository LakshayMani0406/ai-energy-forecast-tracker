"""Shared DuckDB connection helper for all ingest modules."""
import duckdb
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "warehouse.duckdb"


def get_conn() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))
