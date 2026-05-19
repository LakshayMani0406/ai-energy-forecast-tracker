"""
Forecast Memory: permanent timestamped storage for institutional forecasts.

Every forecast ever made is stored with:
  - original assumptions (JSON)
  - methodology notes
  - revision lineage (revision_of → prior forecast_id)
  - actuals and error% filled in once target year passes
  - confidence_score derived from error history
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn  # noqa: E402

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_memory (
    forecast_id       VARCHAR PRIMARY KEY,
    source            VARCHAR NOT NULL,
    report_title      VARCHAR,
    published_date    DATE,
    forecast_vintage  VARCHAR,
    variable          VARCHAR NOT NULL,
    target_year       INTEGER NOT NULL,
    forecast_lo       DOUBLE,
    forecast_mid      DOUBLE NOT NULL,
    forecast_hi       DOUBLE,
    unit              VARCHAR DEFAULT 'TWh/yr',
    assumptions       JSON,
    methodology       VARCHAR,
    revision_of       VARCHAR,
    actual_value      DOUBLE,
    error_pct         DOUBLE,
    confidence_score  DOUBLE,
    notes             VARCHAR,
    created_at        TIMESTAMP DEFAULT current_timestamp
);
"""


def init_store() -> None:
    with get_conn() as conn:
        conn.execute(_SCHEMA)


def upsert(row: dict[str, Any]) -> None:
    init_store()
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO forecast_memory (
                forecast_id, source, report_title, published_date, forecast_vintage,
                variable, target_year, forecast_lo, forecast_mid, forecast_hi, unit,
                assumptions, methodology, revision_of, actual_value, error_pct,
                confidence_score, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            row["forecast_id"],
            row.get("source"),
            row.get("report_title"),
            row.get("published_date"),
            row.get("forecast_vintage"),
            row["variable"],
            int(row["target_year"]),
            row.get("forecast_lo"),
            float(row["forecast_mid"]),
            row.get("forecast_hi"),
            row.get("unit", "TWh/yr"),
            json.dumps(row.get("assumptions", {})),
            row.get("methodology"),
            row.get("revision_of"),
            row.get("actual_value"),
            row.get("error_pct"),
            row.get("confidence_score"),
            row.get("notes"),
        ])


def load_all() -> pd.DataFrame:
    init_store()
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM forecast_memory ORDER BY target_year, source"
        ).df()


def load_graded() -> pd.DataFrame:
    init_store()
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM forecast_memory WHERE actual_value IS NOT NULL ORDER BY target_year, source"
        ).df()
