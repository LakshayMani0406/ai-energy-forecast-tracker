"""
Tests for the benchmark scoring module.

Verifies that benchmark_scores contains the expected entries and
that graded entries have plausible values.

Run with:
  pytest tests/test_benchmarks.py -v
"""
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

GRADED_SOURCES  = {"LBNL", "Masanet et al.", "Guidi et al.", "IEA"}
PENDING_SOURCES = {"Goldman Sachs", "EPRI", "McKinsey", "BloombergNEF"}
VALID_GRADES    = {"A", "B", "C", "D", "F", "pending"}


@pytest.fixture(scope="module")
def scores():
    with get_conn() as conn:
        return conn.execute("SELECT * FROM benchmark_scores").df()


def test_table_not_empty(scores):
    assert len(scores) >= 10, f"Expected ≥10 rows, got {len(scores)}"


def test_all_grades_valid(scores):
    bad = set(scores["grade"].unique()) - VALID_GRADES
    assert not bad, f"Invalid grades found: {bad}"


def test_graded_entries_have_actuals(scores):
    """Every non-pending entry must have an actual_value and error_pct."""
    graded = scores[scores["grade"] != "pending"]
    assert graded["actual_value"].notna().all(), "Some graded entries missing actual_value"
    assert graded["error_pct"].notna().all(), "Some graded entries missing error_pct"


def test_pending_entries_have_no_actuals(scores):
    """Pending entries must have null actual_value."""
    pending = scores[scores["grade"] == "pending"]
    assert pending["actual_value"].isna().all(), "Some pending entries have actual_value"


def test_lbnl_underestimated(scores):
    """LBNL 2016 should have negative error (underestimate) and grade D or F."""
    row = scores[(scores["source"] == "LBNL") & (scores["forecast_year"] == 2020)]
    assert len(row) == 1, "LBNL 2020 entry missing"
    assert row.iloc[0]["error_pct"] < -20, "LBNL 2020 should underestimate by >20%"
    assert row.iloc[0]["grade"] in {"D", "F"}, "LBNL 2020 should grade D or F"


def test_masanet_underestimated(scores):
    row = scores[(scores["source"] == "Masanet et al.") & (scores["forecast_year"] == 2018)]
    assert len(row) == 1
    assert row.iloc[0]["error_pct"] < -15, "Masanet 2018 should underestimate by >15%"


def test_error_pct_consistency(scores):
    """error_pct should equal (forecast_mid - actual_value) / actual_value * 100."""
    graded = scores[scores["grade"] != "pending"].copy()
    graded["recomputed"] = (graded["forecast_mid"] - graded["actual_value"]) / graded["actual_value"] * 100
    diff = (graded["error_pct"] - graded["recomputed"]).abs()
    assert (diff < 0.1).all(), f"error_pct inconsistency: max diff = {diff.max():.4f}"


def test_all_pending_have_url(scores):
    """All benchmark entries should have a non-empty URL."""
    empty_url = scores[scores["url"].isna() | (scores["url"] == "")]
    assert len(empty_url) == 0, f"{len(empty_url)} entries missing URL"
