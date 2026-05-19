"""
Tests for the fusion model outputs stored in DuckDB.

These are integration tests — they verify that the fusion_posterior table
contains the expected data shape and plausible values, not that the PyMC
model converges in a specific way.

Run with:
  pytest tests/test_fusion.py -v
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn


@pytest.fixture(scope="module")
def conn():
    """Shared read-only DuckDB connection for all tests in this module."""
    with get_conn() as c:
        yield c


def test_fusion_posterior_row_count(conn):
    """302 monthly rows for every non-state variable (2001-01 to 2026-02)."""
    result = conn.execute("""
        SELECT variable, count(*) AS n
        FROM fusion_posterior
        WHERE variable NOT LIKE 'state_dc_gwh_%'
        GROUP BY variable
    """).fetchall()
    for var, n in result:
        assert n == 302, f"{var} has {n} rows, expected 302"


def test_key_variables_present(conn):
    """Core variables must exist in fusion_posterior."""
    expected = {"dc_gwh", "dc_twh", "dc_co2_mt_monthly", "ai_gwh", "non_ai_gwh"}
    actual = {r[0] for r in conn.execute(
        "SELECT DISTINCT variable FROM fusion_posterior"
    ).fetchall()}
    missing = expected - actual
    assert not missing, f"Missing variables: {missing}"


def test_state_variables_present(conn):
    """All 13 top-state dc_gwh variables must exist."""
    states = {"VA","TX","CA","GA","OH","IL","AZ","WA","OR","NY","NJ","NC","FL"}
    actual = {r[0].replace("state_dc_gwh_","") for r in conn.execute(
        "SELECT DISTINCT variable FROM fusion_posterior WHERE variable LIKE 'state_dc_gwh_%'"
    ).fetchall()}
    missing = states - actual
    assert not missing, f"Missing state variables: {missing}"


def test_dc_gwh_plausible_range(conn):
    """Annual total facility energy should be in 40–300 TWh for any complete year."""
    result = conn.execute("""
        SELECT year(ds) AS yr, sum(mean)/1000.0 AS twh
        FROM fusion_posterior
        WHERE variable = 'dc_gwh'
        GROUP BY yr
        HAVING count(*) = 12
        ORDER BY yr
    """).fetchall()
    assert len(result) > 0, "No complete years found in fusion_posterior"
    for yr, twh in result:
        assert 40 <= twh <= 300, f"Year {yr}: {twh:.1f} TWh is outside plausible range [40, 300]"


def test_co2_positive_and_bounded(conn):
    """Monthly CO₂ values should be positive and < 20 Mt/month."""
    result = conn.execute("""
        SELECT min(mean), max(mean)
        FROM fusion_posterior
        WHERE variable = 'dc_co2_mt_monthly'
    """).fetchone()
    min_val, max_val = result
    assert min_val > 0, f"CO₂ has non-positive values: min={min_val}"
    assert max_val < 20, f"CO₂ max {max_val:.2f} Mt/month exceeds 20 Mt — likely unit error"


def test_ai_fraction_structural_break(conn):
    """AI GWh fraction should be higher post-2023 than pre-2022."""
    pre_2022 = conn.execute("""
        SELECT sum(CASE WHEN variable='ai_gwh' THEN mean ELSE 0 END) /
               sum(CASE WHEN variable='dc_gwh' THEN mean ELSE 0 END) AS frac
        FROM fusion_posterior
        WHERE year(ds) <= 2022
    """).fetchone()[0]
    post_2023 = conn.execute("""
        SELECT sum(CASE WHEN variable='ai_gwh' THEN mean ELSE 0 END) /
               sum(CASE WHEN variable='dc_gwh' THEN mean ELSE 0 END) AS frac
        FROM fusion_posterior
        WHERE year(ds) >= 2023
    """).fetchone()[0]
    assert post_2023 > pre_2022 * 2, (
        f"AI fraction post-2023 ({post_2023:.3f}) should be >2× pre-2022 ({pre_2022:.3f})"
    )


def test_2024_annual_totals_plausible(conn):
    """2024 annual totals should be in IEA-adjacent range."""
    energy_twh = conn.execute("""
        SELECT sum(mean)/1000.0 FROM fusion_posterior
        WHERE variable='dc_gwh' AND year(ds)=2024
    """).fetchone()[0]
    co2_mt = conn.execute("""
        SELECT sum(mean) FROM fusion_posterior
        WHERE variable='dc_co2_mt_monthly' AND year(ds)=2024
    """).fetchone()[0]
    # Total facility energy: should be between 100 and 400 TWh
    assert 100 <= energy_twh <= 400, f"2024 energy {energy_twh:.1f} TWh outside [100, 400]"
    # CO2: our model gives ~68 Mt, IEA says 105; truth is between
    assert 40 <= co2_mt <= 130, f"2024 CO₂ {co2_mt:.1f} Mt outside [40, 130]"
