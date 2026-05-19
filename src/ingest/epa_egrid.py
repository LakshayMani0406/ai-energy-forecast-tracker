#!/usr/bin/env python3
"""
epa_egrid.py — EPA eGRID state-level CO2 emission factor ingestion.

Downloads eGRID annual Excel files, extracts state CO2 output emission rates
(lb/MWh → g/kWh), writes to warehouse table `egrid_state_yearly`.

Source: https://www.epa.gov/egrid/download-data
Files are ~10–30 MB Excel workbooks; cached locally in data/raw/egrid/.

Usage:
  python src/ingest/epa_egrid.py
"""
import io, requests, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from db import get_conn

ROOT      = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / "data" / "raw" / "egrid"

# Known eGRID file URLs — updated when EPA releases new annual data (typically Jan)
# Sheet naming convention: ST{2-digit year} e.g. "ST22" for eGRID2022
EGRID_YEARS: dict[int, str] = {
    2023: "https://www.epa.gov/system/files/documents/2025-06/egrid2023_data_rev2.xlsx",
    2022: "https://www.epa.gov/system/files/documents/2024-01/egrid2022_data.xlsx",
    2021: "https://www.epa.gov/system/files/documents/2023-01/eGRID2021_data.xlsx",
    2020: "https://www.epa.gov/system/files/documents/2022-09/eGRID2020_Data_v2.xlsx",
    # 2018/2019 URLs removed from EPA server — download manually from
    # https://www.epa.gov/egrid/detailed-data and save to data/raw/egrid/
}

# lb/MWh → g/kWh:  1 lb = 453.592 g, 1 MWh = 1000 kWh
LB_MWH_TO_G_KWH = 453.592 / 1000.0


def _sheet_name(year: int) -> str:
    return f"ST{str(year)[-2:]}"


def _download_or_cache(year: int, url: str) -> Path | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"egrid{year}_data.xlsx"
    if dest.exists():
        print(f"   Using cached {dest.name}")
        return dest
    print(f"   Downloading eGRID{year}...")
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"   Saved {len(r.content) / 1e6:.1f} MB → {dest.name}")
        return dest
    except Exception as e:
        print(f"   ⚠️  eGRID{year} download failed: {e}")
        return None


def parse_egrid_state(path: Path, year: int) -> pd.DataFrame | None:
    sheet = _sheet_name(year)
    try:
        # Row 0 is a title row; row 1 has the actual column headers
        df = pd.read_excel(path, sheet_name=sheet, header=1, engine="openpyxl")
    except Exception as e:
        print(f"   ⚠️  Could not read sheet '{sheet}' from {path.name}: {e}")
        return None

    # Find columns by partial name match (eGRID column names stable across versions)
    state_col = next((c for c in df.columns if "PSTATABB" in str(c).upper()), None)
    co2_col   = next((c for c in df.columns if "STCO2RTA" in str(c).upper()), None)
    gen_col   = next((c for c in df.columns if "STGENNTA" in str(c).upper()), None)

    if not state_col or not co2_col:
        print(f"   ⚠️  Required columns not found in {path.name}. "
              f"Available: {list(df.columns[:20])}")
        return None

    rows = df[[c for c in [state_col, co2_col, gen_col] if c]].copy()
    rows.columns = ["state", "co2_rate_lb_per_mwh"] + (["net_gen_mwh"] if gen_col else [])

    # Drop header remnants and aggregates (US total row, etc.)
    rows = rows[rows["state"].str.len() == 2].dropna(subset=["co2_rate_lb_per_mwh"])
    rows = rows[pd.to_numeric(rows["co2_rate_lb_per_mwh"], errors="coerce").notna()]
    rows["co2_rate_lb_per_mwh"] = rows["co2_rate_lb_per_mwh"].astype(float)
    rows["co2_rate_g_per_kwh"]  = rows["co2_rate_lb_per_mwh"] * LB_MWH_TO_G_KWH
    rows["year"] = year
    if "net_gen_mwh" not in rows.columns:
        rows["net_gen_mwh"] = None
    return rows[["year", "state", "co2_rate_lb_per_mwh", "co2_rate_g_per_kwh", "net_gen_mwh"]]


def write_to_duckdb(df: pd.DataFrame) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    df = df.copy()
    df["fetch_timestamp"] = ts

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS egrid_state_yearly (
                year               INTEGER,
                state              VARCHAR,
                co2_rate_lb_per_mwh DOUBLE,
                co2_rate_g_per_kwh  DOUBLE,
                net_gen_mwh        DOUBLE,
                fetch_timestamp    TIMESTAMP,
                PRIMARY KEY (year, state)
            )
        """)
        conn.register("incoming", df)
        conn.execute("""
            DELETE FROM egrid_state_yearly
            WHERE (year, state) IN (SELECT year, state FROM incoming)
        """)
        conn.execute("""
            INSERT INTO egrid_state_yearly
            SELECT year, state, co2_rate_lb_per_mwh, co2_rate_g_per_kwh,
                   TRY_CAST(net_gen_mwh AS DOUBLE), fetch_timestamp::TIMESTAMP
            FROM incoming
        """)
        n = conn.execute("SELECT COUNT(*) FROM egrid_state_yearly").fetchone()[0]
    print(f"   ✅ egrid_state_yearly: {n} total rows in warehouse")


def main():
    all_frames = []
    for year, url in sorted(EGRID_YEARS.items()):
        print(f"\n📊 eGRID{year}...")
        path = _download_or_cache(year, url)
        if path is None:
            continue
        df = parse_egrid_state(path, year)
        if df is not None:
            print(f"   {len(df)} states parsed")
            all_frames.append(df)

    if not all_frames:
        print("❌ No eGRID data successfully parsed.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    write_to_duckdb(combined)
    print(f"\n✅ eGRID ingest complete — {len(combined)} state-year rows")

    # Quick sanity check: US average CO2 rate trend
    national = (combined.groupby("year")["co2_rate_g_per_kwh"]
                .mean().reset_index()
                .rename(columns={"co2_rate_g_per_kwh": "avg_g_per_kwh"}))
    print("\n   National avg CO2 rate (g/kWh) by year:")
    for _, row in national.iterrows():
        print(f"   {int(row.year)}: {row.avg_g_per_kwh:.1f}")


if __name__ == "__main__":
    main()
