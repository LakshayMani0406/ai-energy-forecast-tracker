#!/usr/bin/env python3
"""
fetch_data.py — pulls US commercial electricity data from EIA API,
converts to datacenter CO2 estimates using EPA emission factors,
saves to data/raw/energy_data.csv.

EIA API key: free at https://www.eia.gov/opendata/register.php
Without a key, falls back to the bundled seed dataset.
"""
import os, json, requests, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH  = ROOT / "data" / "raw" / "energy_data.csv"
SEED_PATH  = ROOT / "data" / "raw" / "seed_data.csv"

# EPA national average grid emission factor (kg CO2 / kWh) — updated annually
EPA_KG_CO2_PER_KWH = 0.386

# LBNL 2024: datacenters consume ~3.5% of US commercial electricity
DATACENTER_SHARE = 0.035

EIA_BASE = (
    "https://api.eia.gov/v2/electricity/retail-sales/data/"
    "?api_key={key}"
    "&frequency=monthly"
    "&data[0]=sales"
    "&facets[sectorid][]=COM"
    "&sort[0][column]=period"
    "&sort[0][direction]=asc"
    "&length=5000"
    "&offset={offset}"
)


def fetch_from_eia(api_key: str) -> pd.DataFrame:
    all_records = []
    offset = 0
    while True:
        url = EIA_BASE.format(key=api_key, offset=offset)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        body    = r.json()["response"]
        records = body.get("data", [])
        total   = int(body.get("total", 0))
        all_records.extend(records)
        print(f"   Fetched {len(all_records)}/{total} records...")
        if len(all_records) >= total or not records:
            break
        offset += len(records)

    rows = []
    for rec in all_records:
        period = rec.get("period", "")
        sales  = rec.get("sales")
        if not period or sales is None:
            continue
        try:
            dt      = datetime.strptime(period, "%Y-%m")
            sales_v = float(sales)
            if sales_v <= 0:
                continue
        except (ValueError, TypeError):
            continue
        rows.append({"ds": dt, "commercial_gwh": sales_v})

    if not rows:
        raise ValueError("EIA API returned no usable records")

    df = pd.DataFrame(rows)
    # Aggregate state-level rows to national monthly totals
    df = df.groupby("ds", as_index=False)["commercial_gwh"].sum()
    df = df.sort_values("ds").reset_index(drop=True)

    # Drop months with implausibly low totals (partial/lagged data)
    median_sales = df["commercial_gwh"].median()
    df = df[df["commercial_gwh"] > median_sales * 0.5].reset_index(drop=True)

    return df


def build_seed_data() -> pd.DataFrame:
    """
    Synthetic monthly US commercial electricity 2015–2025 based on
    EIA/LBNL published annual totals — realistic enough to demo the pipeline.
    Replace with real EIA API data for production use.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    # US commercial sector: ~1,350 TWh/yr (1,350,000 GWh), slight upward trend
    base = np.linspace(112_000, 120_000, len(dates))          # GWh/month
    seasonality = 15_000 * np.sin(2 * np.pi * (dates.month - 7) / 12)  # summer peak
    noise = rng.normal(0, 2_000, len(dates))
    df = pd.DataFrame({"ds": dates, "commercial_gwh": base + seasonality + noise})
    df["commercial_gwh"] = df["commercial_gwh"].clip(lower=80_000)
    return df


def compute_co2(df: pd.DataFrame) -> pd.DataFrame:
    """
    commercial_gwh → datacenter_twh → datacenter_co2_mt (megatons)
    """
    df = df.copy()
    kwh = df["commercial_gwh"] * 1e6              # GWh → kWh
    datacenter_kwh   = kwh * DATACENTER_SHARE
    datacenter_kg_co2 = datacenter_kwh * EPA_KG_CO2_PER_KWH
    df["y"] = datacenter_kg_co2 / 1e9             # kg → megatons
    df["datacenter_twh"] = df["commercial_gwh"] * DATACENTER_SHARE / 1e3
    return df[["ds", "y", "datacenter_twh", "commercial_gwh"]]


def main():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("EIA_API_KEY")

    if api_key and api_key != "your_key_here":
        print("📡 Fetching from EIA API...")
        try:
            df = fetch_from_eia(api_key)
            print(f"   {len(df)} monthly records fetched")
        except Exception as e:
            print(f"   ⚠️  EIA API failed ({e}), using seed data")
            df = build_seed_data()
    else:
        print("ℹ️  No EIA_API_KEY — using seed dataset")
        print("   Get a free key at https://www.eia.gov/opendata/register.php")
        df = build_seed_data()

    df = compute_co2(df)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Saved {len(df)} rows → {DATA_PATH}")
    print(f"   Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"   CO2 range:  {df['y'].min():.2f} – {df['y'].max():.2f} Mt/month")
    return df


if __name__ == "__main__":
    main()
