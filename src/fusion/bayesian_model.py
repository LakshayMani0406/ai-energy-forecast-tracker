#!/usr/bin/env python3
"""
bayesian_model.py — Hierarchical Bayesian fusion model for US datacenter energy.

The latent quantity is monthly datacenter electricity consumption, decomposed
into AI and non-AI workloads.  Each data source contributes as a measurement
with its own noise model and lag.

Current data sources wired in:
  EIA commercial electricity (DuckDB: eia_commercial_monthly) — national anchor
  eGRID state CO2 emission rates (DuckDB: egrid_state_yearly) — carbon accounting
  LBNL 2024 (Shehabi et al.) — prior on datacenter share
  Guidi et al. 2024 benchmark — 31.5 Mt CO2 in 2018
  IEA Energy and AI 2025 — 183 TWh and 105 Mt CO2 in 2024

When FERC and hyperscaler capex data become available, add them as
additional observation nodes in the same model (see Step 3 / Step 5 comments).

Usage:
  python src/fusion/bayesian_model.py          # sample + export
  python src/fusion/bayesian_model.py --chains 4 --draws 2000
"""
import argparse, sys
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import mlflow
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP = "ai-energy-forecast-tracker"
mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")

# ---------------------------------------------------------------------------
# Datacenter capacity weights by state (approx., LBNL 2024 + JLL 2024)
# These are used to compute a DC-weighted grid emission rate, which is higher
# than the national avg because VA/TX/GA are fossil-heavier grids.
# ---------------------------------------------------------------------------
STATE_DC_WEIGHTS = {
    "VA": 0.350,   # Northern Virginia — largest market globally
    "TX": 0.085,
    "CA": 0.070,
    "GA": 0.055,   # Atlanta metro
    "OH": 0.040,
    "IL": 0.038,
    "AZ": 0.032,
    "WA": 0.028,
    "OR": 0.022,
    "NY": 0.020,
    "NJ": 0.015,
    "NC": 0.012,
    "FL": 0.010,
}
# Remainder allocated to "other" bucket using national avg emission rate

# Benchmark observations (from master_dataset.csv in the old repo)
# These are hard annual constraints that anchor the model.
BENCHMARKS = {
    "energy_2024_twh":  183.0,   # IEA Energy and AI 2025
    "co2_2024_mt":      105.0,   # IEA Energy and AI 2025 (Guidi et al. corroboration)
    "co2_2018_mt":       31.5,   # Guidi et al. 2024 (2,132 data centers analyzed)
}


def load_eia(conn) -> pd.DataFrame:
    return conn.execute(
        "SELECT ds, y, datacenter_twh, commercial_gwh FROM eia_commercial_monthly ORDER BY ds"
    ).df()


def load_egrid_national(conn) -> pd.DataFrame:
    """Datacenter-capacity-weighted national CO2 rate by year from eGRID state data."""
    states = list(STATE_DC_WEIGHTS.keys())
    placeholders = ", ".join(f"'{s}'" for s in states)

    # Weight only the known DC states; rest gets national avg weight
    df_states = conn.execute(f"""
        SELECT year, state, co2_rate_g_per_kwh, COALESCE(net_gen_mwh, 1e9) as gen_mwh
        FROM egrid_state_yearly
        WHERE state IN ({placeholders})
        ORDER BY year, state
    """).df()

    df_natl = conn.execute("""
        SELECT year, AVG(co2_rate_g_per_kwh) as natl_avg_g_kwh
        FROM egrid_state_yearly GROUP BY year ORDER BY year
    """).df()

    rows = []
    for year in df_states["year"].unique():
        yr_states = df_states[df_states["year"] == year]
        natl_avg  = df_natl[df_natl["year"] == year]["natl_avg_g_kwh"].values[0]

        # Compute weighted avg: DC-states weighted by DC capacity share
        known_weight = sum(STATE_DC_WEIGHTS[s] for s in yr_states["state"])
        other_weight = 1.0 - known_weight
        dc_weighted_rate = (
            sum(
                STATE_DC_WEIGHTS[row["state"]] * row["co2_rate_g_per_kwh"]
                for _, row in yr_states.iterrows()
            )
            + other_weight * natl_avg
        )
        rows.append({"year": year, "dc_weighted_g_kwh": dc_weighted_rate, "natl_avg_g_kwh": natl_avg})

    return pd.DataFrame(rows)


def build_co2_rate_series(egrid: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Interpolate/extrapolate eGRID NATIONAL AVERAGE CO2 rate to monthly resolution.
    Uses natl_avg_g_kwh (not DC-weighted) because the IEA CO2 benchmark (105 Mt)
    implicitly uses national average methodology with PUE overhead.
    Pre-2020/post-2023: extrapolate using the observed decarbonisation linear trend.
    """
    years = egrid["year"].values.astype(float)
    rates = egrid["natl_avg_g_kwh"].values   # national average, not DC-weighted

    slope    = np.polyfit(years, rates, 1)
    trend_fn = np.poly1d(slope)

    month_years = dates.dt.year + (dates.dt.month - 1) / 12.0
    return trend_fn(month_years)


def run_model(draws: int = 1000, tune: int = 1000, chains: int = 2, random_seed: int = 42):
    with get_conn() as conn:
        eia    = load_eia(conn)
        egrid  = load_egrid_national(conn)

    dates          = pd.to_datetime(eia["ds"])
    commercial_gwh = eia["commercial_gwh"].values
    T              = len(commercial_gwh)

    # Monthly decimal year (e.g., 2001.0 = Jan 2001)
    month_year = dates.dt.year.values + (dates.dt.month.values - 1) / 12.0

    # DC-weighted CO2 rate series (g/kWh) — deterministic from eGRID + trend
    co2_rate_series = build_co2_rate_series(egrid, dates)

    # Index of specific months needed for benchmark likelihood
    def _idx(year: int, month: int = 7) -> int:
        matches = np.where((dates.dt.year == year) & (dates.dt.month == month))[0]
        return int(matches[0]) if len(matches) else int(np.argmin(np.abs(month_year - year)))

    idx_2018 = _idx(2018)
    idx_2024 = _idx(2024)

    print(f"\n📐 Model dimensions: {T} months ({dates.iloc[0].date()} → {dates.iloc[-1].date()})")
    print(f"   National avg CO2 rate 2020–2023 (g/kWh): "
          f"{egrid['natl_avg_g_kwh'].values}")
    print(f"   Benchmark indices: 2018={idx_2018}, 2024={idx_2024}")
    print(f"   CO2 rate at 2024 benchmark: {co2_rate_series[idx_2024]:.1f} g/kWh")

    # Verification: at LBNL dc_share=3.5%, natl avg 366 g/kWh, PUE 1.5:
    # dc_total = commercial_2024 * 0.035  (total facility, incl. cooling)
    # dc_it    = dc_total / 1.5           (IT load only)
    # CO2      = dc_total * 366e-6 Mt/GWh
    # Constraint: dc_it * 12 ~ 183 TWh  and  CO2 * 12 ~ 105 Mt
    # => PUE = dc_total / dc_it = (183 / dc_share_implied) * 12...
    # The model infers PUE jointly with dc_share from both benchmarks.

    # ------------------------------------------------------------------
    # PyMC model — reparameterised for convergence
    # dc_share: linear trend (avoids logistic k/t0 correlation)
    # pue:      latent, reconciles energy (IT load) vs CO2 benchmarks
    # ------------------------------------------------------------------
    with pm.Model() as fusion_model:
        # 1. Datacenter share of US commercial electricity — linear trend
        #    LBNL 2024 anchors share at 3.5% in 2024.
        #    We parameterise as (share_2024, slope) and interpolate.
        dc_share_2024  = pm.Beta("dc_share_2024", alpha=35, beta=965, initval=0.035)
        dc_share_slope = pm.Normal("dc_share_slope", mu=0.001, sigma=0.0005, initval=0.001)
        dc_share = dc_share_2024 + dc_share_slope * (month_year - 2024.0)
        dc_share = pm.math.clip(dc_share, 0.005, 0.20)   # physical bounds

        # 2. Power Usage Effectiveness (PUE) — total facility / IT load
        #    Typical 2024 US average: 1.4–1.6 (EPA/Uptime Institute data)
        #    The model infers PUE jointly from the energy benchmark (IT load)
        #    and the CO2 benchmark (total facility × grid emission rate).
        pue = pm.TruncatedNormal("pue", mu=1.5, sigma=0.15, lower=1.1, upper=2.2,
                                  initval=1.5)

        # 3. AI fraction of total datacenter electricity — structural break at 2023
        ai_frac_base   = pm.Beta("ai_frac_base", alpha=5, beta=95, initval=0.05)
        ai_frac_growth = pm.HalfNormal("ai_frac_growth", sigma=0.4, initval=0.3)
        years_post_2023 = pm.math.maximum(0.0, month_year - 2023.0)
        ai_frac = ai_frac_base + (1 - ai_frac_base) * (
            1 - pm.math.exp(-ai_frac_growth * years_post_2023)
        )

        # 4. Derived energy quantities
        #    dc_gwh     = total facility energy (IT + cooling) — what EIA captures
        #    dc_it_gwh  = IT load only (183 TWh IEA benchmark is IT-only)
        dc_gwh     = pm.Deterministic("dc_gwh",     commercial_gwh * dc_share)
        dc_it_gwh  = pm.Deterministic("dc_it_gwh",  dc_gwh / pue)
        ai_gwh     = pm.Deterministic("ai_gwh",     dc_gwh * ai_frac)
        non_ai_gwh = pm.Deterministic("non_ai_gwh", dc_gwh * (1 - ai_frac))
        dc_twh     = pm.Deterministic("dc_twh",     dc_gwh / 1000.0)

        # 5. CO2 from total facility energy × national avg emission rate
        #    dc_gwh [GWh] × co2_rate [g/kWh] × 1e6 [kWh/GWh] / 1e12 [g/Mt]
        #    = dc_gwh × co2_rate × 1e-6  [Mt/month]
        dc_co2_mt_monthly = pm.Deterministic(
            "dc_co2_mt_monthly",
            dc_gwh * co2_rate_series * 1e-6
        )

        # Annual aggregates at benchmark years (mid-year month × 12 approximation)
        dc_it_twh_annual_2024  = dc_it_gwh[idx_2024]  / 1000.0 * 12   # IT TWh/yr
        dc_co2_mt_annual_2024  = dc_co2_mt_monthly[idx_2024] * 12
        dc_co2_mt_annual_2018  = dc_co2_mt_monthly[idx_2018] * 12

        # 6. Observation likelihoods (benchmark constraints)
        sigma_energy = pm.HalfNormal("sigma_energy", sigma=15.0, initval=10.0)
        sigma_co2    = pm.HalfNormal("sigma_co2",    sigma=10.0, initval=8.0)

        # IEA 2025: 183 TWh IT load in 2024 (obs on IT load, not total facility)
        pm.Normal("obs_energy_2024",
                  mu=dc_it_twh_annual_2024, sigma=sigma_energy,
                  observed=BENCHMARKS["energy_2024_twh"])

        # IEA 2025: 105 Mt CO2 from datacenters in 2024 (total facility)
        pm.Normal("obs_co2_2024",
                  mu=dc_co2_mt_annual_2024, sigma=sigma_co2,
                  observed=BENCHMARKS["co2_2024_mt"])

        # Guidi et al. 2024: 31.5 Mt CO2 in 2018
        pm.Normal("obs_co2_2018",
                  mu=dc_co2_mt_annual_2018, sigma=sigma_co2,
                  observed=BENCHMARKS["co2_2018_mt"])

    print(f"\n🔬 Sampling (chains={chains}, draws={draws}, tune={tune})...")
    with fusion_model:
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=random_seed, progressbar=True,
            nuts_sampler="numpyro",
            target_accept=0.95,
        )

    return trace, eia, egrid, co2_rate_series, dates


def summarise_posterior(trace, eia: pd.DataFrame,
                         co2_rate_series: np.ndarray,
                         dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract posterior mean ± CI for each month, plus state decomposition."""
    vars_ = ["dc_gwh", "ai_gwh", "non_ai_gwh", "dc_twh", "dc_co2_mt_monthly"]
    rows  = []

    for v in vars_:
        samples = trace.posterior[v].values  # (chains, draws, T)
        flat    = samples.reshape(-1, samples.shape[-1])
        rows.append({
            "variable": v,
            "mean":  flat.mean(axis=0),
            "p2_5":  np.percentile(flat, 2.5,  axis=0),
            "p50":   np.percentile(flat, 50,   axis=0),
            "p97_5": np.percentile(flat, 97.5, axis=0),
        })

    # Build a long-form DataFrame: one row per (date, variable)
    records = []
    for r in rows:
        for i, ds in enumerate(dates):
            records.append({
                "ds":       ds.strftime("%Y-%m-%d"),
                "variable": r["variable"],
                "mean":     float(r["mean"][i]),
                "p2_5":     float(r["p2_5"][i]),
                "p50":      float(r["p50"][i]),
                "p97_5":    float(r["p97_5"][i]),
                "co2_rate_g_kwh": float(co2_rate_series[i]),
            })

    # Add state decomposition columns (post-processing, no MCMC needed)
    # Use posterior mean dc_gwh × fixed state weights
    dc_mean = rows[0]["mean"]  # dc_gwh mean series
    for state, w in STATE_DC_WEIGHTS.items():
        for i, ds in enumerate(dates):
            records.append({
                "ds":       ds.strftime("%Y-%m-%d"),
                "variable": f"state_dc_gwh_{state}",
                "mean":     float(dc_mean[i] * w),
                "p2_5":     float(rows[0]["p2_5"][i] * w),
                "p50":      float(rows[0]["p50"][i] * w),
                "p97_5":    float(rows[0]["p97_5"][i] * w),
                "co2_rate_g_kwh": float(co2_rate_series[i]),
            })

    return pd.DataFrame(records)


def write_to_duckdb(summary: pd.DataFrame) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    summary = summary.copy()
    summary["fetch_timestamp"] = ts

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fusion_posterior (
                ds              DATE,
                variable        VARCHAR,
                mean            DOUBLE,
                p2_5            DOUBLE,
                p50             DOUBLE,
                p97_5           DOUBLE,
                co2_rate_g_kwh  DOUBLE,
                fetch_timestamp TIMESTAMP,
                PRIMARY KEY (ds, variable)
            )
        """)
        conn.register("incoming", summary)
        conn.execute("DELETE FROM fusion_posterior WHERE ds IN (SELECT ds::DATE FROM incoming)")
        conn.execute("""
            INSERT INTO fusion_posterior
            SELECT ds::DATE, variable, mean, p2_5, p50, p97_5, co2_rate_g_kwh,
                   fetch_timestamp::TIMESTAMP
            FROM incoming
        """)
        n = conn.execute("SELECT COUNT(*) FROM fusion_posterior").fetchone()[0]
    print(f"   ✅ fusion_posterior: {n} rows written to warehouse")


def print_sanity_check(summary: pd.DataFrame) -> None:
    """Print key annual aggregates for sanity-checking against published estimates."""
    df = summary[summary["variable"] == "dc_gwh"].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["year"] = df["ds"].dt.year

    co2_df = summary[summary["variable"] == "dc_co2_mt_monthly"].copy()
    co2_df["ds"] = pd.to_datetime(co2_df["ds"])
    co2_df["year"] = co2_df["ds"].dt.year

    ai_df = summary[summary["variable"] == "ai_gwh"].copy()
    ai_df["ds"] = pd.to_datetime(ai_df["ds"])
    ai_df["year"] = ai_df["ds"].dt.year

    print("\n" + "="*70)
    print("POSTERIOR SANITY CHECK — annual aggregates")
    print("="*70)
    print(f"{'Year':>4}  {'DC Energy TWh':>14}  {'DC CO2 Mt':>10}  {'AI Share %':>10}  {'95% CI CO2':>14}")
    print("-"*70)

    for year in sorted(df["year"].unique()):
        if year < 2014 or year > 2026:
            continue
        twh_mean  = df[df["year"] == year]["mean"].sum() / 1000
        twh_lo    = df[df["year"] == year]["p2_5"].sum() / 1000
        twh_hi    = df[df["year"] == year]["p97_5"].sum() / 1000

        co2_mean  = co2_df[co2_df["year"] == year]["mean"].sum()
        co2_lo    = co2_df[co2_df["year"] == year]["p2_5"].sum()
        co2_hi    = co2_df[co2_df["year"] == year]["p97_5"].sum()

        ai_frac   = (ai_df[ai_df["year"] == year]["mean"].sum()
                     / df[df["year"] == year]["mean"].sum()) * 100

        print(f"{year:>4}  {twh_mean:>8.1f} TWh    {co2_mean:>8.1f} Mt   "
              f"{ai_frac:>8.1f}%   [{co2_lo:6.1f} – {co2_hi:6.1f}]")

    print("-"*70)
    print("Published benchmarks:")
    print("  LBNL 2024: 62.2 Mt (2024, using 340 g/kWh flat — known underestimate)")
    print("  IEA 2025:  105.0 Mt CO2 / 183 TWh (2024)")
    print("  Guidi et al.: 31.5 Mt (2018)")
    print("="*70)


def main(draws=1000, tune=1000, chains=2):
    mlflow.set_experiment(MLFLOW_EXP)

    with mlflow.start_run(run_name=f"fusion-bayesian-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        mlflow.log_params({
            "model":         "hierarchical_bayesian",
            "sampler":       "NUTS (numpyro)",
            "draws":         draws,
            "tune":          tune,
            "chains":        chains,
            "data_sources":  "EIA,eGRID,LBNL-prior,IEA-2025,Guidi-2024",
            "priors":        "LBNL2024-informed logistic dc_share, structural-break ai_frac",
        })

        trace, eia, egrid, co2_rate_series, dates = run_model(draws, tune, chains)

        # Posterior parameter summary
        params_summary = az.summary(trace, var_names=[
            "dc_share_2024", "dc_share_slope", "pue",
            "ai_frac_base", "ai_frac_growth",
            "sigma_energy", "sigma_co2",
        ])
        print("\n📊 Posterior parameter summary:")
        print(params_summary.to_string())

        # Log scalar posterior stats to MLflow
        for param in ["dc_share_2024", "dc_share_slope", "pue",
                      "ai_frac_base", "ai_frac_growth"]:
            vals = trace.posterior[param].values.flatten()
            mlflow.log_metrics({
                f"{param}_mean":  float(vals.mean()),
                f"{param}_sd":    float(vals.std()),
                f"{param}_r_hat": float(params_summary.loc[param, "r_hat"]),
            })

        # Build posterior summary DataFrame
        summary = summarise_posterior(trace, eia, co2_rate_series, dates)
        write_to_duckdb(summary)

        # Save trace artifact
        trace_path = ROOT / "data" / "exports" / "fusion_trace.nc"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace.to_netcdf(str(trace_path))
        mlflow.log_artifact(str(trace_path))

        print_sanity_check(summary)
        print(f"\n✅ Fusion model run complete — MLflow run: {run.info.run_id}")

    return trace, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws",  type=int, default=1000)
    parser.add_argument("--tune",   type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()
    main(args.draws, args.tune, args.chains)
