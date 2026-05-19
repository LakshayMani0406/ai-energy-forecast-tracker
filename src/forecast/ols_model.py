#!/usr/bin/env python3
"""
ols_model.py — OLS regression baseline.

Ports the regression spec from the old repo's phase2_regression.rmd:
  CO2 ~ DC_Energy + Grid_CO2_Rate

Uses annual aggregates from fusion_posterior + eGRID (matching the original
11-year panel dataset). Forecasts by projecting both predictors forward using
their own linear trends, then applying the fitted OLS coefficients.

Logs R², coefficients, and holdout MAE to MLflow.
Writes forecasts to model_forecasts table in DuckDB.

Usage:
  python src/forecast/ols_model.py
"""
import sys, mlflow, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP     = "ai-energy-forecast-tracker"
MODEL_NAME     = "ols"
HOLDOUT_MONTHS = 12
FORECAST_END   = "2030-12-01"

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")


def load_annual_data() -> pd.DataFrame:
    """Aggregate fusion_posterior monthly values to annual, join with eGRID."""
    with get_conn() as conn:
        # Annual CO2 and energy from fusion posterior
        energy_co2 = conn.execute("""
            SELECT year(ds) as year,
                   SUM(CASE WHEN variable='dc_co2_mt_monthly' THEN mean END) as co2_mt,
                   SUM(CASE WHEN variable='dc_gwh' THEN mean END) / 1000.0 as dc_twh
            FROM fusion_posterior
            WHERE variable IN ('dc_co2_mt_monthly', 'dc_gwh')
            GROUP BY 1
            HAVING COUNT(*) = 24
            ORDER BY 1
        """).df()

        # National avg CO2 rate from eGRID
        co2_rates = conn.execute("""
            SELECT year, AVG(co2_rate_g_per_kwh) as co2_rate_g_kwh
            FROM egrid_state_yearly GROUP BY year ORDER BY year
        """).df()

    # Join on year; for years without eGRID, interpolate/extrapolate linearly
    df = energy_co2.merge(co2_rates, on="year", how="left")
    if df["co2_rate_g_kwh"].isna().any():
        # Linear fill: fit trend on available eGRID years
        known = df.dropna(subset=["co2_rate_g_kwh"])
        if len(known) >= 2:
            slope = np.polyfit(known["year"], known["co2_rate_g_kwh"], 1)
            fn    = np.poly1d(slope)
            df["co2_rate_g_kwh"] = df["co2_rate_g_kwh"].fillna(
                df["year"].apply(lambda y: fn(y))
            )
    return df.dropna()


def train_predict(df: pd.DataFrame, holdout: int = HOLDOUT_MONTHS) -> tuple:
    # Convert holdout months → years (round up)
    holdout_years = max(1, holdout // 12)
    cutoff_year   = int(df["year"].max()) - holdout_years
    train = df[df["year"] <= cutoff_year]
    test  = df[df["year"] >  cutoff_year]

    X_train = train[["dc_twh", "co2_rate_g_kwh"]].values
    y_train = train["co2_mt"].values
    X_test  = test[["dc_twh", "co2_rate_g_kwh"]].values
    y_test  = test["co2_mt"].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2  = r2_score(y_train, model.predict(X_train))
    mae = mean_absolute_error(y_test, model.predict(X_test)) if len(y_test) > 0 else float("nan")

    return model, r2, mae, train, test, df


def forecast_to_2030(model: LinearRegression, df: pd.DataFrame) -> pd.DataFrame:
    """
    Project predictors forward using their own linear trends,
    apply OLS coefficients to produce monthly CO2 forecasts through 2030.
    """
    last_year = int(df["year"].max())
    future_years = np.arange(last_year + 1, 2031)

    # Trend-project each predictor
    slope_energy = np.polyfit(df["year"], df["dc_twh"], 1)
    slope_rate   = np.polyfit(df["year"], df["co2_rate_g_kwh"], 1)

    fn_energy = np.poly1d(slope_energy)
    fn_rate   = np.poly1d(slope_rate)

    rows = []
    for yr in future_years:
        dc_twh_proj  = fn_energy(yr)
        co2_rate_proj = fn_rate(yr)
        co2_annual    = model.predict([[dc_twh_proj, co2_rate_proj]])[0]
        rows.append({"year": yr, "dc_twh": dc_twh_proj,
                     "co2_rate_g_kwh": co2_rate_proj, "co2_mt": co2_annual})

    future_df = pd.DataFrame(rows)
    # Convert annual CO2 to monthly (flat distribution across months)
    monthly_rows = []
    for _, row in pd.concat([df, future_df]).iterrows():
        for month in range(1, 13):
            ds = pd.Timestamp(year=int(row["year"]), month=month, day=1)
            monthly_rows.append({"ds": ds, "yhat": row["co2_mt"] / 12.0})

    return pd.DataFrame(monthly_rows)


def write_forecasts(fc_monthly: pd.DataFrame,
                    test: pd.DataFrame,
                    ts: str) -> None:
    test_years = set(test["year"].astype(int))
    rows = []
    for _, row in fc_monthly.iterrows():
        if row["ds"] > pd.Timestamp(FORECAST_END):
            continue
        rows.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "variable": "dc_co2_mt_monthly",
            "model": MODEL_NAME,
            "yhat": float(row["yhat"]),
            "yhat_lower": float(row["yhat"] * 0.85),
            "yhat_upper": float(row["yhat"] * 1.15),
            "is_holdout": row["ds"].year in test_years,
            "run_timestamp": ts,
        })
    df = pd.DataFrame(rows)
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_forecasts (
                ds DATE, variable VARCHAR, model VARCHAR,
                yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE,
                is_holdout BOOLEAN, run_timestamp TIMESTAMP,
                PRIMARY KEY (ds, variable, model)
            )
        """)
        conn.register("incoming", df)
        conn.execute("DELETE FROM model_forecasts WHERE model = 'ols'")
        conn.execute("""
            INSERT INTO model_forecasts
            SELECT ds::DATE, variable, model, yhat, yhat_lower, yhat_upper,
                   is_holdout, run_timestamp::TIMESTAMP
            FROM incoming
        """)


def main():
    mlflow.set_experiment(MLFLOW_EXP)
    ts = datetime.now(timezone.utc).isoformat()

    with mlflow.start_run(run_name=f"ols-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        print("📐 Fitting OLS (CO2 ~ DC_Energy + Grid_CO2_Rate)...")
        df = load_annual_data()
        print(f"   Training data: {int(df['year'].min())}–{int(df['year'].max())} "
              f"({len(df)} annual rows)")

        model, r2, mae, train, test, full_df = train_predict(df)

        print(f"   R²   (train): {r2:.4f}")
        print(f"   Coefficients: DC_Energy={model.coef_[0]:.4f} Mt/TWh, "
              f"CO2_Rate={model.coef_[1]:.4f} Mt/(g/kWh)")
        print(f"   Intercept:    {model.intercept_:.4f}")
        print(f"   Holdout MAE:  {mae:.4f} Mt/yr")

        mlflow.log_params({
            "model": MODEL_NAME,
            "formula": "CO2 ~ DC_Energy_TWh + Grid_CO2_Rate_g_kWh",
            "train_years": f"{int(train['year'].min())}–{int(train['year'].max())}",
            "holdout_months": HOLDOUT_MONTHS,
        })
        mlflow.log_metrics({
            "r2_train": r2,
            "national_co2_mae": mae,
            "coef_dc_energy": float(model.coef_[0]),
            "coef_co2_rate":  float(model.coef_[1]),
            "intercept": float(model.intercept_),
        })

        fc_monthly = forecast_to_2030(model, full_df)
        write_forecasts(fc_monthly, test, ts)

        fc_2030 = fc_monthly[fc_monthly["ds"].dt.year == 2030]
        if not fc_2030.empty:
            print(f"   2030 CO2 projection: {fc_2030['yhat'].sum():.1f} Mt/yr")

        print(f"✅ OLS — R²={r2:.4f}, holdout MAE={mae:.4f} Mt | run: {run.info.run_id}")
        return mae


if __name__ == "__main__":
    main()
