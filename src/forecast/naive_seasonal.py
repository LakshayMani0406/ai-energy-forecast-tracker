#!/usr/bin/env python3
"""
naive_seasonal.py — Seasonal naive baseline forecast.

Forecast for month M of year Y = actual value for month M of year Y-1.
For multi-step forecasting beyond one year, repeats the last full year.

This is the simplest possible baseline: any useful model must beat it.

Usage:
  python src/forecast/naive_seasonal.py
"""
import sys, mlflow, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP     = "ai-energy-forecast-tracker"
MODEL_NAME     = "naive_seasonal"
HOLDOUT_MONTHS = 12
FORECAST_END   = "2030-12-01"

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")


def load_series() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT ds, mean as y
            FROM fusion_posterior
            WHERE variable = 'dc_co2_mt_monthly'
            ORDER BY ds
        """).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def seasonal_naive_forecast(df: pd.DataFrame,
                             holdout: int = HOLDOUT_MONTHS) -> tuple:
    cutoff = df["ds"].max() - pd.DateOffset(months=holdout)
    train  = df[df["ds"] <= cutoff].set_index("ds")["y"]
    test   = df[df["ds"] >  cutoff].set_index("ds")["y"]

    # Build a lookup: month → last observed value for that month
    last_year_start = train.index.max() - pd.DateOffset(months=11)
    last_year = train[train.index >= last_year_start]

    def predict_month(dt):
        matches = last_year[last_year.index.month == dt.month]
        return float(matches.iloc[-1]) if len(matches) > 0 else float(train.iloc[-1])

    # Holdout predictions
    holdout_pred = test.index.map(predict_month)
    mae = mean_absolute_error(test.values, holdout_pred)

    # Full forecast through FORECAST_END
    forecast_dates = pd.date_range(
        start=train.index.max() + pd.DateOffset(months=1),
        end=FORECAST_END,
        freq="MS",
    )
    forecast_vals  = np.array([predict_month(d) for d in forecast_dates])

    return train, test, forecast_dates, forecast_vals, mae


def write_forecasts(train: pd.Series, test: pd.Series,
                    forecast_dates: pd.DatetimeIndex,
                    forecast_vals: np.ndarray, ts: str) -> None:
    # Historical fit (in-sample)
    rows = []
    for ds, y in train.items():
        rows.append({
            "ds": ds.strftime("%Y-%m-%d"), "variable": "dc_co2_mt_monthly",
            "model": MODEL_NAME, "yhat": float(y),
            "yhat_lower": float(y), "yhat_upper": float(y),
            "is_holdout": False, "run_timestamp": ts,
        })
    # Forecast (including holdout window)
    for ds, yhat in zip(forecast_dates, forecast_vals):
        rows.append({
            "ds": ds.strftime("%Y-%m-%d"), "variable": "dc_co2_mt_monthly",
            "model": MODEL_NAME, "yhat": float(yhat),
            "yhat_lower": float(yhat * 0.9), "yhat_upper": float(yhat * 1.1),
            "is_holdout": ds in test.index,
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
        conn.execute("DELETE FROM model_forecasts WHERE model = 'naive_seasonal'")
        conn.execute("""
            INSERT INTO model_forecasts
            SELECT ds::DATE, variable, model, yhat, yhat_lower, yhat_upper,
                   is_holdout, run_timestamp::TIMESTAMP
            FROM incoming
        """)


def main():
    mlflow.set_experiment(MLFLOW_EXP)
    ts = datetime.now(timezone.utc).isoformat()

    with mlflow.start_run(run_name=f"naive-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        print("📊 Seasonal naive baseline...")
        df = load_series()
        train, test, forecast_dates, forecast_vals, mae = seasonal_naive_forecast(df)

        print(f"   Holdout MAE: {mae:.4f} Mt")
        mlflow.log_params({
            "model": MODEL_NAME,
            "method": "last_year_same_month",
            "holdout_months": HOLDOUT_MONTHS,
            "forecast_end": FORECAST_END,
        })
        mlflow.log_metric("national_co2_mae", mae)

        write_forecasts(train, test, forecast_dates, forecast_vals, ts)

        # 2030 projection
        mask_2030 = forecast_dates.year == 2030
        if mask_2030.any():
            print(f"   2030 CO2 projection: {forecast_vals[mask_2030].sum():.1f} Mt/yr")

        print(f"✅ Naive seasonal — holdout MAE: {mae:.4f} Mt | run: {run.info.run_id}")
        return mae


if __name__ == "__main__":
    main()
