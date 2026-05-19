#!/usr/bin/env python3
"""
sarima_model.py — SARIMA(1,1,1)(1,1,1)[12] baseline forecast.

Reads dc_co2_mt_monthly from fusion_posterior, fits a seasonal ARIMA,
forecasts through 2030, logs to MLflow, writes to model_forecasts table.

Usage:
  python src/forecast/sarima_model.py
"""
import sys, warnings, mlflow, numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP     = "ai-energy-forecast-tracker"
MODEL_NAME     = "sarima"
HOLDOUT_MONTHS = 12
FORECAST_END   = "2030-12-01"

ORDER         = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)

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
    df = df.set_index("ds")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df


def train_predict(df: pd.DataFrame, holdout: int = HOLDOUT_MONTHS) -> tuple:
    cutoff = df.index.max() - pd.DateOffset(months=holdout)
    train  = df[df.index <= cutoff]["y"]
    test   = df[df.index >  cutoff]["y"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)

    # Forecast months needed to reach FORECAST_END
    last = df.index.max()
    periods_to_end = (pd.Timestamp(FORECAST_END).year - last.year) * 12 + \
                     (pd.Timestamp(FORECAST_END).month - last.month)
    forecast_periods = holdout + periods_to_end

    fc = model.get_forecast(steps=forecast_periods)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int()

    # Holdout MAE
    holdout_pred = fc_mean[fc_mean.index.isin(test.index)]
    if len(holdout_pred) > 0:
        mae = mean_absolute_error(test.values[:len(holdout_pred)], holdout_pred.values)
    else:
        mae = float("nan")

    return model, fc_mean, fc_ci, mae, test


def write_forecasts(fc_mean: pd.Series, fc_ci: pd.DataFrame,
                    test_index: pd.DatetimeIndex, ts: str) -> None:
    rows = []
    for dt, yhat in fc_mean.items():
        if dt > pd.Timestamp(FORECAST_END):
            continue
        lower = fc_ci.loc[dt].iloc[0] if dt in fc_ci.index else yhat
        upper = fc_ci.loc[dt].iloc[1] if dt in fc_ci.index else yhat
        rows.append({
            "ds": dt.strftime("%Y-%m-%d"),
            "variable": "dc_co2_mt_monthly",
            "model": MODEL_NAME,
            "yhat": float(yhat),
            "yhat_lower": float(lower),
            "yhat_upper": float(upper),
            "is_holdout": dt in test_index,
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
        conn.execute("DELETE FROM model_forecasts WHERE model = 'sarima'")
        conn.execute("""
            INSERT INTO model_forecasts
            SELECT ds::DATE, variable, model, yhat, yhat_lower, yhat_upper,
                   is_holdout, run_timestamp::TIMESTAMP
            FROM incoming
        """)


def main():
    mlflow.set_experiment(MLFLOW_EXP)
    ts = datetime.now(timezone.utc).isoformat()

    with mlflow.start_run(run_name=f"sarima-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        mlflow.log_params({
            "model": MODEL_NAME,
            "order": str(ORDER),
            "seasonal_order": str(SEASONAL_ORDER),
            "holdout_months": HOLDOUT_MONTHS,
            "forecast_end": FORECAST_END,
        })

        print("📈 Fitting SARIMA(1,1,1)(1,1,1)[12]...")
        df = load_series()
        model, fc_mean, fc_ci, mae, test = train_predict(df)

        print(f"   Holdout MAE: {mae:.4f} Mt")
        mlflow.log_metric("national_co2_mae", mae)
        mlflow.log_metric("aic", float(model.aic))

        write_forecasts(fc_mean, fc_ci, test.index, ts)

        # 2030 projection
        fc_2030 = fc_mean[fc_mean.index.year == 2030]
        if not fc_2030.empty:
            annual_2030 = fc_2030.sum()
            print(f"   2030 CO2 projection: {annual_2030:.1f} Mt/yr")

        print(f"✅ SARIMA — holdout MAE: {mae:.4f} Mt | run: {run.info.run_id}")
        return mae


if __name__ == "__main__":
    main()
