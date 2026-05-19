#!/usr/bin/env python3
"""
prophet_model.py — Prophet forecast on fusion_posterior dc_co2_mt_monthly.

Produces:
  - National monthly CO2 forecast through 2030
  - State-level DC energy forecasts for top-13 states through 2030
  - Logs to MLflow; writes predictions to model_forecasts table in DuckDB

Usage:
  python src/forecast/prophet_model.py
"""
import sys, mlflow, mlflow.prophet, numpy as np, pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP    = "ai-energy-forecast-tracker"
MODEL_NAME    = "prophet"
HOLDOUT_MONTHS = 12
FORECAST_END  = "2030-12-01"

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")

TOP_STATES = [
    "VA", "TX", "CA", "GA", "OH", "IL",
    "AZ", "WA", "OR", "NY", "NJ", "NC", "FL",
]


def load_series(variable: str = "dc_co2_mt_monthly") -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT ds, mean as y
            FROM fusion_posterior
            WHERE variable = '{variable}'
            ORDER BY ds
        """).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def train_predict(df: pd.DataFrame, holdout: int = HOLDOUT_MONTHS,
                  changepoint_prior: float = 0.05) -> tuple:
    cutoff = df["ds"].max() - pd.DateOffset(months=holdout)
    train  = df[df["ds"] <= cutoff][["ds", "y"]]
    test   = df[df["ds"] >  cutoff][["ds", "y"]]

    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(train)

    # Holdout + forecast through 2030
    last = df["ds"].max()
    months_to_2030 = (pd.Timestamp(FORECAST_END).year - last.year) * 12 + \
                     (pd.Timestamp(FORECAST_END).month - last.month) + holdout
    future   = model.make_future_dataframe(periods=months_to_2030, freq="MS")
    forecast = model.predict(future)

    # Holdout MAE
    pred_hold = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat"]].set_index("ds")
    actual    = test.set_index("ds")["y"]
    aligned   = actual.align(pred_hold["yhat"], join="inner")
    mae = mean_absolute_error(aligned[0], aligned[1]) if len(aligned[0]) > 0 else float("nan")

    return model, forecast, mae, train, test


def write_forecasts(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_forecasts (
                ds            DATE,
                variable      VARCHAR,
                model         VARCHAR,
                yhat          DOUBLE,
                yhat_lower    DOUBLE,
                yhat_upper    DOUBLE,
                is_holdout    BOOLEAN,
                run_timestamp TIMESTAMP,
                PRIMARY KEY (ds, variable, model)
            )
        """)
        conn.register("incoming", df)
        conn.execute("""
            DELETE FROM model_forecasts
            WHERE model = 'prophet'
              AND variable IN (SELECT DISTINCT variable FROM incoming)
        """)
        conn.execute("""
            INSERT INTO model_forecasts
            SELECT ds::DATE, variable, model, yhat, yhat_lower, yhat_upper,
                   is_holdout, run_timestamp::TIMESTAMP
            FROM incoming
        """)


def main():
    mlflow.set_experiment(MLFLOW_EXP)
    ts = datetime.now(timezone.utc).isoformat()

    with mlflow.start_run(run_name=f"prophet-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        mlflow.log_params({
            "model": "prophet",
            "holdout_months": HOLDOUT_MONTHS,
            "forecast_end": FORECAST_END,
            "changepoint_prior_scale": 0.05,
            "source": "fusion_posterior",
        })

        forecast_rows = []

        # ── National CO2 forecast ──────────────────────────────────────────
        print("🔮 National CO2 forecast...")
        df_co2 = load_series("dc_co2_mt_monthly")
        model, fc, mae, train, test = train_predict(df_co2)
        print(f"   Holdout MAE: {mae:.4f} Mt")
        mlflow.log_metric("national_co2_mae", mae)

        holdout_ds = set(test["ds"].dt.strftime("%Y-%m-%d"))
        for _, row in fc.iterrows():
            ds_str = row["ds"].strftime("%Y-%m-%d")
            if row["ds"] > pd.Timestamp(FORECAST_END):
                continue
            forecast_rows.append({
                "ds": ds_str, "variable": "dc_co2_mt_monthly", "model": MODEL_NAME,
                "yhat": row["yhat"], "yhat_lower": row["yhat_lower"],
                "yhat_upper": row["yhat_upper"],
                "is_holdout": ds_str in holdout_ds,
                "run_timestamp": ts,
            })

        # Log model to MLflow registry
        from prophet import serialize
        model_info = mlflow.prophet.log_model(
            model, artifact_path="prophet_co2",
            registered_model_name="ai-energy-forecast-model",
        )

        # ── National energy (dc_gwh) forecast ─────────────────────────────
        print("🔮 National energy forecast...")
        df_gwh = load_series("dc_gwh")
        _, fc_gwh, mae_gwh, _, _ = train_predict(df_gwh)
        mlflow.log_metric("national_energy_mae", mae_gwh)

        for _, row in fc_gwh.iterrows():
            ds_str = row["ds"].strftime("%Y-%m-%d")
            if row["ds"] > pd.Timestamp(FORECAST_END):
                continue
            forecast_rows.append({
                "ds": ds_str, "variable": "dc_gwh", "model": MODEL_NAME,
                "yhat": row["yhat"], "yhat_lower": row["yhat_lower"],
                "yhat_upper": row["yhat_upper"],
                "is_holdout": False,
                "run_timestamp": ts,
            })

        # ── State-level DC energy forecasts ───────────────────────────────
        state_maes = {}
        for state in TOP_STATES:
            var = f"state_dc_gwh_{state}"
            print(f"   {state}...", end="", flush=True)
            df_s = load_series(var)
            if df_s.empty or df_s["y"].sum() == 0:
                print(" skipped (no data)")
                continue
            _, fc_s, mae_s, _, _ = train_predict(df_s)
            state_maes[state] = mae_s
            print(f" MAE={mae_s:.1f} GWh")

            for _, row in fc_s.iterrows():
                ds_str = row["ds"].strftime("%Y-%m-%d")
                if row["ds"] > pd.Timestamp(FORECAST_END):
                    continue
                forecast_rows.append({
                    "ds": ds_str, "variable": var, "model": MODEL_NAME,
                    "yhat": max(0, row["yhat"]), "yhat_lower": row["yhat_lower"],
                    "yhat_upper": row["yhat_upper"],
                    "is_holdout": False,
                    "run_timestamp": ts,
                })

        if state_maes:
            mlflow.log_metric("avg_state_mae_gwh",
                              float(np.mean(list(state_maes.values()))))

        write_forecasts(forecast_rows)
        print(f"\n✅ Prophet — national CO2 MAE: {mae:.4f} Mt | "
              f"{len(forecast_rows)} rows → model_forecasts")
        print(f"   MLflow run: {run.info.run_id}")

        # 2030 projection print
        fc_2030 = fc[fc["ds"].dt.year == 2030]
        if not fc_2030.empty:
            annual_2030 = fc_2030["yhat"].sum()
            print(f"   2030 CO2 projection: {annual_2030:.1f} Mt/yr "
                  f"(95% CI: {fc_2030['yhat_lower'].sum():.1f}–"
                  f"{fc_2030['yhat_upper'].sum():.1f})")

        return mae


if __name__ == "__main__":
    main()
