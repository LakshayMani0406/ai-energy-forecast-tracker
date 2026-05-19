#!/usr/bin/env python3
"""
train.py — trains a Prophet model on the latest energy data,
logs everything to MLflow, and registers the model in the model registry.

Usage:
  python src/train.py                  # train with defaults
  python src/train.py --changepoint 0.1 --horizon 12
"""
import argparse, os, sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.prophet
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from datetime import datetime

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "energy_data.csv"
EXPERIMENT_NAME  = "ai-energy-forecast-tracker"
REGISTERED_MODEL = "ai-energy-forecast-model"

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("No data found — running fetch_data.py first...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "src" / "ingest" / "eia.py")], check=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    return df.sort_values("ds").reset_index(drop=True)


def train_evaluate(df: pd.DataFrame, changepoint_prior: float,
                   seasonality_mode: str, horizon: int):
    """Train Prophet, return model + metrics on holdout."""
    cutoff = df["ds"].max() - pd.DateOffset(months=horizon)
    train  = df[df["ds"] <= cutoff][["ds", "y"]]
    test   = df[df["ds"] >  cutoff][["ds", "y"]]

    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)

    # Align predictions with actuals on holdout period
    pred = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat"]].set_index("ds")
    actual = test.set_index("ds")["y"]
    aligned = actual.align(pred["yhat"], join="inner")

    mae  = mean_absolute_error(aligned[0], aligned[1])
    rmse = np.sqrt(mean_squared_error(aligned[0], aligned[1]))
    mape = (np.abs((aligned[0] - aligned[1]) / aligned[0]).mean()) * 100

    metrics = {"mae": mae, "rmse": rmse, "mape": mape,
               "train_rows": len(train), "test_rows": len(test)}
    return model, forecast, metrics


def main(changepoint_prior: float = 0.05, seasonality_mode: str = "additive",
         horizon: int = 12):
    df = load_data()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"prophet-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        params = {
            "changepoint_prior_scale": changepoint_prior,
            "seasonality_mode": seasonality_mode,
            "forecast_horizon_months": horizon,
            "data_end": str(df["ds"].max().date()),
        }
        mlflow.log_params(params)

        print(f"🔬 Training Prophet (changepoint={changepoint_prior}, "
              f"seasonality={seasonality_mode}, horizon={horizon}m)...")
        model, forecast, metrics = train_evaluate(df, changepoint_prior, seasonality_mode, horizon)

        mlflow.log_metrics(metrics)
        print(f"   MAE:  {metrics['mae']:.4f} Mt")
        print(f"   RMSE: {metrics['rmse']:.4f} Mt")
        print(f"   MAPE: {metrics['mape']:.2f}%")

        # Log model to registry and transition to Staging
        model_info = mlflow.prophet.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL,
        )
        client = mlflow.tracking.MlflowClient()
        # Get the version that was just registered
        versions = client.get_latest_versions(REGISTERED_MODEL)
        if versions:
            latest_v = max(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                REGISTERED_MODEL, latest_v.version, "Staging"
            )
            print(f"   Transitioned v{latest_v.version} → Staging")

        # Save forecast CSV as artifact
        forecast_path = ROOT / "data" / "raw" / "latest_forecast.csv"
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_path, index=False)
        mlflow.log_artifact(str(forecast_path))

        print(f"✅ Run logged: {run.info.run_id}")
        print(f"   Model registered as '{REGISTERED_MODEL}'")
        return run.info.run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--changepoint", type=float, default=0.05)
    parser.add_argument("--seasonality", type=str, default="additive",
                        choices=["additive", "multiplicative"])
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    main(args.changepoint, args.seasonality, args.horizon)
