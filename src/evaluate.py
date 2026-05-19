#!/usr/bin/env python3
"""
evaluate.py — promotion logic.

Compares the latest trained model (Staging) against the current
Production model on a 6-month holdout. Promotes if new model
is ≥5% better on MAE. Logs the decision to MLflow.

Usage:
  python src/evaluate.py
"""
import sys, pandas as pd, numpy as np, mlflow, mlflow.prophet
from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "energy_data.csv"
EXPERIMENT_NAME  = "ai-energy-forecast-tracker"
REGISTERED_MODEL = "ai-energy-forecast-model"
HOLDOUT_MONTHS   = 6
PROMOTION_THRESHOLD = 0.95   # new MAE must be < 95% of prod MAE to promote

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")


def get_model_version(stage: str):
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL, stages=[stage])
    return versions[0] if versions else None


def score_model(model_uri: str, df: pd.DataFrame, holdout: int) -> float:
    cutoff = df["ds"].max() - pd.DateOffset(months=holdout)
    test   = df[df["ds"] > cutoff][["ds", "y"]]
    model  = mlflow.prophet.load_model(model_uri)
    # Extend forecast to cover the full holdout window
    future = model.make_future_dataframe(periods=holdout + 2, freq="MS")
    forecast = model.predict(future)
    pred = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat"]].set_index("ds")
    actual = test.set_index("ds")["y"]
    aligned = actual.align(pred["yhat"], join="inner")
    if len(aligned[0]) == 0:
        raise ValueError("No overlapping dates between forecast and holdout")
    return mean_absolute_error(aligned[0], aligned[1])


def main():
    client = mlflow.tracking.MlflowClient()
    df = pd.read_csv(DATA_PATH, parse_dates=["ds"]).sort_values("ds")

    staging_v = get_model_version("Staging")
    prod_v    = get_model_version("Production")

    if not staging_v:
        print("No Staging model found — run train.py first")
        sys.exit(1)

    staging_uri = f"models:/{REGISTERED_MODEL}/Staging"
    staging_mae = score_model(staging_uri, df, HOLDOUT_MONTHS)
    print(f"📊 Staging  MAE ({HOLDOUT_MONTHS}m holdout): {staging_mae:.4f} Mt")

    if prod_v:
        try:
            prod_uri = f"models:/{REGISTERED_MODEL}/Production"
            prod_mae = score_model(prod_uri, df, HOLDOUT_MONTHS)
            print(f"📊 Production MAE ({HOLDOUT_MONTHS}m holdout): {prod_mae:.4f} Mt")
            improvement = (prod_mae - staging_mae) / prod_mae
            promote = staging_mae < prod_mae * PROMOTION_THRESHOLD
            print(f"   Improvement: {improvement*100:+.1f}% (threshold: +5%)")
        except ValueError:
            prod_mae = None
            promote = True
            print("   Production model incompatible with current data — promoting automatically")
    else:
        prod_mae = None
        promote = True
        print("   No production model yet — promoting automatically")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="promotion-check"):
        mlflow.log_params({
            "staging_version": staging_v.version,
            "prod_version": prod_v.version if prod_v else "none",
            "holdout_months": HOLDOUT_MONTHS,
            "threshold": PROMOTION_THRESHOLD,
        })
        mlflow.log_metrics({
            "staging_mae": staging_mae,
            "prod_mae": prod_mae if prod_mae else staging_mae,
            "promoted": int(promote),
        })

    if promote:
        # Archive old production
        if prod_v:
            client.transition_model_version_stage(
                REGISTERED_MODEL, prod_v.version, "Archived"
            )
        # Promote staging → production
        client.transition_model_version_stage(
            REGISTERED_MODEL, staging_v.version, "Production"
        )
        print(f"✅ Promoted v{staging_v.version} → Production")
    else:
        print(f"⏭  Kept existing Production (staging did not beat threshold)")

    return promote


if __name__ == "__main__":
    main()
