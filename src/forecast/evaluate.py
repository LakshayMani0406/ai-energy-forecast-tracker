#!/usr/bin/env python3
"""
evaluate.py — Multi-model holdout evaluation and promotion.

Reads is_holdout rows from model_forecasts, joins against fusion_posterior
actuals, computes MAE per model, logs leaderboard to MLflow, and promotes
the best Prophet version to Production in the model registry.

All four models must have been trained before running this:
  python src/forecast/naive_seasonal.py
  python src/forecast/sarima_model.py
  python src/forecast/ols_model.py
  python src/forecast/prophet_model.py
  python src/forecast/evaluate.py

Usage:
  python src/forecast/evaluate.py
"""
import sys, mlflow, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP        = "ai-energy-forecast-tracker"
REGISTERED_MODEL  = "ai-energy-forecast-model"
VARIABLE          = "dc_co2_mt_monthly"
PROMOTION_THRESHOLD = 0.95   # new MAE must be < 95% of prod MAE to promote

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")


def load_holdout_comparison() -> pd.DataFrame:
    """Join model_forecasts holdout rows with fusion_posterior actuals."""
    with get_conn() as conn:
        df = conn.execute("""
            SELECT mf.model,
                   mf.ds,
                   mf.yhat,
                   fp.mean AS y_actual
            FROM model_forecasts mf
            JOIN fusion_posterior fp
              ON mf.ds = fp.ds
             AND fp.variable = mf.variable
            WHERE mf.variable = ?
              AND mf.is_holdout = TRUE
            ORDER BY mf.model, mf.ds
        """, [VARIABLE]).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, grp in df.groupby("model"):
        mae = mean_absolute_error(grp["y_actual"], grp["yhat"])
        rows.append({"model": model, "holdout_mae_mt": mae, "n_holdout": len(grp)})
    lb = pd.DataFrame(rows).sort_values("holdout_mae_mt").reset_index(drop=True)
    lb["rank"] = lb.index + 1
    return lb


def get_model_version(stage: str):
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL, stages=[stage])
        return versions[0] if versions else None
    except Exception:
        return None


def promote_winner(winner_model: str, winner_mae: float) -> None:
    """
    Promote the latest registered version of the winning model to Production.
    Searches all versions, finds the most recent one whose run logged winner_model
    as a param, and transitions it. Falls back to latest version if param search fails.
    """
    client = mlflow.tracking.MlflowClient()
    prod_v = get_model_version("Production")

    # Fetch all registered versions ordered by creation time desc
    try:
        all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    except Exception as e:
        print(f"   Registry lookup failed: {e}")
        return

    if not all_versions:
        print("   No registered model versions found — run the model scripts first")
        return

    # Try to find a version matching the winner model by params
    winner_version = None
    for v in sorted(all_versions, key=lambda v: int(v.version), reverse=True):
        try:
            run = client.get_run(v.run_id)
            run_model = run.data.params.get("model", "")
            if run_model == winner_model:
                winner_version = v
                break
        except Exception:
            continue

    # Fall back to the most-recent version
    if winner_version is None:
        winner_version = sorted(all_versions, key=lambda v: int(v.version))[-1]
        print(f"   Could not match winner by params — using latest v{winner_version.version}")

    if prod_v and prod_v.version == winner_version.version:
        print(f"   v{prod_v.version} already in Production — no change")
        return

    if prod_v:
        client.transition_model_version_stage(
            REGISTERED_MODEL, prod_v.version, "Archived"
        )
    client.transition_model_version_stage(
        REGISTERED_MODEL, winner_version.version, "Production"
    )
    print(f"   Promoted {winner_model} v{winner_version.version} → Production"
          f" (MAE {winner_mae:.4f} Mt/mo)")


def main():
    mlflow.set_experiment(MLFLOW_EXP)

    df = load_holdout_comparison()
    if df.empty:
        print("❌  No holdout rows found — run the four model scripts first")
        sys.exit(1)

    lb = compute_leaderboard(df)
    winner = lb.iloc[0]

    print("\n📊 Model leaderboard (12-month holdout MAE on dc_co2_mt_monthly):")
    print(f"{'Rank':<6}{'Model':<20}{'MAE (Mt/mo)':<16}{'N holdout'}")
    print("─" * 52)
    for _, row in lb.iterrows():
        marker = " ← winner" if row["rank"] == 1 else ""
        print(f"{int(row['rank']):<6}{row['model']:<20}{row['holdout_mae_mt']:<16.4f}"
              f"{int(row['n_holdout'])}{marker}")

    metrics = {f"mae_{row['model']}": row["holdout_mae_mt"] for _, row in lb.iterrows()}
    metrics["winner_mae"] = float(winner["holdout_mae_mt"])

    with mlflow.start_run(run_name=f"evaluate-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_params({
            "variable": VARIABLE,
            "winner_model": winner["model"],
            "models_compared": ",".join(lb["model"].tolist()),
        })
        mlflow.log_metrics(metrics)

    print(f"\n🏆 Winner: {winner['model']} — MAE {winner['holdout_mae_mt']:.4f} Mt/mo")

    promote_winner(winner["model"], float(winner["holdout_mae_mt"]))

    return winner["model"]


if __name__ == "__main__":
    main()
