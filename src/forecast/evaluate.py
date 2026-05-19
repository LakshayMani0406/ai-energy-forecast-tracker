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


def promote_prophet(lb: pd.DataFrame) -> None:
    """Promote the latest Prophet registry version if it won or no Production exists."""
    client = mlflow.tracking.MlflowClient()
    staging_v = get_model_version("Staging")
    prod_v    = get_model_version("Production")

    if staging_v:
        promote = prod_v is None
        if not promote:
            prophet_row = lb[lb["model"] == "prophet"]
            if not prophet_row.empty:
                promote = float(prophet_row["holdout_mae_mt"].iloc[0]) < PROMOTION_THRESHOLD
        if promote:
            if prod_v:
                client.transition_model_version_stage(
                    REGISTERED_MODEL, prod_v.version, "Archived"
                )
            client.transition_model_version_stage(
                REGISTERED_MODEL, staging_v.version, "Production"
            )
            print(f"   Promoted v{staging_v.version} → Production")
        else:
            print(f"   Staging v{staging_v.version} did not beat threshold — kept Production")
    elif prod_v:
        print(f"   Prophet already in Production (v{prod_v.version}) — no action needed")
    else:
        # Promote whatever version exists (first run)
        try:
            all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
            if all_versions:
                latest = sorted(all_versions, key=lambda v: int(v.version))[-1]
                client.transition_model_version_stage(
                    REGISTERED_MODEL, latest.version, "Production"
                )
                print(f"   Promoted v{latest.version} → Production")
        except Exception as e:
            print(f"   Registry promotion skipped: {e}")


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

    promote_prophet(lb)

    return winner["model"]


if __name__ == "__main__":
    main()
