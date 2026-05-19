"""
dashboard.py — Streamlit monitoring dashboard.

Run with:  streamlit run src/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mlflow
import mlflow.prophet
from pathlib import Path

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "energy_data.csv"
REGISTERED_MODEL = "ai-energy-forecast-model"

mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")

st.set_page_config(page_title="Datacenter CO₂ Forecaster", layout="wide",
                   page_icon="🏭")
st.title("🏭 Datacenter CO₂ Forecaster")
st.caption("Self-maintaining MLOps pipeline · Prophet · MLflow · EIA data")


# ── helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_actuals():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["ds", "y"])
    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    return df.sort_values("ds")


@st.cache_data(ttl=300)
def load_mlflow_runs():
    client = mlflow.tracking.MlflowClient()
    try:
        exp = client.get_experiment_by_name("ai-energy-forecast-tracker")
        if not exp:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.mlflow.runName != 'promotion-check'",
            order_by=["start_time DESC"],
            max_results=20,
        )
        records = []
        for r in runs:
            records.append({
                "run_id": r.info.run_id[:8],
                "date": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                "MAE": round(r.data.metrics.get("mae", np.nan), 4),
                "RMSE": round(r.data.metrics.get("rmse", np.nan), 4),
                "MAPE %": round(r.data.metrics.get("mape", np.nan), 2),
                "changepoint": r.data.params.get("changepoint_prior_scale", "—"),
                "seasonality": r.data.params.get("seasonality_mode", "—"),
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_production_model_info():
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])
        if not versions:
            return None
        v = versions[0]
        return {
            "version": v.version,
            "created": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d"),
            "run_id": v.run_id[:8],
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_forecast(horizon: int = 24):
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])
        if not versions:
            return None
        model = mlflow.prophet.load_model(f"models:/{REGISTERED_MODEL}/Production")
        df = load_actuals()
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        st.warning(f"Could not load production model: {e}")
        return None


# ── layout ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
df_actuals = load_actuals()
prod_info  = load_production_model_info()

with col1:
    if not df_actuals.empty:
        latest_co2 = df_actuals["y"].iloc[-1]
        st.metric("Latest CO₂ (Mt/month)", f"{latest_co2:.2f}",
                  delta=f"{latest_co2 - df_actuals['y'].iloc[-2]:.2f} vs prev month")
    else:
        st.metric("Latest CO₂", "—")

with col2:
    runs_df = load_mlflow_runs()
    if not runs_df.empty:
        best_mae = runs_df["MAE"].min()
        st.metric("Best Model MAE", f"{best_mae:.4f} Mt", delta=f"{len(runs_df)} runs logged")
    else:
        st.metric("MLflow Runs", "No runs yet")

with col3:
    if prod_info:
        st.metric("Production Model", f"v{prod_info['version']}",
                  delta=f"Trained {prod_info['created']}")
    else:
        st.metric("Production Model", "None yet")

st.divider()

# ── forecast chart ───────────────────────────────────────────────────────────
st.subheader("📈 Forecast")
horizon = st.slider("Forecast horizon (months)", 6, 36, 24)

forecast = get_forecast(horizon)
fig = go.Figure()

if not df_actuals.empty:
    fig.add_trace(go.Scatter(
        x=df_actuals["ds"], y=df_actuals["y"],
        mode="lines+markers", name="Actuals",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
    ))

if forecast is not None:
    cutoff = df_actuals["ds"].max() if not df_actuals.empty else pd.Timestamp.now()
    future_fc = forecast[forecast["ds"] > cutoff]
    hist_fc   = forecast[forecast["ds"] <= cutoff]

    fig.add_trace(go.Scatter(
        x=hist_fc["ds"], y=hist_fc["yhat"],
        mode="lines", name="In-sample fit",
        line=dict(color="#aec7e8", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future_fc["ds"], y=future_fc["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="#ff7f0e", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    ))

fig.update_layout(
    xaxis_title="Date", yaxis_title="CO₂ Emissions (Megatons/month)",
    hovermode="x unified", height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ── mlflow experiment history ─────────────────────────────────────────────────
st.subheader("🧪 Experiment History")
if not runs_df.empty:
    # MAE over time sparkline
    mae_fig = go.Figure(go.Scatter(
        x=runs_df["date"][::-1], y=runs_df["MAE"][::-1],
        mode="lines+markers", line=dict(color="#2ca02c"),
        name="MAE per run",
    ))
    mae_fig.update_layout(height=200, margin=dict(t=10, b=10),
                          yaxis_title="MAE (Mt)", xaxis_title="")
    st.plotly_chart(mae_fig, use_container_width=True)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
else:
    st.info("No MLflow runs yet. Run `python src/train.py` to start.")

# ── model registry ────────────────────────────────────────────────────────────
st.subheader("🏷️ Model Registry")
if prod_info:
    st.json(prod_info)
else:
    st.info("No production model. After training, run `python src/evaluate.py` to promote.")

# ── pipeline status ───────────────────────────────────────────────────────────
with st.expander("⚙️ Pipeline Commands"):
    st.code("""
# 1. Pull latest data
python src/fetch_data.py

# 2. Train + log to MLflow
python src/train.py

# 3. Promote if better than production
python src/evaluate.py

# 4. View MLflow UI
mlflow ui --backend-store-uri mlruns/

# 5. This dashboard
streamlit run src/dashboard.py
    """, language="bash")
