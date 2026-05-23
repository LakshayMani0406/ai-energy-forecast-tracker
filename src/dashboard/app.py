"""
AI Infrastructure Digital Twin — Dashboard
Six tabs:
  1. National Forecast     — historical + SARIMA/Prophet/OLS/naive to 2030
  2. Model Leaderboard     — holdout evaluation
  3. Institutional Benchmarks
  4. State Breakdown
  5. Futures Engine        — Monte Carlo scenario fan charts
  6. Forecast Graveyard    — decay analysis + org credibility
"""
from __future__ import annotations

import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "dashboard"))
sys.path.insert(0, str(ROOT / "src" / "ingest"))
sys.path.insert(0, str(ROOT / "src"))

from data import (
    MODEL_COLORS, MODEL_LABELS, GRADE_COLORS,
    load_co2_history, load_energy_history,
    load_model_forecasts, load_holdout_comparison,
    load_leaderboard, load_2030_projections,
    load_benchmark_scores,
    load_state_2024, load_state_2030,
    load_simulation_summary,
    load_run_manifest, load_scenario_report, load_sensitivity,
    load_co2_annual_history, load_energy_annual_history,
    load_forecast_memory, load_org_credibility,
    load_decay_curve, load_assumption_autopsy,
    load_gpu_demand_agent, load_emissions_agent,
)

from simulation_engine.scenarios import SCENARIOS, SCENARIO_MAP


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


st.set_page_config(
    page_title="AI Infrastructure Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — Bloomberg dark ───────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  h1 { letter-spacing: -0.5px; font-size: 1.7rem !important; }
  h2 { font-size: 1.1rem !important; color: #94a3b8; letter-spacing: 0.5px; text-transform: uppercase; }
  h3 { font-size: 1.0rem !important; }
  .stMetric label { font-size: 0.68rem !important; color: #64748b !important; letter-spacing: 1px; text-transform: uppercase; }
  .stMetric [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #e6edf3 !important; }
  .stTabs [data-baseweb="tab-list"] { gap: 2px; }
  .stTabs [data-baseweb="tab"] { font-size: 0.78rem; padding: 6px 14px; }
  div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI Infrastructure Digital Twin")
st.caption(
    "Probabilistic forecasting · Monte Carlo simulation · Forecast audit system  "
    "|  Bayesian fusion of EIA + EPA data · 8 scenarios · 10k trajectories"
)

tabs = st.tabs([
    "📈 National Forecast",
    "🏆 Model Leaderboard",
    "📋 Institutional Benchmarks",
    "🗺️  State Breakdown",
    "🔮 Futures Engine",
    "🪦 Forecast Graveyard",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  —  National Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    hist     = load_co2_history()
    energy   = load_energy_history()
    fc_all   = load_model_forecasts("dc_co2_mt_monthly")
    proj2030 = load_2030_projections()

    last_ds = hist["ds"].max()
    yr2024  = hist[hist["ds"].dt.year == 2024]["mean"].sum()
    yr2023  = hist[hist["ds"].dt.year == 2023]["mean"].sum()
    yr2025  = hist[hist["ds"].dt.year == 2025]["mean"].sum()

    sarima_2030 = proj2030[proj2030["model"] == "sarima"]["co2_mt_2030"].values
    sarima_val  = float(sarima_2030[0]) if len(sarima_2030) else float("nan")
    en2024      = energy[energy["yr"] == 2024]["twh"].values
    en2024_val  = float(en2024[0]) if len(en2024) else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2024 Total Facility Energy", f"{en2024_val:.0f} TWh")
    c2.metric("2024 CO₂", f"{yr2024:.1f} Mt", delta=f"{yr2024-yr2023:+.1f} vs 2023")
    c3.metric("2025 CO₂ (partial)", f"{yr2025:.1f} Mt")
    c4.metric("SARIMA 2030 Projection", f"{sarima_val:.0f} Mt/yr")

    st.divider()
    st.subheader("Monthly DC CO₂ Emissions + Model Forecasts (2001–2030)")

    holdout_start = last_ds - pd.DateOffset(months=12)
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=pd.concat([hist["ds"], hist["ds"][::-1]]),
        y=pd.concat([hist["p97_5"], hist["p2_5"][::-1]]),
        fill="toself", fillcolor="rgba(34,197,94,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Fusion 95% CI",
    ))
    fig1.add_vrect(
        x0=holdout_start, x1=last_ds,
        fillcolor="rgba(255,255,255,0.04)", line_width=0,
        annotation_text="Holdout", annotation_position="top left",
        annotation_font_size=11,
    )
    fig1.add_trace(go.Scatter(
        x=hist["ds"], y=hist["mean"],
        line=dict(color="#22c55e", width=2),
        name="Fusion posterior (actual)",
    ))

    for model, color in MODEL_COLORS.items():
        fc_m = fc_all[(fc_all["model"] == model) & (fc_all["ds"] > last_ds)]
        if fc_m.empty:
            continue
        is_winner = (model == "sarima")
        fig1.add_trace(go.Scatter(
            x=fc_m["ds"], y=fc_m["yhat"],
            line=dict(color=color, width=2.5 if is_winner else 1.5,
                      dash="solid" if is_winner else "dash"),
            name=MODEL_LABELS[model],
        ))
        if is_winner:
            fig1.add_trace(go.Scatter(
                x=pd.concat([fc_m["ds"], fc_m["ds"][::-1]]),
                y=pd.concat([fc_m["yhat_upper"], fc_m["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(0,180,216,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="SARIMA 95% CI",
            ))

    fig1.add_trace(go.Scatter(
        x=[pd.Timestamp("2024-07-01")], y=[105.0 / 12],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#ef4444"),
        text=["IEA 2025<br>(105 Mt/yr)"], textposition="top right",
        name="IEA benchmark (2024)",
    ))

    fig1.update_layout(
        xaxis_title=None, yaxis_title="DC CO₂ (Mt / month)",
        hovermode="x unified", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2030 Annual CO₂ Projections by Model")
    fig2 = go.Figure()
    for _, row in proj2030.iterrows():
        color = MODEL_COLORS.get(row["model"], "#888")
        fig2.add_trace(go.Bar(
            x=[row["model_label"]], y=[row["co2_mt_2030"]],
            error_y=dict(type="data", symmetric=False,
                         array=[row["co2_upper"] - row["co2_mt_2030"]],
                         arrayminus=[row["co2_mt_2030"] - row["co2_lower"]]),
            marker_color=color, showlegend=False,
        ))
    fig2.add_hline(y=105, line_dash="dot", line_color="#ef4444",
                   annotation_text="IEA 2024 (105 Mt)", annotation_position="bottom right")
    fig2.update_layout(
        yaxis_title="Annual CO₂ (Mt/yr)", height=320, margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Annual total facility energy history"):
        fig3 = go.Figure(go.Scatter(
            x=energy["yr"], y=energy["twh"], mode="lines+markers",
            line=dict(color="#22c55e", width=2), marker=dict(size=5),
        ))
        fig3.update_layout(
            xaxis_title="Year", yaxis_title="Annual Energy (TWh)", height=300,
            margin=dict(t=10, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  —  Model Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    lb  = load_leaderboard()
    cmp = load_holdout_comparison()
    proj = load_2030_projections()

    st.subheader("12-Month Holdout Evaluation  —  dc_co2_mt_monthly")

    def _row_style(row):
        return ["background-color: rgba(0,180,216,0.15)"] * len(row) if row["Rank"] == 1 else [""] * len(row)

    st.dataframe(
        lb[["Rank", "Model", "Holdout MAE (Mt/mo)", "N holdout"]].style.apply(_row_style, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Holdout MAE by Model")
        fig_mae = go.Figure()
        for _, row in lb.iterrows():
            fig_mae.add_trace(go.Bar(
                x=[row["Model"]], y=[row["Holdout MAE (Mt/mo)"]],
                marker_color=MODEL_COLORS.get(row["model_key"], "#888"), showlegend=False,
            ))
        fig_mae.update_layout(yaxis_title="MAE (Mt/month)", height=300,
                               margin=dict(t=10, b=20),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_right:
        st.subheader("Holdout: Actual vs Predicted")
        model_sel = st.selectbox("Model", options=list(MODEL_COLORS.keys()),
                                 format_func=lambda m: MODEL_LABELS[m])
        fc_hold = cmp[cmp["model"] == model_sel].sort_values("ds")
        fig_hold = go.Figure()
        fig_hold.add_trace(go.Scatter(
            x=fc_hold["ds"], y=fc_hold["y_actual"],
            mode="lines+markers", line=dict(color="#22c55e", width=2),
            name="Actual",
        ))
        fig_hold.add_trace(go.Scatter(
            x=fc_hold["ds"], y=fc_hold["yhat"],
            mode="lines+markers",
            line=dict(color=MODEL_COLORS[model_sel], width=2, dash="dash"),
            name="Predicted",
        ))
        fig_hold.update_layout(
            yaxis_title="CO₂ (Mt/month)", height=300, margin=dict(t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hold, use_container_width=True)

    st.divider()
    st.subheader("2030 CO₂ Projections vs IEA 2025 Benchmark")
    proj_disp = proj[["model_label", "co2_mt_2030", "co2_lower", "co2_upper"]].copy()
    proj_disp.columns = ["Model", "2030 CO₂ (Mt/yr)", "Lower", "Upper"]
    proj_disp["vs IEA 105 Mt (%)"] = ((proj_disp["2030 CO₂ (Mt/yr)"] - 105) / 105 * 100).round(1)
    st.dataframe(proj_disp.round(1), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  —  Institutional Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    scores = load_benchmark_scores()
    st.subheader("Institutional Forecast Scoreboard")
    st.caption(
        "Published energy / CO₂ forecasts graded against fusion_posterior actuals.  "
        "Error% = (forecast_mid − actual) / actual × 100.  "
        "Grade: A < 5%, B < 15%, C < 30%, D < 50%, F ≥ 50%."
    )

    VAR_LABEL = {
        "energy_twh": "Energy TWh (total facility)",
        "energy_twh_it": "Energy TWh (IT load only)",
        "co2_mt": "CO₂ Mt/yr",
    }

    graded  = scores[scores["grade"] != "pending"].copy()
    pending = scores[scores["grade"] == "pending"].copy()

    if not graded.empty:
        graded["Variable"] = graded["variable"].map(VAR_LABEL).fillna(graded["variable"])
        graded["Forecast"] = graded.apply(
            lambda r: f"{r['forecast_lo']:.0f}–{r['forecast_mid']:.0f}–{r['forecast_hi']:.0f}"
            if pd.notna(r["forecast_lo"]) and r["forecast_lo"] != r["forecast_mid"]
            else f"{r['forecast_mid']:.1f}", axis=1,
        )
        graded["Actual"] = graded["actual_value"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        graded["Error%"] = graded["error_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        graded["Bias"]   = graded["bias"].fillna("—")

        def _grade_style(row):
            bg = GRADE_COLORS.get(row["Grade"], "#ffffff")
            return [f"background-color: {bg}22" if col == "Grade" else "" for col in row.index]

        disp = graded[["source", "report_year", "forecast_year", "Variable",
                        "Forecast", "Actual", "Error%", "grade", "Bias"]].copy()
        disp.columns = ["Source", "Report yr", "Target yr", "Variable",
                        "Forecast", "Actual", "Error%", "Grade", "Bias"]
        st.dataframe(disp.style.apply(_grade_style, axis=1),
                     use_container_width=True, hide_index=True)

    st.subheader("Forecast Error by Institution")
    fig_err = go.Figure()
    for _, row in graded.iterrows():
        if pd.isna(row["error_pct"]):
            continue
        color = GRADE_COLORS.get(row["grade"], "#888")
        label = f"{row['source']} ({row['forecast_year']})"
        fig_err.add_trace(go.Bar(
            x=[label], y=[row["error_pct"]], marker_color=color, showlegend=False,
            hovertemplate=(
                f"<b>{row['source']}</b><br>Target: {row['forecast_year']}<br>"
                f"Error: {row['error_pct']:+.1f}%<br>Grade: {row['grade']}<extra></extra>"
            ),
        ))
    fig_err.add_hline(y=0, line_color="#475569", line_width=1)
    fig_err.update_layout(
        yaxis_title="Error% (positive = too high)", height=320, margin=dict(t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_err, use_container_width=True)

    if not pending.empty:
        st.subheader("Pending — No Actuals Yet (2026–2030)")
        pending["Variable"] = pending["variable"].map(VAR_LABEL).fillna(pending["variable"])
        pending["Range"] = pending.apply(
            lambda r: f"{r['forecast_lo']:.0f}–{r['forecast_mid']:.0f}–{r['forecast_hi']:.0f}"
            if pd.notna(r["forecast_lo"]) else f"{r['forecast_mid']:.1f}", axis=1,
        )
        pend_disp = pending[["source", "report_year", "forecast_year",
                              "Variable", "Range", "notes"]].copy()
        pend_disp.columns = ["Source", "Report yr", "Target yr", "Variable", "Range", "Notes"]
        st.dataframe(pend_disp, use_container_width=True, hide_index=True)

    with st.expander("ℹ️  Methodology notes"):
        st.markdown("""
**IEA gap (68 Mt vs 105 Mt):** IEA uses PUE=1.58 + US national avg emission factor ~490 g/kWh.
Our model converges to PUE=1.34 under joint EIA+eGRID constraints. The gap is a documented scope difference, not an error.

**Pre-2023 forecast failure:** LBNL, Masanet assumed efficiency gains would offset demand growth.
The 2023 AI boom invalidated this — demand grew faster than any model predicted.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  —  State Breakdown
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    states24 = load_state_2024()
    states30 = load_state_2030()

    st.subheader("State-Level DC Energy  —  2024 Actuals vs 2030 Forecasts")
    merged = states24.merge(states30, on="state", how="outer").sort_values("twh_2024", ascending=False)
    merged["growth_pct"] = ((merged["twh_2030"] - merged["twh_2024"]) / merged["twh_2024"] * 100).round(1)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(name="2024 actual", x=merged["state"], y=merged["twh_2024"],
                                marker_color="#22c55e"))
        fig_st.add_trace(go.Bar(name="2030 (Prophet)", x=merged["state"], y=merged["twh_2030"],
                                marker_color=MODEL_COLORS["prophet"]))
        fig_st.update_layout(barmode="group", yaxis_title="Annual Energy (TWh)", height=360,
                              margin=dict(t=10, b=20),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig_st, use_container_width=True)

    with col_b:
        fig_pie = go.Figure(go.Pie(
            labels=states24["state"], values=states24["twh_2024"].round(2),
            hole=0.4, textinfo="label+percent", textfont_size=11,
        ))
        fig_pie.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    disp_st = merged.copy()
    disp_st["twh_2024"] = disp_st["twh_2024"].round(1)
    disp_st["twh_2030"] = disp_st["twh_2030"].round(1)
    disp_st.columns = ["State", "2024 Energy (TWh)", "2030 Forecast (TWh)", "Growth %"]
    st.dataframe(disp_st, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  —  Futures Engine
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    summary = load_simulation_summary()
    hist_co2 = load_co2_annual_history()
    hist_energy = load_energy_annual_history()

    st.subheader("AI Infrastructure Futures Engine")
    st.caption(
        "10,000 Monte Carlo trajectories per scenario · 8 alternative futures · 2025–2040"
    )

    if summary is None:
        st.warning(
            "Simulation data not found. Run: `python run_pipeline.py --all`"
        )
    else:
        # ── Scenario selector ─────────────────────────────────────────────
        selected = st.radio(
            "Scenario",
            options=[s.name for s in SCENARIOS],
            format_func=lambda n: SCENARIO_MAP[n].label,
            horizontal=True,
        )
        sc = SCENARIO_MAP[selected]

        st.markdown(
            f"<p style='color:#94a3b8; font-size:0.85rem; margin-top:-8px'>{sc.description}</p>",
            unsafe_allow_html=True,
        )

        # ── Run metadata strip ────────────────────────────────────────────
        manifest = load_run_manifest(selected)
        if manifest:
            _ts = manifest.get("timestamp", "")[:19].replace("T", " ")
            _meta_cols = st.columns(5)
            _meta_cols[0].metric("Trajectories", f"{manifest.get('n_sims', 0):,}")
            _meta_cols[1].metric("Years", f"2025–{manifest.get('years', [0])[-1]}")
            _meta_cols[2].metric("Seed", manifest.get("seed", "—"))
            _meta_cols[3].metric("Runtime", f"{manifest.get('runtime_seconds', 0):.2f}s")
            _meta_cols[4].metric("Run timestamp", _ts + " UTC")

        # ── Variable toggle ───────────────────────────────────────────────
        var_choice = st.radio("Variable", ["CO₂ (Mt/yr)", "Energy (TWh/yr)"], horizontal=True)
        sim_var  = "dc_co2_mt"   if "CO₂" in var_choice else "dc_twh"
        hist_col = "co2_mt"      if "CO₂" in var_choice else "dc_twh"
        y_label  = "Annual CO₂ (Mt/yr)" if "CO₂" in var_choice else "Annual Energy (TWh/yr)"
        hist_df  = hist_co2      if "CO₂" in var_choice else hist_energy

        sc_sum = summary[(summary["scenario"] == selected) & (summary["variable"] == sim_var)]

        # ── Fan chart ─────────────────────────────────────────────────────
        fig_fan = go.Figure()

        fig_fan.add_trace(go.Scatter(
            x=hist_df["year"], y=hist_df[hist_col],
            line=dict(color="#22c55e", width=2.5),
            name="Historical (fusion posterior)",
        ))

        fig_fan.add_trace(go.Scatter(
            x=pd.concat([sc_sum["year"], sc_sum["year"][::-1]]),
            y=pd.concat([sc_sum["p95"], sc_sum["p5"][::-1]]),
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(sc.color)},0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th–95th pct",
        ))

        fig_fan.add_trace(go.Scatter(
            x=pd.concat([sc_sum["year"], sc_sum["year"][::-1]]),
            y=pd.concat([sc_sum["p75"], sc_sum["p25"][::-1]]),
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(sc.color)},0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th–75th pct",
        ))

        fig_fan.add_trace(go.Scatter(
            x=sc_sum["year"], y=sc_sum["p50"],
            line=dict(color=sc.color, width=2.5),
            name="Median",
        ))

        if "CO₂" in var_choice:
            fig_fan.add_hline(y=105, line_dash="dot", line_color="#ef4444",
                              annotation_text="IEA 2024 (105 Mt/yr)",
                              annotation_position="bottom right")

        fig_fan.add_vline(x=2024.5, line_dash="dash", line_color="#475569",
                          annotation_text="Now", annotation_position="top right",
                          annotation_font_size=11)

        fig_fan.update_layout(
            xaxis_title=None, yaxis_title=y_label,
            hovermode="x unified", height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
            margin=dict(t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # ── Tail risk cards (CO₂ only) ────────────────────────────────────
        if "CO₂" in var_choice and not sc_sum.empty:
            _row2030 = sc_sum[sc_sum["year"] == 2030]
            if not _row2030.empty and "cvar_95" in _row2030.columns:
                r = _row2030.iloc[0]
                st.markdown("**2030 Tail Risk — from 10,000 real trajectories**")
                _rc = st.columns(4)
                _rc[0].metric("Median CO₂", f"{r['p50']:.1f} Mt/yr")
                _rc[1].metric("CVaR(95%)", f"{r['cvar_95']:.1f} Mt/yr",
                              help="Expected CO₂ across worst 5% of simulations")
                _p_iea = r.get("prob_exceed_iea", float("nan"))
                _p_2x  = r.get("prob_exceed_2x_anchor", float("nan"))
                _rc[2].metric("P(> IEA 105 Mt)", f"{_p_iea:.1%}" if _p_iea == _p_iea else "—")
                _rc[3].metric("P(> 2× 2024)", f"{_p_2x:.1%}" if _p_2x == _p_2x else "—")

        # ── Sensitivity tornado ───────────────────────────────────────────
        if "CO₂" in var_choice:
            sens_df = load_sensitivity()
            if sens_df is not None:
                col_s1, col_s2 = st.columns(2)
                for col_s, yr in zip([col_s1, col_s2], [2030, 2040]):
                    sc_sens = (
                        sens_df[(sens_df["scenario"] == selected) & (sens_df["year"] == yr)]
                        .sort_values("r_squared_pct")
                    )
                    if sc_sens.empty:
                        continue
                    bar_colors = [
                        f"rgba({_hex_to_rgb(sc.color)},{0.4 + 0.6 * (r / sc_sens['r_squared_pct'].max())})"
                        for r in sc_sens["r_squared_pct"]
                    ]
                    fig_s = go.Figure(go.Bar(
                        x=sc_sens["r_squared_pct"],
                        y=sc_sens["display_name"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"ρ={r:+.2f}  R²={r2:.0f}%"
                              for r, r2 in zip(sc_sens["spearman_r"], sc_sens["r_squared_pct"])],
                        textposition="outside",
                        cliponaxis=False,
                    ))
                    fig_s.update_layout(
                        title=f"Parameter Sensitivity — {yr}",
                        xaxis=dict(title="% CO₂ variance explained", range=[0, 110]),
                        yaxis=dict(tickfont=dict(size=11)),
                        height=240, margin=dict(t=40, b=10, l=10, r=80),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    col_s.plotly_chart(fig_s, use_container_width=True)
                st.caption(
                    "Spearman rank correlation R² — fraction of CO₂ variance explained by each driver "
                    "independently across 10,000 trajectories. "
                    "Efficiency multiplier: higher cumulative value = less improvement accumulated = more CO₂."
                )

        # ── Scenario comparison ───────────────────────────────────────────
        st.subheader("Scenario Comparison — 2030 & 2040 Median Outcomes")

        rows_cmp = []
        for sc_obj in SCENARIOS:
            s2030 = summary[
                (summary["scenario"] == sc_obj.name) &
                (summary["variable"] == sim_var) &
                (summary["year"] == 2030)
            ]
            s2040 = summary[
                (summary["scenario"] == sc_obj.name) &
                (summary["variable"] == sim_var) &
                (summary["year"] == 2040)
            ]
            if s2030.empty or s2040.empty:
                continue
            _r30 = s2030.iloc[0]
            _r40 = s2040.iloc[0]
            row = {
                "Scenario": sc_obj.label,
                "2030 median": round(float(_r30["p50"]), 1),
                "2030 range": f"{_r30['p5']:.0f} – {_r30['p95']:.0f}",
                "2040 median": round(float(_r40["p50"]), 1),
                "2040 range": f"{_r40['p5']:.0f} – {_r40['p95']:.0f}",
                "_color": sc_obj.color,
            }
            if "cvar_95" in _r30 and "CO₂" in var_choice:
                row["CVaR(95%) 2030"] = round(float(_r30["cvar_95"]), 1)
                _piea = _r30.get("prob_exceed_iea", float("nan"))
                row["P(>IEA)"] = f"{_piea:.1%}" if _piea == _piea else "—"
            rows_cmp.append(row)

        cmp_df = pd.DataFrame(rows_cmp)
        unit_label = "Mt CO₂/yr" if "CO₂" in var_choice else "TWh/yr"

        fig_cmp = go.Figure()
        for _, row in cmp_df.iterrows():
            fig_cmp.add_trace(go.Bar(
                name=row["Scenario"], x=[row["Scenario"]],
                y=[row["2030 median"]], marker_color=row["_color"],
                showlegend=False,
            ))
        fig_cmp.update_layout(
            yaxis_title=f"2030 Median ({unit_label})", height=320,
            margin=dict(t=10, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        _base_cols = ["Scenario", "2030 median", "2030 range", "2040 median", "2040 range"]
        _extra_cols = [c for c in ["CVaR(95%) 2030", "P(>IEA)"] if c in cmp_df.columns]
        display_cmp = cmp_df[_base_cols + _extra_cols].copy()
        display_cmp.rename(columns={
            "2030 median": f"2030 median ({unit_label})",
            "2040 median": f"2040 median ({unit_label})",
        }, inplace=True)
        st.dataframe(display_cmp, use_container_width=True, hide_index=True)

        # ── Scenario report expander ──────────────────────────────────────
        report_md = load_scenario_report(selected)
        if report_md:
            with st.expander(f"Full simulation report — {sc.label}", expanded=False):
                st.markdown(report_md)

        # ── Agent signals ─────────────────────────────────────────────────
        st.subheader("Agent Signals")
        col_gpu, col_emi = st.columns(2)

        with col_gpu:
            gpu_df = load_gpu_demand_agent()
            if not gpu_df.empty:
                fig_gpu = go.Figure()
                fig_gpu.add_trace(go.Scatter(
                    x=gpu_df["year"], y=gpu_df["lo"],
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                ))
                fig_gpu.add_trace(go.Scatter(
                    x=gpu_df["year"], y=gpu_df["hi"],
                    fill="tonexty", fillcolor="rgba(0,180,216,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="GPU Agent CI",
                ))
                fig_gpu.add_trace(go.Scatter(
                    x=gpu_df["year"], y=gpu_df["baseline"],
                    line=dict(color="#00b4d8", width=2), name="GPU Demand Agent",
                ))
                fig_gpu.update_layout(
                    title="GPU Demand Agent — AI Energy (TWh/yr)",
                    height=300, margin=dict(t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None, yaxis_title="TWh/yr",
                )
                st.plotly_chart(fig_gpu, use_container_width=True)

        with col_emi:
            emi_df = load_emissions_agent()
            if not emi_df.empty:
                fig_emi = go.Figure()
                fig_emi.add_trace(go.Scatter(
                    x=emi_df["year"], y=emi_df["lo"],
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                ))
                fig_emi.add_trace(go.Scatter(
                    x=emi_df["year"], y=emi_df["hi"],
                    fill="tonexty", fillcolor="rgba(34,197,94,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="Emissions Agent CI",
                ))
                fig_emi.add_trace(go.Scatter(
                    x=emi_df["year"], y=emi_df["baseline"],
                    line=dict(color="#22c55e", width=2), name="Emissions Agent",
                ))
                fig_emi.update_layout(
                    title="Emissions Agent — DC CO₂ (Mt/yr)",
                    height=300, margin=dict(t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None, yaxis_title="Mt CO₂/yr",
                )
                st.plotly_chart(fig_emi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6  —  Forecast Graveyard
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Forecast Graveyard")
    st.caption(
        "Permanent record of institutional forecasts graded against fusion_posterior actuals.  "
        "Every assumption. Every failure. Every revision."
    )

    cred_df   = load_org_credibility()
    decay_df  = load_decay_curve()
    memory_df = load_forecast_memory()
    autopsy_df = load_assumption_autopsy()

    if memory_df.empty:
        st.info("Forecast memory not seeded. Run: `python src/simulation_engine/run.py`")
    else:
        # ── Org credibility ───────────────────────────────────────────────
        st.subheader("Organization Credibility Scores")
        if not cred_df.empty:
            fig_cred = go.Figure()
            colors = ["#22c55e" if s > 0.7 else "#f59e0b" if s > 0.4 else "#ef4444"
                      for s in cred_df["confidence_score"]]
            fig_cred.add_trace(go.Bar(
                x=cred_df["source"], y=cred_df["confidence_score"],
                marker_color=colors, showlegend=False,
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ))
            fig_cred.add_hline(y=0.7, line_dash="dot", line_color="#22c55e",
                               annotation_text="Good (0.7)", annotation_position="right")
            fig_cred.add_hline(y=0.4, line_dash="dot", line_color="#f59e0b",
                               annotation_text="Poor (0.4)", annotation_position="right")
            fig_cred.update_layout(
                yaxis_title="Credibility Score (1 = perfect)", height=320,
                yaxis=dict(range=[0, 1.05]),
                margin=dict(t=10, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cred, use_container_width=True)

            cred_disp = cred_df[["source", "confidence_score", "mean_abs_error",
                                  "bias", "worst_error", "bias_direction", "n_forecasts"]].copy()
            cred_disp.columns = ["Organization", "Credibility", "Mean |Error%|",
                                  "Bias%", "Worst Error%", "Bias Direction", "N Forecasts"]
            st.dataframe(cred_disp.round(2), use_container_width=True, hide_index=True)

        st.divider()

        # ── Decay curve ───────────────────────────────────────────────────
        st.subheader("Forecast Error vs Horizon")
        st.caption("How forecast accuracy degrades as the forecast horizon (years to target) increases.")
        if not decay_df.empty:
            fig_decay = go.Figure()
            orgs = decay_df["source"].unique()
            palette = ["#00b4d8", "#f97316", "#22c55e", "#a855f7",
                       "#ef4444", "#f59e0b", "#8b5cf6", "#ec4899"]
            for i, org in enumerate(orgs):
                sub = decay_df[decay_df["source"] == org].sort_values("horizon_years")
                color = palette[i % len(palette)]
                fig_decay.add_trace(go.Scatter(
                    x=sub["horizon_years"], y=sub["abs_error"],
                    mode="markers+lines", name=org,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=8, color=color),
                    hovertemplate=(
                        f"<b>{org}</b><br>Horizon: %{{x}}yr<br>"
                        "|Error|: %{y:.1f}%<extra></extra>"
                    ),
                ))
            fig_decay.update_layout(
                xaxis_title="Forecast Horizon (years to target)",
                yaxis_title="|Error%|",
                height=360, margin=dict(t=10, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
            )
            st.plotly_chart(fig_decay, use_container_width=True)

        st.divider()

        # ── Full forecast record ──────────────────────────────────────────
        st.subheader("Full Forecast Record")
        graded_mem = memory_df[memory_df["actual_value"].notna()].copy()
        if not graded_mem.empty:
            graded_mem["error_pct"] = graded_mem["error_pct"].round(1)
            graded_mem["confidence_score"] = graded_mem["confidence_score"].round(3)
            disp_mem = graded_mem[[
                "source", "published_date", "target_year", "variable",
                "forecast_mid", "actual_value", "error_pct", "confidence_score", "notes"
            ]].copy()
            disp_mem.columns = [
                "Source", "Published", "Target yr", "Variable",
                "Forecast", "Actual", "Error%", "Confidence", "Notes"
            ]
            st.dataframe(disp_mem, use_container_width=True, hide_index=True)

        # ── Assumption autopsy ────────────────────────────────────────────
        if not autopsy_df.empty:
            st.divider()
            st.subheader("Assumption Autopsy")
            st.caption("Which assumption categories appeared most often in failed forecasts (|error| > 30%)?")
            st.dataframe(autopsy_df, use_container_width=True, hide_index=True)
