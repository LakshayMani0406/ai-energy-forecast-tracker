"""
app.py — US AI Datacenter Energy Reference Dashboard.

Four-tab Streamlit app reading entirely from DuckDB.

Run with:
  streamlit run src/dashboard/app.py
"""
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "dashboard"))
sys.path.insert(0, str(ROOT / "src" / "ingest"))

from data import (
    MODEL_COLORS, MODEL_LABELS, GRADE_COLORS,
    load_co2_history, load_energy_history,
    load_model_forecasts, load_holdout_comparison,
    load_leaderboard, load_2030_projections,
    load_benchmark_scores,
    load_state_2024, load_state_2030,
)

st.set_page_config(
    page_title="US AI Datacenter Energy",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ US AI Datacenter Energy Reference Model")
st.caption(
    "Bayesian fusion of EIA commercial sector data · Four forecast models · "
    "Institutional benchmark scoreboard  |  Data through Feb 2026"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 National Forecast",
    "🏆 Model Leaderboard",
    "📋 Institutional Benchmarks",
    "🗺️  State Breakdown",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  —  National Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    hist     = load_co2_history()
    energy   = load_energy_history()
    fc_all   = load_model_forecasts("dc_co2_mt_monthly")
    proj2030 = load_2030_projections()

    last_ds = hist["ds"].max()

    # ── KPI metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    # 2024 annual actual
    yr2024 = hist[hist["ds"].dt.year == 2024]["mean"].sum()
    yr2023 = hist[hist["ds"].dt.year == 2023]["mean"].sum()
    # 2025 partial
    yr2025 = hist[hist["ds"].dt.year == 2025]["mean"].sum()

    sarima_2030 = proj2030[proj2030["model"] == "sarima"]["co2_mt_2030"].values
    sarima_2030_val = float(sarima_2030[0]) if len(sarima_2030) else float("nan")

    en2024 = energy[energy["yr"] == 2024]["twh"].values
    en2024_val = float(en2024[0]) if len(en2024) else float("nan")

    c1.metric("2024 Total Facility Energy", f"{en2024_val:.0f} TWh",
              help="Sum of fusion_posterior dc_gwh (incl. cooling overhead)")
    c2.metric("2024 CO₂ Emissions", f"{yr2024:.1f} Mt",
              delta=f"{yr2024-yr2023:+.1f} vs 2023",
              help="Sum of dc_co2_mt_monthly · national avg emission factor")
    c3.metric("2025 CO₂ (partial)", f"{yr2025:.1f} Mt",
              help="Jan–Dec 2025 from fusion_posterior")
    c4.metric("SARIMA 2030 Projection", f"{sarima_2030_val:.0f} Mt/yr",
              help="Best model (lowest holdout MAE)")

    st.divider()

    # ── CO2 forecast chart ────────────────────────────────────────────────────
    st.subheader("Monthly DC CO₂ Emissions + Model Forecasts (2001–2030)")

    holdout_start = last_ds - pd.DateOffset(months=12)

    fig1 = go.Figure()

    # 95% CI band (history)
    fig1.add_trace(go.Scatter(
        x=pd.concat([hist["ds"], hist["ds"][::-1]]),
        y=pd.concat([hist["p97_5"], hist["p2_5"][::-1]]),
        fill="toself",
        fillcolor="rgba(45,106,79,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Fusion 95% CI",
        showlegend=True,
    ))

    # Holdout window shading
    fig1.add_vrect(
        x0=holdout_start, x1=last_ds,
        fillcolor="rgba(200,200,200,0.18)",
        line_width=0,
        annotation_text="Holdout",
        annotation_position="top left",
        annotation_font_size=11,
    )

    # Historical mean
    fig1.add_trace(go.Scatter(
        x=hist["ds"], y=hist["mean"],
        line=dict(color="#2d6a4f", width=2),
        name="Fusion posterior (actual)",
    ))

    # Future model forecasts
    for model, color in MODEL_COLORS.items():
        fc_m = fc_all[(fc_all["model"] == model) & (fc_all["ds"] > last_ds)]
        if fc_m.empty:
            continue
        is_winner = (model == "sarima")
        fig1.add_trace(go.Scatter(
            x=fc_m["ds"], y=fc_m["yhat"],
            line=dict(color=color,
                      width=2.5 if is_winner else 1.5,
                      dash="solid" if is_winner else "dash"),
            name=MODEL_LABELS[model],
        ))
        if is_winner:
            # CI band for SARIMA
            fig1.add_trace(go.Scatter(
                x=pd.concat([fc_m["ds"], fc_m["ds"][::-1]]),
                y=pd.concat([fc_m["yhat_upper"], fc_m["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor=f"rgba(0,180,216,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="SARIMA 95% CI",
                showlegend=True,
            ))

    # IEA 2024 CO2 benchmark point
    fig1.add_trace(go.Scatter(
        x=[pd.Timestamp("2024-07-01")],
        y=[105.0 / 12],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#e63946"),
        text=["IEA 2025<br>(105 Mt/yr)"],
        textposition="top right",
        name="IEA benchmark (2024)",
    ))

    fig1.update_layout(
        xaxis_title=None,
        yaxis_title="DC CO₂ (Mt / month)",
        hovermode="x unified",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── 2030 projections bar chart ────────────────────────────────────────────
    st.subheader("2030 Annual CO₂ Projections by Model")
    fig2 = go.Figure()
    for _, row in proj2030.iterrows():
        color = MODEL_COLORS.get(row["model"], "#888")
        fig2.add_trace(go.Bar(
            x=[row["model_label"]],
            y=[row["co2_mt_2030"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["co2_upper"] - row["co2_mt_2030"]],
                arrayminus=[row["co2_mt_2030"] - row["co2_lower"]],
            ),
            marker_color=color,
            name=row["model_label"],
            showlegend=False,
        ))

    # IEA 2024 reference line
    fig2.add_hline(y=105, line_dash="dot", line_color="#e63946",
                   annotation_text="IEA 2024 benchmark (105 Mt)",
                   annotation_position="bottom right")

    fig2.update_layout(
        yaxis_title="Annual CO₂ (Mt/yr)",
        height=340,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Annual energy history ─────────────────────────────────────────────────
    with st.expander("Annual total facility energy history (dc_gwh, all workloads)"):
        fig3 = go.Figure(go.Scatter(
            x=energy["yr"], y=energy["twh"],
            mode="lines+markers",
            line=dict(color="#2d6a4f", width=2),
            marker=dict(size=5),
            name="Total DC energy",
        ))
        fig3.update_layout(
            xaxis_title="Year",
            yaxis_title="Annual Energy (TWh)",
            height=300,
            margin=dict(t=10, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  —  Model Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    lb      = load_leaderboard()
    cmp     = load_holdout_comparison()
    proj    = load_2030_projections()

    st.subheader("12-Month Holdout Evaluation  —  dc_co2_mt_monthly")
    st.caption(
        "Holdout = last 12 months of fusion_posterior data. "
        "MAE computed against posterior-mean actuals."
    )

    # Styled leaderboard table
    def _row_style(row):
        return ["background-color: rgba(0,180,216,0.15)"] * len(row) if row["Rank"] == 1 else [""] * len(row)

    display_lb = lb[["Rank", "Model", "Holdout MAE (Mt/mo)", "N holdout"]].copy()
    st.dataframe(
        display_lb.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Holdout MAE by Model")
        fig_mae = go.Figure()
        for _, row in lb.iterrows():
            key = row["model_key"]
            fig_mae.add_trace(go.Bar(
                x=[row["Model"]],
                y=[row["Holdout MAE (Mt/mo)"]],
                marker_color=MODEL_COLORS.get(key, "#888"),
                showlegend=False,
            ))
        fig_mae.update_layout(
            yaxis_title="MAE (Mt/month)",
            height=300,
            margin=dict(t=10, b=20),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_right:
        st.subheader("Holdout: Actual vs Predicted")
        model_sel = st.selectbox(
            "Model",
            options=list(MODEL_COLORS.keys()),
            format_func=lambda m: MODEL_LABELS[m],
        )
        fc_hold = cmp[cmp["model"] == model_sel].sort_values("ds")
        fig_hold = go.Figure()
        fig_hold.add_trace(go.Scatter(
            x=fc_hold["ds"], y=fc_hold["y_actual"],
            mode="lines+markers",
            line=dict(color="#2d6a4f", width=2),
            name="Actual (fusion posterior)",
        ))
        fig_hold.add_trace(go.Scatter(
            x=fc_hold["ds"], y=fc_hold["yhat"],
            mode="lines+markers",
            line=dict(color=MODEL_COLORS[model_sel], width=2, dash="dash"),
            name="Predicted",
        ))
        fig_hold.update_layout(
            yaxis_title="CO₂ (Mt/month)",
            height=300,
            margin=dict(t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig_hold, use_container_width=True)

    st.divider()

    # 2030 projections comparison
    st.subheader("2030 CO₂ Projections vs IEA 2025 Benchmark")
    proj_disp = proj[["model_label", "co2_mt_2030", "co2_lower", "co2_upper"]].copy()
    proj_disp.columns = ["Model", "2030 CO₂ (Mt/yr)", "Lower", "Upper"]
    proj_disp["vs IEA 105 Mt (%)"] = ((proj_disp["2030 CO₂ (Mt/yr)"] - 105) / 105 * 100).round(1)
    st.dataframe(proj_disp.round(1), use_container_width=True, hide_index=True)

    with st.expander("ℹ️  Why does OLS underperform?"):
        st.markdown("""
**OLS works on annual aggregates**, then distributes CO₂ flat across months (annual/12).
This flattens intra-year seasonality, inflating monthly MAE even when the annual total is reasonable.

**SARIMA wins** because the fusion posterior `dc_co2_mt_monthly` has strong and consistent
12-month seasonality that SARIMA captures directly, while Prophet adds flexibility but also noise.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  —  Institutional Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    scores = load_benchmark_scores()

    st.subheader("Institutional Forecast Scoreboard")
    st.caption(
        "Published energy / CO₂ forecasts graded against fusion_posterior posterior-mean actuals.  "
        "Error% = (forecast_mid − actual) / actual × 100.  "
        "Grade: A < 5%, B < 15%, C < 30%, D < 50%, F ≥ 50%."
    )

    VAR_LABEL = {
        "energy_twh":    "Energy TWh (total facility)",
        "energy_twh_it": "Energy TWh (IT load only)",
        "co2_mt":        "CO₂ Mt/yr",
    }

    graded  = scores[scores["grade"] != "pending"].copy()
    pending = scores[scores["grade"] == "pending"].copy()

    # Graded table
    if not graded.empty:
        graded["Variable"] = graded["variable"].map(VAR_LABEL).fillna(graded["variable"])
        graded["Forecast"] = graded.apply(
            lambda r: f"{r['forecast_lo']:.0f}–{r['forecast_mid']:.0f}–{r['forecast_hi']:.0f}"
            if pd.notna(r["forecast_lo"]) and r["forecast_lo"] != r["forecast_mid"]
            else f"{r['forecast_mid']:.1f}",
            axis=1,
        )
        graded["Actual"] = graded["actual_value"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        graded["Error%"] = graded["error_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        graded["Bias"] = graded["bias"].fillna("—")

        def _grade_style(row):
            bg = GRADE_COLORS.get(row["grade"], "#ffffff")
            return [f"background-color: {bg}22" if col == "grade" else "" for col in row.index]

        disp = graded[["source", "report_year", "forecast_year", "Variable",
                        "Forecast", "Actual", "Error%", "grade", "Bias"]].copy()
        disp.columns = ["Source", "Report yr", "Target yr", "Variable",
                        "Forecast", "Actual", "Error%", "Grade", "Bias"]
        st.dataframe(
            disp.style.apply(_grade_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # Error bar chart
    st.subheader("Forecast Error by Institution")
    fig_err = go.Figure()
    for _, row in graded.iterrows():
        if pd.isna(row["error_pct"]):
            continue
        color = GRADE_COLORS.get(row["grade"], "#888")
        label = f"{row['source']} ({row['forecast_year']})"
        fig_err.add_trace(go.Bar(
            x=[label],
            y=[row["error_pct"]],
            marker_color=color,
            showlegend=False,
            hovertemplate=(
                f"<b>{row['source']}</b><br>"
                f"Target: {row['forecast_year']}<br>"
                f"Variable: {row['variable']}<br>"
                f"Error: {row['error_pct']:+.1f}%<br>"
                f"Grade: {row['grade']}<extra></extra>"
            ),
        ))
    fig_err.add_hline(y=0, line_color="#333", line_width=1)
    fig_err.update_layout(
        yaxis_title="Error% (positive = forecast too high)",
        height=320,
        margin=dict(t=10, b=20),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # Pending table
    if not pending.empty:
        st.subheader("Pending — No Actuals Yet (2026–2030)")
        pending["Variable"] = pending["variable"].map(VAR_LABEL).fillna(pending["variable"])
        pending["Range (TWh or Mt)"] = pending.apply(
            lambda r: f"{r['forecast_lo']:.0f}–{r['forecast_mid']:.0f}–{r['forecast_hi']:.0f}"
            if pd.notna(r["forecast_lo"]) else f"{r['forecast_mid']:.1f}",
            axis=1,
        )
        pend_disp = pending[["source", "report_year", "forecast_year",
                              "Variable", "Range (TWh or Mt)", "notes"]].copy()
        pend_disp.columns = ["Source", "Report yr", "Target yr", "Variable", "Range", "Notes"]
        st.dataframe(pend_disp, use_container_width=True, hide_index=True)

    with st.expander("ℹ️  Methodology notes"):
        st.markdown("""
**energy_twh (total facility)** — compared against `sum(dc_gwh)/1000` from fusion_posterior.
Includes IT equipment + cooling + power distribution.

**energy_twh_it (IT load only)** — compared against `sum(dc_gwh)/1000 / PUE` where PUE = 1.34
(fusion model posterior mean). IEA's 183 TWh (2024) is IT-load scope.

**co2_mt** — compared against `sum(dc_co2_mt_monthly)` from fusion_posterior.
Uses national-avg emission factor (366–390 g/kWh from eGRID 2020–2023).

**Key finding**: Pre-2023 forecasts (LBNL, Masanet) severely underestimated demand growth
because they assumed workload-efficiency improvements would offset compute expansion.
The AI boom of 2023+ invalidated this assumption.

**IEA 2025 tension**: Our model's CO₂ estimate (68 Mt) is 37 Mt below IEA's 105 Mt.
Root cause: IEA uses national-avg emission factors × total facility scope (PUE ~1.58);
our model's PUE converges to 1.34 under the joint energy + CO₂ constraints — a
documented limitation explained in the fusion model methodology.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  —  State Breakdown
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    states24 = load_state_2024()
    states30 = load_state_2030()

    st.subheader("State-Level DC Energy  —  2024 Actuals vs 2030 Forecasts")
    st.caption(
        "2024: fusion_posterior posterior mean (state_dc_gwh_XX × Dirichlet weights).  "
        "2030: Prophet model forecast."
    )

    # Merge 2024 + 2030
    merged = states24.merge(states30, on="state", how="outer").sort_values(
        "twh_2024", ascending=False
    )
    merged["growth_pct"] = ((merged["twh_2030"] - merged["twh_2024"]) / merged["twh_2024"] * 100).round(1)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("2024 vs 2030 Energy by State (TWh)")
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(
            name="2024 actual",
            x=merged["state"],
            y=merged["twh_2024"],
            marker_color="#2d6a4f",
        ))
        fig_st.add_trace(go.Bar(
            name="2030 forecast (Prophet)",
            x=merged["state"],
            y=merged["twh_2030"],
            marker_color=MODEL_COLORS["prophet"],
        ))
        fig_st.update_layout(
            barmode="group",
            yaxis_title="Annual Energy (TWh)",
            height=360,
            margin=dict(t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig_st, use_container_width=True)

    with col_b:
        st.subheader("2024 State Share")
        total24 = states24["twh_2024"].sum()
        fig_pie = go.Figure(go.Pie(
            labels=states24["state"],
            values=states24["twh_2024"].round(2),
            hole=0.4,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig_pie.update_layout(
            height=360,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # State table
    st.subheader("State Summary Table")
    disp_st = merged.copy()
    disp_st["twh_2024"] = disp_st["twh_2024"].round(1)
    disp_st["twh_2030"] = disp_st["twh_2030"].round(1)
    disp_st.columns = ["State", "2024 Energy (TWh)", "2030 Forecast (TWh)", "Growth %"]
    st.dataframe(disp_st, use_container_width=True, hide_index=True)

    with st.expander("ℹ️  State methodology"):
        st.markdown("""
**State weights** are fixed Dirichlet priors derived from CBRE / JLL datacenter
market reports (2024):  VA 35%, TX 8.5%, CA 7%, GA 5.5%, OH 4%, IL 3.8%, etc.

**2024 state energy** = national `dc_gwh` × state weight (applied monthly).

**2030 state forecasts** use Prophet trained independently on each state's
`state_dc_gwh_XX` time series from fusion_posterior.

The top 13 states account for ~76% of US datacenter capacity by this weighting.
        """)
