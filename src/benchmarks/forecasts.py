#!/usr/bin/env python3
"""
forecasts.py — Institutional forecast benchmarking scoreboard.

Grades published energy / CO2 forecasts from major institutions against
posterior-mean actuals from the fusion_posterior model. Writes scores to
benchmark_scores in DuckDB and logs the leaderboard to MLflow.

Variables compared
  energy_twh     → total facility energy (IT + cooling): sum(dc_gwh)/1000 annually
  energy_twh_it  → IT load only: sum(dc_gwh)/1000/PUE_POSTERIOR_MEAN annually
  co2_mt         → CO2 emissions: sum(dc_co2_mt_monthly) annually

Grading scale  |error%|  →  A<5%, B<15%, C<30%, D<50%, F≥50%
Bias                     →  "under" = forecast too low, "over" = forecast too high

Usage:
  python src/benchmarks/forecasts.py
"""
import sys, mlflow, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MLFLOW_EXP = "ai-energy-forecast-tracker"
mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")

# Posterior mean PUE from fusion model — used to infer IT-load-only estimates
PUE_POSTERIOR_MEAN = 1.34

# ── Published institutional forecasts ─────────────────────────────────────────
# lo / hi are published range bounds; mid is the primary point estimate.
# forecast_year is the year the forecast was made FOR (not when published).
# variable: energy_twh (total facility), energy_twh_it (IT load), co2_mt.
PUBLISHED_FORECASTS = [
    # ── LBNL ─────────────────────────────────────────────────────────────────
    {
        "source": "LBNL",
        "report": "Shehabi et al. 2016",
        "report_year": 2016,
        "forecast_year": 2020,
        "variable": "energy_twh",
        "lo": 45.0, "mid": 73.0, "hi": 145.0,
        "notes": (
            "3 scenarios (low/base/high); base assumes continued server efficiency "
            "improvements via hyperscale migration. Widely cited DOE report."
        ),
        "url": "https://eta.lbl.gov/publications/united-states-data-center-energy",
    },
    # ── Masanet ───────────────────────────────────────────────────────────────
    {
        "source": "Masanet et al.",
        "report": "Science 2020",
        "report_year": 2020,
        "forecast_year": 2018,
        "variable": "energy_twh",
        "lo": 76.0, "mid": 90.0, "hi": 107.0,
        "notes": (
            "2018 US DC baseline estimate. Paper argued workload-adjusted efficiency "
            "would keep energy flat through 2030 — a prediction not borne out by AI boom."
        ),
        "url": "https://doi.org/10.1126/science.aba3758",
    },
    # ── Guidi ─────────────────────────────────────────────────────────────────
    {
        "source": "Guidi et al.",
        "report": "Carbon Intensity of AI in Cloud 2024",
        "report_year": 2024,
        "forecast_year": 2018,
        "variable": "co2_mt",
        "lo": None, "mid": 31.5, "hi": None,
        "notes": (
            "2018 US datacenter CO2 estimate using DC-weighted emission factor "
            "(~317 g/kWh) — lower than national avg because DCs concentrate in "
            "VA which has a nuclear-heavy grid."
        ),
        "url": "https://doi.org/10.1145/3620666.3651329",
    },
    # ── IEA ───────────────────────────────────────────────────────────────────
    {
        "source": "IEA",
        "report": "Energy and AI 2025",
        "report_year": 2025,
        "forecast_year": 2024,
        "variable": "energy_twh_it",
        "lo": 183.0, "mid": 183.0, "hi": 183.0,
        "notes": (
            "IT load only (excludes cooling overhead). Uses national-avg PUE ~1.58 "
            "to derive total facility load. Bottom-up facility inventory methodology."
        ),
        "url": "https://www.iea.org/reports/energy-and-ai",
    },
    {
        "source": "IEA",
        "report": "Energy and AI 2025",
        "report_year": 2025,
        "forecast_year": 2024,
        "variable": "co2_mt",
        "lo": 105.0, "mid": 105.0, "hi": 105.0,
        "notes": (
            "National avg emission factor × total facility energy (IT + cooling). "
            "Scope: all US datacenters including non-AI workloads."
        ),
        "url": "https://www.iea.org/reports/energy-and-ai",
    },
    {
        "source": "IEA",
        "report": "Energy and AI 2025 (Announced Pledges)",
        "report_year": 2025,
        "forecast_year": 2026,
        "variable": "energy_twh_it",
        "lo": 250.0, "mid": 325.0, "hi": 400.0,
        "notes": (
            "Scenario projection from 2024 baseline; IT load only. "
            "Range reflects announced-projects vs stated-policies scenarios."
        ),
        "url": "https://www.iea.org/reports/energy-and-ai",
    },
    # ── Goldman Sachs ─────────────────────────────────────────────────────────
    {
        "source": "Goldman Sachs",
        "report": "Power Up 2024",
        "report_year": 2024,
        "forecast_year": 2030,
        "variable": "energy_twh",
        "lo": 290.0, "mid": 310.0, "hi": 340.0,
        "notes": (
            "US datacenter total energy incl. cooling; derived from 160% growth "
            "in power demand projection. AI workloads assumed ~40% of capacity by 2030."
        ),
        "url": (
            "https://www.goldmansachs.com/intelligence/pages/"
            "ai-is-poised-to-drive-160-increase-in-power-demand.html"
        ),
    },
    # ── EPRI ──────────────────────────────────────────────────────────────────
    {
        "source": "EPRI",
        "report": "Powering Intelligence 2024",
        "report_year": 2024,
        "forecast_year": 2030,
        "variable": "energy_twh",
        "lo": 325.0, "mid": 390.0, "hi": 500.0,
        "notes": (
            "Scenario analysis: 6.7–9.1% of US electricity by 2030. "
            "Mid = ~390 TWh at 4,200 TWh total US generation. "
            "Assumes US total electricity ~4,200 TWh."
        ),
        "url": "https://www.epri.com/research/products/000000003002028905",
    },
    # ── McKinsey ──────────────────────────────────────────────────────────────
    {
        "source": "McKinsey",
        "report": "Rising Data Center Economy 2024",
        "report_year": 2024,
        "forecast_year": 2030,
        "variable": "energy_twh",
        "lo": 260.0, "mid": 340.0, "hi": 440.0,
        "notes": (
            "10–15% CAGR in installed DC capacity from ~17 GW in 2023; "
            "midpoint converted to annual TWh at 0.75 utilization factor."
        ),
        "url": (
            "https://www.mckinsey.com/industries/technology-media-and-telecommunications/"
            "our-insights/investing-in-the-rising-data-center-economy"
        ),
    },
    # ── BloombergNEF ──────────────────────────────────────────────────────────
    {
        "source": "BloombergNEF",
        "report": "AI Power Demand Outlook 2024",
        "report_year": 2024,
        "forecast_year": 2030,
        "variable": "energy_twh",
        "lo": 240.0, "mid": 290.0, "hi": 380.0,
        "notes": (
            "US datacenter demand including AI accelerator clusters; "
            "more conservative than EPRI on cooling efficiency improvements."
        ),
        "url": "https://about.bnef.com/blog/ai-power-demand-is-booming/",
    },
]


# ── Actuals from fusion_posterior ─────────────────────────────────────────────

def load_actuals() -> dict[str, dict[int, float]]:
    """
    Return {variable: {year: value}} for full-year rows in fusion_posterior.
    All three variable types are derived from dc_gwh and dc_co2_mt_monthly.
    """
    with get_conn() as conn:
        df = conn.execute("""
            SELECT year(ds) as yr,
                   sum(CASE WHEN variable='dc_gwh' THEN mean ELSE 0.0 END)          as gwh,
                   sum(CASE WHEN variable='dc_co2_mt_monthly' THEN mean ELSE 0.0 END) as co2
            FROM fusion_posterior
            WHERE variable IN ('dc_gwh', 'dc_co2_mt_monthly')
              AND year(ds) <= 2025
            GROUP BY yr
            HAVING count(DISTINCT variable) = 2
               AND count(DISTINCT ds) = 12
            ORDER BY yr
        """).df()

    actuals: dict[str, dict[int, float]] = {
        "energy_twh":    {},
        "energy_twh_it": {},
        "co2_mt":        {},
    }
    for _, row in df.iterrows():
        yr = int(row["yr"])
        total_twh = row["gwh"] / 1_000.0
        actuals["energy_twh"][yr]    = total_twh
        actuals["energy_twh_it"][yr] = total_twh / PUE_POSTERIOR_MEAN
        actuals["co2_mt"][yr]        = row["co2"]

    return actuals


# ── Scoring ───────────────────────────────────────────────────────────────────

def assign_grade(error_pct: float) -> str:
    a = abs(error_pct)
    if a <  5: return "A"
    if a < 15: return "B"
    if a < 30: return "C"
    if a < 50: return "D"
    return "F"


def assign_bias(forecast_mid: float, actual: float) -> str:
    ratio = (forecast_mid - actual) / actual
    if ratio >  0.02: return "over"
    if ratio < -0.02: return "under"
    return "accurate"


def score_forecasts(actuals: dict) -> list[dict]:
    rows = []
    ts = datetime.now(timezone.utc).isoformat()

    for fc in PUBLISHED_FORECASTS:
        yr  = fc["forecast_year"]
        var = fc["variable"]
        actual = actuals.get(var, {}).get(yr)

        if actual is not None and fc["mid"] is not None:
            error_pct = (fc["mid"] - actual) / actual * 100.0
            grade     = assign_grade(error_pct)
            bias      = assign_bias(fc["mid"], actual)
        else:
            error_pct = None
            grade     = "pending"
            bias      = None

        rows.append({
            "source":        fc["source"],
            "report":        fc["report"],
            "report_year":   fc["report_year"],
            "forecast_year": yr,
            "variable":      var,
            "forecast_lo":   fc.get("lo"),
            "forecast_mid":  fc["mid"],
            "forecast_hi":   fc.get("hi"),
            "actual_value":  actual,
            "error_pct":     error_pct,
            "bias":          bias,
            "grade":         grade,
            "notes":         fc.get("notes", ""),
            "url":           fc.get("url", ""),
            "scored_at":     ts,
        })

    return rows


# ── Persistence ───────────────────────────────────────────────────────────────

def write_scores(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_scores (
                source        VARCHAR,
                report        VARCHAR,
                report_year   INTEGER,
                forecast_year INTEGER,
                variable      VARCHAR,
                forecast_lo   DOUBLE,
                forecast_mid  DOUBLE,
                forecast_hi   DOUBLE,
                actual_value  DOUBLE,
                error_pct     DOUBLE,
                bias          VARCHAR,
                grade         VARCHAR,
                notes         VARCHAR,
                url           VARCHAR,
                scored_at     TIMESTAMP,
                PRIMARY KEY (source, report, forecast_year, variable)
            )
        """)
        conn.execute("DELETE FROM benchmark_scores")
        conn.register("incoming", df)
        conn.execute("""
            INSERT INTO benchmark_scores
            SELECT source, report, report_year, forecast_year, variable,
                   forecast_lo, forecast_mid, forecast_hi,
                   actual_value, error_pct, bias, grade, notes, url,
                   scored_at::TIMESTAMP
            FROM incoming
        """)


# ── Display ───────────────────────────────────────────────────────────────────

GRADE_STAR = {"A": "★★★★★", "B": "★★★★☆", "C": "★★★☆☆",
              "D": "★★☆☆☆", "F": "★☆☆☆☆", "pending": "  —  "}

VAR_LABEL = {
    "energy_twh":    "Energy TWh (total)",
    "energy_twh_it": "Energy TWh (IT load)",
    "co2_mt":        "CO2 Mt/yr",
}


def print_scoreboard(rows: list[dict]) -> None:
    graded  = [r for r in rows if r["grade"] != "pending"]
    pending = [r for r in rows if r["grade"] == "pending"]

    print("\n" + "═" * 80)
    print("  INSTITUTIONAL FORECAST SCOREBOARD  —  US AI Datacenter Energy & CO2")
    print("═" * 80)
    print(f"\n  Actuals: fusion_posterior posterior mean  |  PUE assumed {PUE_POSTERIOR_MEAN} for IT-load\n")

    hdr = f"  {'Source':<18} {'Report yr':>9} {'Target':>7}  {'Variable':<20}  {'Forecast':>9}  {'Actual':>7}  {'Error%':>7}  {'Grade':<10} {'Bias'}"
    print(hdr)
    print("  " + "─" * 76)

    for r in sorted(graded, key=lambda x: (x["forecast_year"], x["source"])):
        forecast_str = (f"{r['forecast_lo']:.0f}–{r['forecast_mid']:.0f}–{r['forecast_hi']:.0f}"
                        if r["forecast_lo"] is not None and r["forecast_lo"] != r["forecast_mid"]
                        else f"{r['forecast_mid']:.1f}")
        actual_str   = f"{r['actual_value']:.1f}" if r["actual_value"] is not None else "—"
        error_str    = f"{r['error_pct']:+.1f}%" if r["error_pct"] is not None else "—"
        stars        = GRADE_STAR.get(r["grade"], "—")
        bias_str     = r["bias"] or "—"
        print(f"  {r['source']:<18} {r['report_year']:>9} {r['forecast_year']:>7}  "
              f"{VAR_LABEL[r['variable']]:<20}  {forecast_str:>9}  {actual_str:>7}  "
              f"{error_str:>7}  {r['grade']:<3} {stars:<8} {bias_str}")

    if pending:
        print(f"\n  {'─'*30} PENDING (no actuals yet) {'─'*30}")
        for r in sorted(pending, key=lambda x: (x["forecast_year"], x["source"])):
            range_str = (f"{r['forecast_lo']:.0f}–{r['forecast_mid']:.0f}–{r['forecast_hi']:.0f}"
                         if r["forecast_lo"] is not None else f"{r['forecast_mid']:.1f}")
            print(f"  {r['source']:<18} {r['report_year']:>9} {r['forecast_year']:>7}  "
                  f"{VAR_LABEL[r['variable']]:<20}  {range_str}")

    print("\n  Notes")
    print("  ─────")
    print("  energy_twh     = total facility energy (IT + cooling + other overhead)")
    print("  energy_twh_it  = IT load only = total / PUE; IEA scope")
    print("  co2_mt         = sum(dc_co2_mt_monthly); national avg emission factor × total facility")
    print("  Bias 'under'   = forecast lower than our model actual (missed demand growth)")
    print("  Bias 'over'    = forecast higher than our model actual")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_experiment(MLFLOW_EXP)

    actuals = load_actuals()
    rows    = score_forecasts(actuals)
    write_scores(rows)
    print_scoreboard(rows)

    # MLflow logging
    graded = [r for r in rows if r["error_pct"] is not None]
    grade_map = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    avg_score = (sum(grade_map[r["grade"]] for r in graded) / len(graded)) if graded else 0

    mlflow_metrics = {f"error_pct_{r['source'].lower().replace(' ', '_')}_{r['variable']}_{r['forecast_year']}": r["error_pct"]
                      for r in graded}
    mlflow_metrics["avg_grade_score"] = avg_score
    mlflow_metrics["n_graded"]        = len(graded)
    mlflow_metrics["n_pending"]       = len(rows) - len(graded)

    with mlflow.start_run(run_name=f"benchmarks-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_params({
            "n_institutions": len({r["source"] for r in rows}),
            "variables": ",".join(sorted({r["variable"] for r in rows})),
            "pue_assumed": PUE_POSTERIOR_MEAN,
        })
        mlflow.log_metrics(mlflow_metrics)

    print(f"✅  {len(rows)} benchmarks scored ({len(graded)} graded, "
          f"{len(rows)-len(graded)} pending) → benchmark_scores table")

    return rows


if __name__ == "__main__":
    main()
