# NIGHTLOG — AI Energy Forecast Tracker

**Session started:** 2026-05-19 ~07:00 local  
**Session ended:** 2026-05-19 ~07:55 local  
**Operator:** Claude Sonnet 4.6 (autonomous mode)  
**Scope:** Steps 7–9, 11 of the master plan + tests bonus

---

## MORNING DECISIONS NEEDED

1. **IEA 2025 CO₂ gap (68 Mt vs 105 Mt):**  
   Root cause: PUE scope (1.34 vs ~1.58) and facility boundary definition.  
   Tonight's approach: treated as *documented methodological difference*, not a model bug.  
   Every mention has an honest explanation (dashboard, methodology doc, scoreboard).  
   **Decision needed:** Re-run fusion model with PUE prior anchored at 1.58 (IEA's implicit value) to close the gap? Or keep 1.34 and document the scope difference?  
   Recommendation: keep 1.34 + document it. The gap IS the finding. Anchoring PUE at 1.58 would force a specific assumption that contradicts the EIA data constraint.

2. **FERC interconnection data:**  
   All 7 ISOs require manual portal download. `ferc_interconnection_datacenter` table is empty.  
   **Decision needed:** Build a PJM scraper (public queue, no login required) or treat FERC as a next-sprint feature?  
   Recommendation: Next sprint. PJM's public queue format is complex and will need 2–3 hours of data-wrangling. The model works fine without it.

3. **MLflow file-based tracking deprecation:**  
   MLflow 2.9+ warns that file-based tracking backend is deprecated; SQLite migration recommended.  
   Not a blocker tonight (everything works). **Decision needed:** Migrate before step 10 deployment? Yes.

---

## INNOVATIONS

**SARIMA pyfunc registry (step 7.5 between 7 and 8):**  
The best model (SARIMA, MAE 0.056) was not in the MLflow model registry — only Prophet was, because MLflow has native Prophet flavor but no SARIMA flavor. Fixed by wrapping SARIMAX results in a `mlflow.pyfunc.PythonModel` (pickles fitted results, exposes `predict(periods) → DataFrame`). This is now the live Production model. The `evaluate.py` was also refactored from `promote_prophet()` to `promote_winner(model, mae)` which searches all registry versions by the `model` param tag and promotes the actual winner. Logged as commit `21a6272`.

---

## PROGRESS LOG

**[07:00]** Step 7 complete — Streamlit dashboard  
app.py: four-tab dashboard (National Forecast, Model Leaderboard, Institutional Benchmarks, State Breakdown).  
data.py: 9 shared `@st.cache_data` functions.  
All data loading tested end-to-end before commit.  
Health check: `ok`. Smoke test: 5381 bytes HTML, no Python errors.  
Commit: `4ff3942` → pushed.

**[07:08]** SARIMA registry fix (known issue from user's goodnight message)  
Prophet v1 was in Production; SARIMA (actual winner) was not registered.  
Added `SARIMAForecastModel(mlflow.pyfunc.PythonModel)` + `register_sarima()` to sarima_model.py.  
Refactored evaluate.py: `promote_winner()` instead of `promote_prophet()`.  
SARIMA v2 → Production confirmed.  
Commit: `21a6272` → pushed.

**[07:15]** Step 8 — FastAPI public reference API  
7 endpoints: /health, /meta, /actuals, /forecast, /forecast/states/{state}, /models, /benchmarks.  
Fixed HAVING bug in /actuals (count(DISTINCT ds)=48 → =12).  
All endpoints smoke-tested locally.  
Dockerfile: python:3.11-slim + uvicorn[standard].  
Commit: `b1108bc` → pushed.

**[07:30]** Step 9 — methodology writeup + blog  
blog/methodology.md: ~2700 words, full technical writeup.  
blog/post.md: ~1350 words, accessible version.  
blog/figures/: 4 PNGs (national CO₂, benchmark errors, state breakdown, model MAE) via Plotly/kaleido.  
Both docs address the IEA CO₂ gap honestly with root-cause analysis.  
Commit: `1444633` → pushed.

**[07:45]** Step 11 — README + vault update  
README.md: full project docs with findings, scoreboard table, quick-start, API reference.  
Obsidian vault (projects/datacenter-co2-forecaster.md): updated to reflect ai-energy-forecast-tracker — new GitHub URL, findings, leaderboard, 30-sec pitch, TODO.  
Commit: `abe671d` → pushed.

**[07:50]** Tests — integration test suite  
15 tests in tests/test_fusion.py + tests/test_benchmarks.py.  
All 15 pass. Fixed one edge case (partial 2026 year in range test).  
Commit: `1374462` → pushed.

---

## STOPPING

**Reason:** Steps 7, 8, 9, 11 + bonus tests all complete and pushed.  
Step 10 (deployment) is explicitly out of scope per instructions.

**Final git log:**
```
1374462  test: fusion model and benchmarks integration test suite
abe671d  feat(step11): README, vault update, cross-links  
1444633  feat(step9): methodology writeup and blog post with figures
b1108bc  feat(step8): FastAPI public reference API + Dockerfile
21a6272  fix(registry): register SARIMA as pyfunc and promote to Production
4ff3942  feat(steps5-7): four forecast models, multi-model evaluator, benchmarks, dashboard
```

**Next action for you:**
1. Check the two MORNING DECISIONS (IEA gap + FERC) — my recommendations are above
2. Run `streamlit run src/dashboard/app.py` to see the full dashboard
3. Optionally: `pytest tests/` to confirm tests pass on your machine
4. Step 10 (deployment) is ready whenever you want — Streamlit Cloud + Fly.io. Dockerfile is written.
