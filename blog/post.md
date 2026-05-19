# Every Major AI Energy Forecast Was Wrong. Here's What the Data Actually Shows.

*And why the divergence between estimates matters more than any single number.*

---

Depending on which report you read, US datacenters consumed somewhere between 73 and 187 terawatt-hours of electricity in 2024 — a range wide enough to fit the entire electricity consumption of Texas inside the uncertainty band. Three major forecasting institutions produced estimates that differed by more than 50% from measured reality. One of the most cited papers in the field predicted flat energy growth through 2030. That prediction was already obsolete by the time the paper was published.

This is the story of those forecasts, why they were wrong, and what a cleaner model shows about where datacenter energy is actually headed.

---

## The Setup: Why Estimates Diverge

When researchers estimate datacenter energy, they make three choices that determine the result:

**1. What counts as "datacenter energy"?**  
IT equipment (servers, networking, storage) uses electricity. So does cooling. So does power conversion and distribution. The ratio of total facility energy to IT-only energy is called PUE (Power Usage Effectiveness). A facility with PUE 1.5 uses 50% more electricity than its servers alone.

The IEA's 183 TWh estimate for 2024 is IT load only. The LBNL and Masanet estimates are total facility. A 183 TWh IT load with PUE 1.5 implies a 275 TWh total facility draw. These are measuring the same thing from different ends of the power meter.

**2. Which emission factor do you use for CO₂?**  
The national average US grid emits about 390 grams of CO₂ per kilowatt-hour. But 35% of US datacenter capacity sits in Virginia, where Dominion Energy runs a relatively clean nuclear and gas portfolio at around 320 g/kWh. If you weight by where the datacenters actually are, your emission factor is meaningfully lower — and your CO₂ estimate drops by roughly 18% before you even start counting buildings.

**3. Did you model the AI inflection?**  
This is where most pre-2023 estimates failed. Every efficiency-based forecast assumed that server consolidation, hyperscale migration, and improved cooling would offset growth in compute workloads. For fifteen years, this was broadly correct. Then 2023 happened.

---

## The Finding That Changes Everything

Training a large foundation model (GPT-4 scale, or Claude-scale) requires sustained multi-month GPU clusters running at near-100% utilization. GPU-based training is fundamentally less energy-efficient per unit of useful compute than the inference and database workloads that defined the datacenter industry through 2022.

Our model infers AI workloads as a fraction of total datacenter energy over time, anchored to public benchmark estimates. The result: AI's share jumped from roughly 5% of datacenter energy before 2023 to 37% by 2024.

That's not a gradual trend. That's a structural break.

Masanet et al.'s 2020 paper — the most rigorous efficiency-adjusted analysis of its time — projected that the efficiency-growth offset would hold through 2030. They were right about the mechanism. They were wrong about the magnitude of the disruption that was coming. The AI boom was not in any historical dataset they could have trained on.

The LBNL 2016 base scenario predicted 73 TWh for 2020. Our model shows 146 TWh for that year — a 50% miss. Grade: F.

---

## What the Model Does

We built a hierarchical Bayesian model that:

1. Starts with EIA monthly commercial sector electricity data (2001–2026, 302 months)
2. Infers three latent variables: what fraction goes to datacenters, what's the PUE, what fraction is AI
3. Anchors those inferences to IEA's 2024 estimates and Guidi et al.'s 2018 CO₂ estimate
4. Produces a posterior distribution over monthly datacenter energy and CO₂

The model is fit using PyMC 5 with NUTS sampling. All code is open source; anyone can run it.

**2024 posterior estimates:**
- Total facility energy: 187.6 TWh (IT load ~140 TWh)
- Annual CO₂: 67.7 Mt
- PUE: 1.34 (posterior mean)
- AI workload fraction: 37%

---

## The Honest Disagreement with IEA

Our CO₂ estimate (67.7 Mt) is 37 Mt below IEA's (105 Mt). This gap is real and we don't paper over it.

IEA uses a bottom-up facility inventory methodology — they count real buildings from industry databases and apply standardized assumptions. Our model infers datacenter share from EIA's commercial sector classification, which may not capture all facility types.

The PUE is also a factor. IEA's methodology implies PUE ~1.58. Our Bayesian model, constrained by both the energy and CO₂ benchmarks simultaneously, converges on 1.34. The two benchmarks are in tension — satisfying IEA's CO₂ number with our energy estimate would require PUE = 1.55, but satisfying IEA's energy number alone implies lower CO₂.

The honest interpretation: 68 Mt is a lower-bound estimate; 105 Mt is IEA's figure using broader scope; the true value is likely in between. The gap represents what is genuinely unknown about US datacenter scope, not a modeling error.

---

## Four Models, One Winner

We trained four forecast models on the posterior mean CO₂ series and evaluated each on a 12-month holdout:

| Model | Holdout MAE | 2030 Projection |
|-------|------------|----------------|
| SARIMA | 0.056 Mt/mo | 80.8 Mt/yr |
| Prophet | 0.121 Mt/mo | 79.0 Mt/yr |
| Seasonal naive | 0.179 Mt/mo | 68.3 Mt/yr |
| OLS regression | 0.391 Mt/mo | 79.2 Mt/yr |

SARIMA wins by a factor of two over Prophet. The CO₂ series has clean 12-month seasonality that ARIMA captures precisely. OLS ranks last because it works on annual aggregates and distributes CO₂ flatly across months, destroying the intra-year variation that makes monthly MAE hard.

The three probabilistic models (SARIMA, Prophet, OLS) agree on the 2030 range: 79–81 Mt/yr. The naive model is lower (68 Mt) because it projects 2025 patterns forward with no growth trend.

---

## What It Means for 2030

**Worst case (no efficiency improvement):** 80–81 Mt CO₂/yr, ~200–220 TWh total facility energy.

**Current trajectory:** SARIMA's 80.8 Mt/yr assumes the current AI build-out continues at its 2023–2025 pace. This may be optimistic about the AI build-out and pessimistic about grid decarbonization. Nuclear license extensions, new solar capacity, and battery storage are all progressing; the emission factor for datacenter-heavy grids may fall faster than the historical trend.

**The institutional forecasts for 2030 (Goldman 310 TWh, EPRI 390 TWh, McKinsey 340 TWh)** will all be gradeable by 2031. The same methodology that graded LBNL and Masanet will grade them. We will find out then whether the current wave of AI infrastructure investment scales as projected or whether the next efficiency breakthrough (inference-optimized hardware, smaller models, chip architectural improvements) absorbs it.

---

## The Two Things We Got Right

Two findings in this analysis are, we believe, methodologically solid and worth highlighting:

**1. The workload-efficiency narrative failed.** Every forecast that relied on efficiency improvements offsetting compute growth missed by 38–50%. This isn't a retrospective criticism — it's a calibration lesson. When efficiency trajectories collide with step-change hardware adoption, historical trends break. Any 2030 forecast that relies primarily on continued efficiency improvement without explicitly modeling the adoption curve of new AI workload types is making the same bet.

**2. Emission factor methodology matters more than the energy estimate.** The ~18% difference between DC-weighted and national-average CO₂ rates is not a rounding issue. Virginia's nuclear grid directly subsidizes the CO₂ accounting of US hyperscale operators. If datacenter capacity diversifies toward Texas or Georgia — both coal/gas heavy — the CO₂ trajectory diverges significantly from the energy trajectory. A model that uses a single national rate will miss this.

---

## Using This Work

The full model is open source at [github.com/LakshayMani0406/ai-energy-forecast-tracker](https://github.com/LakshayMani0406/ai-energy-forecast-tracker).

A public API returns forecast data in JSON:
```
GET /actuals        # annual DC energy + CO₂ from fusion_posterior
GET /forecast       # model forecasts through 2030 (monthly)
GET /benchmarks     # institutional forecast grades
GET /models         # holdout MAE leaderboard
```

The Streamlit dashboard runs locally with `streamlit run src/dashboard/app.py` — no cloud account required.

The model updates automatically when EIA publishes new monthly commercial data. The benchmark scoreboard will automatically grade the 2030 projections from Goldman Sachs, EPRI, McKinsey, and BloombergNEF as actuals become available.

---

*Full technical methodology: [blog/methodology.md](methodology.md)*  
*Repository: github.com/LakshayMani0406/ai-energy-forecast-tracker*
