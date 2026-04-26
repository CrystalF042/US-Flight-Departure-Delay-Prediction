# US Flight Departure Delay Prediction

A calibrated LightGBM classifier for pre-departure flight delay prediction on US domestic flights, trained on ~6.97M BTS records (2024) augmented with NOAA daily weather observations. Deployed as an interactive Shiny for Python application with SHAP-based per-flight explanations.

**🔗 [Live Demo: Flight Delay Predictor](https://crystalguo.shinyapps.io/flight-delay-predictor/)**

---

## Key Findings

1. **Weather integration is the most impactful feature addition.** NOAA daily observations pushed origin precipitation to SHAP rank 3 globally, with 6 of the top 15 features being weather variables. The model learned a nonlinear rain threshold (~5mm) and a rain × departure-hour interaction reflecting delay propagation.

2. **LightGBM and CatBoost performed identically** (CV ROC-AUC: 0.710 vs 0.709), both outperforming Logistic Regression by ~2.3pp. Weather features introduced threshold effects that only tree-based models could exploit.

3. **Scheduled departure hour dominates all features** (~3× the SHAP importance of the next predictor), reflecting delay compounding through daily aircraft rotations.

4. **Isotonic regression calibration** ensures predicted probabilities match actual delay rates — a "24% chance" means exactly that.

5. **The Shiny app** provides calibrated probability, color-coded risk level, and SHAP-based top contributing factors for any user-specified flight.

---

## 1. Introduction

This project predicts whether a US domestic flight departure will be delayed >15 minutes (FAA definition) using only pre-departure information. Following Sternberg et al. (2021), it addresses the root delay problem at ensemble scope using ML methods. The literature identifies a performance ceiling of ROC-AUC 0.65–0.70 without real-time congestion data (Sternberg et al., 2021; Rebollo & Balakrishnan, 2014); this project tests whether daily weather observations can push past that ceiling.

All features respect a strict **pre-departure prediction scenario**: operational outcomes (actual departure time, taxi-out, air time, arrival delay) are excluded. Historical baselines are computed using only past flights relative to each observation — never future data.

---

## 2. Data

### Flight Data

| Attribute | Value |
|---|---|
| Source | BTS Reporting Carrier On-Time Performance |
| Period | January – December 2024 |
| Flights (after filtering) | ~6.97 million |
| Airports | 373 |
| Carriers | 17 |
| Target | `dep_delay > 15 min` (~20% positive) |

The 2024 window avoids COVID-era distortions (2020–2021) while capturing current post-recovery patterns.

### Weather Data

| Attribute | Value |
|---|---|
| Source | NOAA daily observations via Meteostat API |
| Airport coverage | 333 airports (98.2% of flights) |
| Variables | tavg, tmin, tmax, prcp, snow, wspd, pres |
| Join keys | Origin + destination airport × flight date |
| Missing values | prcp/snow → 0; temperature/wind → airport monthly mean |

### Feature Summary (30 predictors)

| Category | Features |
|---|---|
| Temporal | month, day of week, departure hour, is_weekend, is_peak_hour, season |
| Carrier | carrier ID, carrier historical avg delay |
| Airport | origin/dest IDs, origin historical avg delay |
| Route | route ID, route avg delay, distance, elapsed time, distance bucket |
| Weather | 7 variables × 2 airports + engineered flags |

### Data Splits

| Split | Period | Purpose |
|---|---|---|
| Training | Jan – Aug 2024 | Model fitting via 3-fold expanding-window CV |
| Validation | Sep – Oct 2024 | Calibration + threshold tuning |
| Test | Nov – Dec 2024 | Final evaluation (used once) |

---

## 3. Methodology

### 3.1 Models

**Logistic Regression** serves as the linear baseline. **LightGBM** provides leaf-wise gradient boosting with native categorical support, suited to high-cardinality features like airport codes. **CatBoost** offers ordered boosting with symmetric trees, included based on Alfarhood et al. (2024) findings for flight delay tasks. The lineup compares one linear model against two algorithmically distinct boosting methods.

### 3.2 Cross-Validation and Tuning

```
All Data (Jan–Dec 2024, ~6.97M flights)
│
├── Training Pool (Jan–Aug) ──→ CV + Hyperparameter Tuning
│     │
│     ├── Fold 1: Train Jan–Feb  →  Validate Mar–Apr
│     ├── Fold 2: Train Jan–Apr  →  Validate May–Jun
│     └── Fold 3: Train Jan–Jun  →  Validate Jul–Aug
│
├── Validation (Sep–Oct) ──→ Probability Calibration + Threshold Tuning
│
└── Test (Nov–Dec) ──→ Final Evaluation (used once)
```

3-fold expanding-window CV was applied on a 2M-row stratified subsample, ensuring the model always trains on past data and validates on future data. Hyperparameters were optimized via **Optuna** (Bayesian optimization) with **PR-AUC** as the objective.

| Model | Optuna Trials | Most Important Hyperparameter | Importance |
|---|---|---|---|
| Logistic Regression | 20 | `C` (regularization strength) | 90% |
| LightGBM | 50 | `learning_rate` | 63% |
| CatBoost | 30 | `random_strength` | 36% |

> *Hyperparameter importance is reported by Optuna's fANOVA analysis, which measures how much each hyperparameter's variation explains the variance in the objective (PR-AUC) across all trials.*

Regularization-related parameters dominated importance across all three models, indicating the key modeling decision is controlling overfitting rather than architecture — consistent with a feature-limited problem where the signal ceiling is modest.

### 3.3 Calibration and Threshold Selection

**Isotonic regression** on the validation set maps raw LightGBM scores to calibrated probabilities (Brier: 0.1097 → 0.1086).

<!-- INSERT IMAGE: calibration_reliability.png -->
*Figure 1: Reliability diagram. Before calibration (left), predictions under-estimate delay probability. After isotonic regression (right), predictions align with actual delay rates.*

The F1-optimal threshold of **0.160** was identified via grid search on the validation set, yielding ~51% recall at ~22% precision. The Shiny app shows calibrated probability directly with color-coded risk levels (🟢 low < 15%, 🟡 moderate 15–30%, 🔴 high > 30%) rather than a binary decision.

<!-- INSERT IMAGE: threshold_tradeoff.png -->
*Figure 2: Precision–recall–F1 tradeoff across thresholds. Optimal F1 at threshold 0.160.*

---

## 4. Results

### 4.1 Cross-Validation

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic Regression | 0.6874 ± 0.006 | 0.4080 ± 0.018 | 0.1776 ± 0.012 |
| **LightGBM** | **0.7101 ± 0.008** | **0.4367 ± 0.018** | **0.1682 ± 0.011** |
| CatBoost | 0.7085 ± 0.010 | 0.4361 ± 0.024 | 0.1685 ± 0.009 |

<!-- INSERT IMAGE: model_comparison.png -->
*Figure 3: CV performance comparison with ± 1 std error bars.*

LightGBM selected as final model — equivalent performance to CatBoost at ~2.5× faster training.

### 4.2 Test Set (Nov–Dec 2024)

| Metric | Value |
|---|---|
| ROC-AUC | 0.664 |
| PR-AUC | 0.292 |
| Precision | 0.281 |
| Recall | 0.480 |
| F1 | 0.354 |

<!-- INSERT IMAGE: confusion_matrix_test.png -->
*Figure 4: Confusion matrix at threshold 0.160. 94,528 true positives / 242,298 false positives / 102,204 false negatives / 695,155 true negatives.*

The CV-to-test gap (0.710 → 0.664) reflects seasonal distribution shift: training on Jan–Aug misses winter weather and holiday patterns present in the Nov–Dec test period. The test ROC-AUC is consistent with the literature ceiling for pre-departure prediction (Sternberg et al., 2021).

---

## 5. Interpretability (SHAP)

SHAP values computed on 10,000 test flights. Feature importance ranked by mean |SHAP value|:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | scheduled_dep_hour | 0.469 |
| 2 | op_unique_carrier | 0.161 |
| 3 | origin_prcp | 0.119 |
| 4 | route_avg_dep_delay | 0.115 |
| 5 | day_of_week | 0.095 |
| 6 | dest_prcp | 0.081 |
| 7 | origin_pres | 0.080 |

<!-- INSERT IMAGE: shap_importance_bar_v2.png -->
*Figure 5: Global feature importance (mean |SHAP|). 6 of the top 15 features are weather variables.*

<!-- INSERT IMAGE: shap_summary_v2.png -->
*Figure 6: SHAP beeswarm plot. Red = high feature value, blue = low. Evening departure hours push strongly toward delay; high precipitation does the same.*

`origin_avg_dep_delay` — the top historical baseline before weather integration — dropped to rank 22, displaced by daily weather signals. The engineered `origin_bad_weather` flag (rank 15) was outperformed by raw `origin_prcp` (rank 3), confirming that tree models extract better thresholds from continuous features than from hand-crafted binary flags.

<!-- INSERT IMAGE: shap_dependence_origin_prcp.png -->
*Figure 7: SHAP dependence for origin precipitation, colored by departure hour. Nonlinear threshold at ~5mm; evening flights (pink/red) receive higher SHAP values than morning flights (blue) at the same precipitation level — the model discovered the delay-propagation interaction independently.*

---

## 6. Shiny Application

**🔗 [Live Demo](https://crystalguo.shinyapps.io/flight-delay-predictor/)**

The Shiny for Python application translates the trained model into an interactive tool where users can assess delay risk for any upcoming flight. The app loads the serialized LightGBM classifier, isotonic calibrator, baseline lookups, and weather reference data from the `models/shiny_bundle/` directory.

**Inputs:** origin airport, destination airport, carrier, scheduled departure date and time.

**Outputs:**
- Calibrated delay probability displayed as a percentage with a color-coded risk gauge (🟢 < 15% / 🟡 15–30% / 🔴 > 30%)
- Top contributing factors derived from SHAP values for that specific flight (e.g., "Evening departure +8%, Rain at origin +6%, Carrier track record +3%")
- Historical context showing the airport's and carrier's baseline delay rates for comparison

A ROC-AUC of 0.664 is a modest number in isolation. But the app demonstrates something more important: the complete pipeline from raw government data to actionable, transparent user-facing predictions. When a user sees:

> *"Your 6 PM American Airlines flight from JFK has a **31% delay probability**.
> Top contributors: Evening departure (+8%), rain at origin (+6%), carrier's historical track record (+4%)."*

...they receive genuinely useful information. They understand *why* the model thinks their flight is at risk, they can judge whether the reasoning makes sense (is it actually raining at JFK today?), and they can act on it.

---

## 7. Limitations and Future Work

### Model Limitations

The seasonal generalization gap (CV 0.710 → test 0.664) would narrow with full-year training data. Real-time congestion features (FAA OPSNET, rolling airport delay averages) would address the dominant missing predictor identified in the literature. The near-identical LightGBM/CatBoost performance suggests the current feature set is near its information ceiling; further gains require richer data, not more complex models. Future extensions include hourly METAR weather, congestion proxies, multi-year training (2022–2024), and a Stage 2 regression model for delay magnitude estimation.

### Weather Data: Historical, Not Real-Time

The deployed Shiny app uses a static lookup of **historical climatology** — the typical weather observed at each airport on each calendar day during 2024 (NOAA daily observations).

- ✅ **Suitable for:** *"What is the typical delay risk for a flight from JFK in mid-June?"* — the app answers reliably because it uses representative weather conditions.
- ❌ **Not suitable for:** *"What is the delay probability for my flight tomorrow?"* — tomorrow's actual weather may differ significantly from the historical average.

This was a deliberate design choice for a portfolio project: the demo runs anywhere without external API keys or rate limits, maintains independence from third-party services, and keeps the focus on the modeling pipeline (calibrated probabilities, leakage-safe baselines, SHAP explanations) rather than weather data plumbing.

### Extending to Real-Time

The model accepts weather features identically regardless of source — only the data layer needs to change. Replace `weather_reference.parquet` with calls to a forecast API:

| Provider | Free Tier | Notes |
|---|---|---|
| OpenWeatherMap | 1,000 calls/day | Easiest, broad coverage |
| Tomorrow.io | 500 calls/day | Higher resolution, ML-optimized |
| NOAA NDFD | Unlimited (US only) | Free, requires more setup |

The change is localized in `app/models/feature_computation.py`: replace the `(iata, day_of_year)` parquet lookup with an API call for `(iata, fl_date)`.

---

## 8. Reproduction

```
flight-delay-prediction/
├── README.md
├── requirements.txt
├── data/                           # gitignored
├── notebooks/
│   └── flight_delay_modeling.ipynb
├── src/features/baselines.py       # leakage-safe baselines
├── models/shiny_bundle/            # app artifacts
├── app/app.py                      # Shiny application
└── reports/figures/
```

## Project Origin

This repository builds on an earlier collaborative project.

The collaborative phase focused on:
- Exploratory Data Analysis (EDA)
- Data cleaning
- Feature engineering

This repository represents an independent continuation of that work, expanding the dataset to focusing on:
- Model development
- Model training
- Hyperparameter tuning
- Model evaluation
- Shiny app development

Some preprocessing ideas and code were adapted from the earlier project:
https://github.com/CrystalF042/applied-statistics


## References

- Sternberg et al. (2021). A Review on Flight Delay Prediction. *arXiv:1703.06118*.
- Rebollo & Balakrishnan (2014). Characterization and prediction of air traffic delays. *Transp. Res. Part C*, 44, 231–241.
- Alfarhood et al. (2024). Flight delay prediction using CatBoost. *Applied Sciences*.
- Hatıpoğlu & Tosun (2024). Flight delay prediction with ML models. *J. Air Transp. Mgmt.*
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.
