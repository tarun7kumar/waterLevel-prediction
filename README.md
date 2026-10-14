# AI-Based Groundwater Level Monitoring and Forecasting System

> **B.Tech Electronics & Communication Engineering (ECE) Final Year Project**  
> **Target:** 7-Day Ahead Groundwater Level Forecasting using Historical Telemetry and Machine Learning

---

## 1. Project Overview

Groundwater depletion is a critical environmental and agricultural challenge. This project implements an end-to-end AI monitoring and predictive forecasting system that:
1. Interfaces with hardware telemetry sensors placed at ground level measuring groundwater depth (sign convention: `-4m` represents water table 4 meters below the ground surface).
2. Cleans, aggregates, and engineers features from historical groundwater time series data (Central Ground Water Board - CGWB Bihar Telemetry).
3. Evaluates machine learning models against a rigorous persistence baseline to forecast groundwater levels **7 days into the future**.
4. Delivers publication-quality visualizations, error diagnostics, and a serialized production model.

---

## 2. Results Summary

All models were evaluated on an unseen, held-out test period (**Sept 1, 2025 – Dec 24, 2025**, target dates ending Dec 31, 2025) using **purged walk-forward chronological splitting** (7-day purge buffer ensuring zero target leakage):

| Model | Test MAE (m) | Test RMSE (m) | Performance vs Baseline | Best For |
|:---|:---:|:---:|:---:|:---|
| **Naive Baseline (Persistence)** | 0.3264 m | 0.4287 m | — | Benchmark |
| **Linear Regression (Champion)** 🏆 | **0.2707 m** | **0.3830 m** | **+17.1% MAE improvement** | **Operational Forecasting** |
| **Random Forest Regressor** | 1.9466 m | 2.0026 m | -496.3% MAE | Non-linear baseline |
| **Gradient Boosting Regressor** | 1.9338 m | 2.0009 m | -492.4% MAE | Gradient tree boosting |

### Key Machine Learning Finding (The Tree Extrapolation Ceiling):
* **Why did Linear Regression outperform Random Forest and Gradient Boosting?**
  * Decision trees predict by averaging training samples within terminal leaves. Consequently, a tree **cannot predict any value outside the range of targets observed during training** $[y_{\min}, y_{\max}]$.
  * During the 2023–April 2025 training period, the shallowest water level observed was ~ `-7.4m`.
  * In late 2025 (test period), heavy monsoon recharge brought the water table to an unprecedented **`-4.9m`**.
  * The tree models hit their leaf ceiling at ~`-7.0m` and could not follow the water table upward, incurring ~`1.9m` error.
  * In contrast, **Linear Regression** learned weights on recent lags summing to $\approx 1.0$, which allowed it to extrapolate smoothly into uncharted territory, achieving an outstanding **`0.271m` (~27.1 cm) error** 7 days ahead!

---

## 3. Model Formula

The final operational model is a **standardized multiple linear regression** that predicts the groundwater level 7 days ahead:

$$
\hat{y}_{t+7} = \beta_0 + \sum_{i=1}^{21} \beta_i \left(\frac{x_{i,t} - \mu_i}{\sigma_i}\right)
$$

Where:

* $\hat{y}_{t+7}$ is the predicted groundwater level on day $t+7$.
* $x_{i,t}$ is feature $i$ calculated from information available on day $t$.
* $\mu_i$ and $\sigma_i$ are the training-set mean and standard deviation used by `StandardScaler`.
* $\beta_0$ is the intercept and $\beta_i$ are the learned regression coefficients.

The training target is:

$$
y_t = \text{GWL}_{t+7}
$$

The 21 input terms are lagged groundwater levels, rolling means and standard deviations, rates of change, and calendar features. The persistence baseline used for comparison is:

$$
\hat{y}_{t+7} = \text{GWL}_t
$$

---

## 4. Repository Directory Structure

```
project-1/
├── data/
│   └── gwl_tel_6_hourly_cgwb_br_2021_2025 (1).csv  # Raw CGWB 6-hourly telemetry data
├── notebooks/
│   ├── 01_data_inspection.ipynb      # Phase 1: EDA, station coverage, data quality checks
│   ├── 02_preprocessing.ipynb        # Phase 2: Daily aggregation, gap filling, 21 features
│   └── 03_modeling.ipynb             # Phase 3: Model training, evaluation, 7-day forecast
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data loading, station filtering, daily mean aggregation
│   ├── features.py                   # Lags, rolling statistics, change velocity, calendar features
│   ├── modeling.py                   # Chronological split, training, evaluation metrics
│   ├── visualization.py              # Publication-quality plotting routines
│   └── main.py                       # Single-command end-to-end execution script
├── outputs/
│   ├── figures/                      # 7 generated plots (time series, forecast, errors)
│   ├── models/                       # best_model.joblib, scaler.joblib, model_metadata.json
│   ├── daily_gwl_khagaul.csv         # Clean continuous daily groundwater series
│   ├── features_khagaul.csv          # Complete 21-feature matrix with target
│   └── forecast_input_khagaul.csv    # Latest rows for operational 7-day forecasting
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 5. How to Run

### Option A: Run the End-to-End Pipeline (Single Command)
Execute the complete pipeline from terminal:
```bash
python src/main.py
```
This automatically runs data loading, station filtering (Khagaul I), daily aggregation, 21-feature creation, model evaluation, figure generation, and saves the trained model to `outputs/models/best_model.joblib`.

### Option B: Interactive Jupyter Notebooks
Step through the project interactively in Jupyter:
1. `notebooks/01_data_inspection.ipynb` — Inspect raw telemetry and explore station quality.
2. `notebooks/02_preprocessing.ipynb` — Understand daily aggregation and feature engineering.
3. `notebooks/03_modeling.ipynb` — Train models, analyze error distributions, and plot the 7-day forecast.

---

## 6. Hardware & Sensor Compatibility

* **Hardware Output:** Ultrasonic/pressure level transmitter measuring water level depth from ground surface.
* **Sign Convention:** Distance below ground is recorded as negative meters (`-4.0` = 4 meters deep).
* **Direct Integration:** Because the CGWB telemetry dataset and our pipeline use the exact same negative meter standard, live sensor readings can be fed directly into `src/features.py` and `outputs/models/best_model.joblib` to produce real-time 7-day forecasts on physical hardware or an IoT dashboard.
