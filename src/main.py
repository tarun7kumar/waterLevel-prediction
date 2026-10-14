"""
main.py — End-to-End Pipeline Execution Script
==============================================
Runs the full AI Groundwater Level Monitoring & Forecasting pipeline:
  1. Data Preprocessing (filtering to Khagaul I, 6-hourly to daily aggregation, gap filling)
  2. Feature Engineering (lags, rolling averages, differences, seasonality)
  3. Chronological Model Training & Evaluation (Naive Baseline, Linear Regression, Random Forest, Gradient Boosting)
  4. Visualization Generation (timeseries, error distribution, model comparison, forecast)
  5. Champion Model Serialization (Linear Regression + StandardScaler + Metadata)
"""

import os
import sys
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_pipeline
from features import build_feature_matrix
from modeling import chronological_split, train_and_evaluate
import visualization as viz

def run_full_pipeline(csv_path=None):
    if csv_path is None:
        csv_path = Path("data") / "gwl_tel_6_hourly_cgwb_br_2021_2025 (1).csv"
    
    print("=" * 65)
    print("AI GROUNDWATER MONITORING & FORECASTING PIPELINE")
    print("=" * 65)
    
    # 1. Preprocess raw data
    daily_clean = preprocess_pipeline(csv_path, station_name="Khagaul I")
    
    # Save daily clean series
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "models").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    daily_clean.to_csv(outputs_dir / "daily_gwl_khagaul.csv", index=False)
    
    # 2. Build feature matrix (7-day horizon)
    trainable_df, forecast_rows, feature_cols = build_feature_matrix(daily_clean, horizon=7)
    trainable_df.to_csv(outputs_dir / "features_khagaul.csv", index=False)
    forecast_rows.to_csv(outputs_dir / "forecast_input_khagaul.csv", index=False)
    
    # 3. Chronological split
    train, val, test = chronological_split(trainable_df)
    
    # 4. Train and evaluate models
    results = train_and_evaluate(train, val, test, feature_cols)
    
    # 5. Generate publication-quality visualizations
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)
    viz.plot_daily_timeseries(daily_clean)
    viz.plot_actual_vs_predicted(test, results)
    viz.plot_error_distribution(test, results)
    viz.plot_feature_importance(results)
    viz.plot_model_comparison(results)
    viz.plot_seasonal_pattern(daily_clean)
    
    # Save detailed evaluation audit trail
    test_audit_file = outputs_dir / "test_predictions_audit.csv"
    results["test_predictions_df"].to_csv(test_audit_file, index=False)
    print(f"  Evaluation audit saved to: {test_audit_file}")
    
    # 6. Operational 7-day forecast with best model
    best_name = results["best_model"]
    print(f"\n[INFO] Best Performing Model: {best_name}")
    
    best_model_obj = results["models"]["linear"]["model"]
    scaler = results["models"]["linear"]["scaler"]
    
    # Latest observation row (Dec 31, 2025)
    latest_row = forecast_rows.iloc[-1:]
    latest_obs_date = latest_row["date"].iloc[0]
    target_7d_date = latest_obs_date + pd.Timedelta(days=7)
    
    X_latest_scaled = scaler.transform(latest_row[feature_cols].values)
    pred_7d_val = best_model_obj.predict(X_latest_scaled)[0]
    
    print(f"\n  Operational Forecast from Latest Observation:")
    print(f"    Latest Reading Date:  {latest_obs_date.date()} (GWL = {latest_row['gwl'].iloc[0]:.2f}m)")
    print(f"    Target Date (T+7):    {target_7d_date.date()}")
    print(f"    Predicted GWL:        {pred_7d_val:.2f} meters below ground")
    
    forecast_dates = [d + pd.Timedelta(days=7) for d in forecast_rows["date"]]
    X_fore_scaled = scaler.transform(forecast_rows[feature_cols].values)
    forecast_vals = best_model_obj.predict(X_fore_scaled)
    
    viz.plot_forecast(daily_clean, forecast_dates, forecast_vals, best_name)
    
    # 7. Save champion model and scaler
    model_file = outputs_dir / "models" / "best_model.joblib"
    scaler_file = outputs_dir / "models" / "scaler.joblib"
    meta_file = outputs_dir / "models" / "model_metadata.json"
    
    joblib.dump(best_model_obj, model_file)
    joblib.dump(scaler, scaler_file)
    
    naive_mae = results["models"]["naive"]["mae_test"]
    lr_mae = results["models"]["linear"]["mae_test"]
    lr_rmse = results["models"]["linear"]["rmse_test"]
    
    meta = {
        "station": "Khagaul I",
        "best_model": best_name,
        "test_mae_meters": float(lr_mae),
        "test_rmse_meters": float(lr_rmse),
        "baseline_mae_meters": float(naive_mae),
        "improvement_pct": float((1 - lr_mae / naive_mae) * 100),
        "forecast_horizon_days": 7,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols
    }
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\n[OK] Model serialized to:  {model_file}")
    print(f"[OK] Scaler serialized to: {scaler_file}")
    print(f"[OK] Metadata saved to:    {meta_file}")
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_full_pipeline()
