"""
modeling.py — Model Training, Evaluation, and Forecasting
==========================================================
This module handles:
  1. Chronological train/validation/test splitting
  2. Training three models: Naive Baseline, Linear Regression, Random Forest
  3. Evaluating all models with MAE and RMSE
  4. Generating the final 7-day-ahead forecast

MODEL COMPARISON STRATEGY:
  We always start with the simplest possible model and work up:
  
  1. NAIVE BASELINE (Persistence Forecast)
     - Predicts: "The GWL 7 days from now = today's GWL"
     - This is the simplest possible prediction — no learning at all
     - Any useful ML model MUST beat this baseline
     - If it can't beat persistence, the ML model adds no value
  
  2. LINEAR REGRESSION
     - Learns a weighted combination of all features
     - Simple, fast, interpretable
     - Good baseline ML model for comparison
  
  3. RANDOM FOREST REGRESSOR
     - Ensemble of decision trees that "vote" on the prediction
     - Captures non-linear relationships that Linear Regression misses
     - More powerful but harder to interpret
     
EVALUATION METRICS (explained simply):
  MAE (Mean Absolute Error):
    "On average, how many meters off is our prediction?"
    Example: MAE = 0.5 means predictions are typically 0.5 meters wrong
    
  RMSE (Root Mean Squared Error):
    Similar to MAE but penalizes BIG errors more heavily
    If RMSE >> MAE, it means the model occasionally makes large mistakes
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def chronological_split(df, train_target_cutoff="2025-04-30", val_target_cutoff="2025-08-31", horizon=7):
    """
    Split data chronologically into train, validation, and test sets with a
    strict 7-day purge buffer to prevent target leakage.
    
    WHY A PURGE BUFFER?
    - Our target is GWL 7 days into the future (target = GWL_{t+7}).
    - If training inputs ran up to April 30, the April 30 sample would have a
      target on May 7, which falls 7 days INSIDE the validation period.
    - To guarantee ZERO target leakage:
        * Train inputs: date <= (train_target_cutoff - 7 days), so all training targets <= train_target_cutoff.
        * Validation inputs: date > train_target_cutoff AND date <= (val_target_cutoff - 7 days),
          so validation targets fall strictly <= val_target_cutoff.
        * Test inputs: date > val_target_cutoff, strictly after all validation targets!
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    train_feat_end = pd.to_datetime(train_target_cutoff) - pd.Timedelta(days=horizon)
    val_feat_start = pd.to_datetime(train_target_cutoff) + pd.Timedelta(days=1)
    val_feat_end = pd.to_datetime(val_target_cutoff) - pd.Timedelta(days=horizon)
    test_feat_start = pd.to_datetime(val_target_cutoff) + pd.Timedelta(days=1)
    
    train = df[df["date"] <= train_feat_end].copy().reset_index(drop=True)
    val = df[(df["date"] >= val_feat_start) & (df["date"] <= val_feat_end)].copy().reset_index(drop=True)
    test = df[df["date"] >= test_feat_start].copy().reset_index(drop=True)
    
    print("=" * 60)
    print("PURGED CHRONOLOGICAL DATA SPLIT (Zero Target Leakage)")
    print("=" * 60)
    print(f"  Train:      Inputs {train['date'].min().date()} to {train['date'].max().date()} -> Targets end {train['target_date'].max().date()} ({len(train)} samples)")
    print(f"  Validation: Inputs {val['date'].min().date()} to {val['date'].max().date()} -> Targets end {val['target_date'].max().date()} ({len(val)} samples)")
    print(f"  Test:       Inputs {test['date'].min().date()} to {test['date'].max().date()} -> Targets end {test['target_date'].max().date()} ({len(test)} samples)")
    print(f"  Total samples across splits: {len(train) + len(val) + len(test)}")
    
    return train, val, test


def train_and_evaluate(train_df, val_df, test_df, feature_cols):
    """
    Train all models and evaluate them on validation and test sets.
    
    Returns a dictionary of results and trained models.
    """
    # Extract features (X) and targets (y)
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values
    
    # =====================================================================
    # MODEL 1: NAIVE BASELINE (Persistence Forecast)
    # =====================================================================
    # "GWL in 7 days = today's GWL" — uses lag_0 which is just 'gwl' column
    # Since lag_1 is yesterday and we want today, we use the current gwl
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # For the naive baseline, the prediction is simply today's GWL
    # In our feature matrix, the current gwl isn't a feature (to prevent leakage)
    # but we stored it in the dataframe
    naive_pred_val = val_df["gwl"].values
    naive_pred_test = test_df["gwl"].values
    
    print("\n--- Model 1: Naive Baseline (Persistence) ---")
    print("  Strategy: Predict GWL(t+7) = GWL(today)")
    naive_mae_val = mean_absolute_error(y_val, naive_pred_val)
    naive_rmse_val = np.sqrt(mean_squared_error(y_val, naive_pred_val))
    naive_mae_test = mean_absolute_error(y_test, naive_pred_test)
    naive_rmse_test = np.sqrt(mean_squared_error(y_test, naive_pred_test))
    print(f"  Validation - MAE: {naive_mae_val:.4f} m, RMSE: {naive_rmse_val:.4f} m")
    print(f"  Test       - MAE: {naive_mae_test:.4f} m, RMSE: {naive_rmse_test:.4f} m")
    
    # =====================================================================
    # MODEL 2: LINEAR REGRESSION
    # =====================================================================
    # Scale features for Linear Regression (important for numerical stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on training data ONLY
    X_val_scaled = scaler.transform(X_val)           # transform validation
    X_test_scaled = scaler.transform(X_test)         # transform test
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    lr_pred_val = lr_model.predict(X_val_scaled)
    lr_pred_test = lr_model.predict(X_test_scaled)
    
    print("\n--- Model 2: Linear Regression ---")
    print("  Strategy: Weighted linear combination of all features")
    lr_mae_val = mean_absolute_error(y_val, lr_pred_val)
    lr_rmse_val = np.sqrt(mean_squared_error(y_val, lr_pred_val))
    lr_mae_test = mean_absolute_error(y_test, lr_pred_test)
    lr_rmse_test = np.sqrt(mean_squared_error(y_test, lr_pred_test))
    print(f"  Validation - MAE: {lr_mae_val:.4f} m, RMSE: {lr_rmse_val:.4f} m")
    print(f"  Test       - MAE: {lr_mae_test:.4f} m, RMSE: {lr_rmse_test:.4f} m")
    
    # Show feature importances for Linear Regression (coefficients)
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lr_model.coef_
    }).sort_values("coefficient", key=abs, ascending=False)
    print(f"\n  Top 5 most important features (by coefficient magnitude):")
    for _, row in coef_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['coefficient']:.4f}")
    
    # =====================================================================
    # MODEL 3: RANDOM FOREST
    # =====================================================================
    # Random Forest doesn't need feature scaling
    rf_model = RandomForestRegressor(
        n_estimators=200,       # 200 trees in the forest
        max_depth=15,           # Limit depth to prevent overfitting
        min_samples_leaf=5,     # Each leaf must have at least 5 samples
        random_state=42,        # Reproducibility
        n_jobs=-1               # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    
    rf_pred_val = rf_model.predict(X_val)
    rf_pred_test = rf_model.predict(X_test)
    
    print("\n--- Model 3: Random Forest ---")
    print("  Strategy: Ensemble of 200 decision trees")
    rf_mae_val = mean_absolute_error(y_val, rf_pred_val)
    rf_rmse_val = np.sqrt(mean_squared_error(y_val, rf_pred_val))
    rf_mae_test = mean_absolute_error(y_test, rf_pred_test)
    rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))
    print(f"  Validation - MAE: {rf_mae_val:.4f} m, RMSE: {rf_rmse_val:.4f} m")
    print(f"  Test       - MAE: {rf_mae_test:.4f} m, RMSE: {rf_rmse_test:.4f} m")
    
    # Feature importances from Random Forest
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n  Top 5 most important features (by tree importance):")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # =====================================================================
    # MODEL 4: GRADIENT BOOSTING
    # =====================================================================
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    gb_pred_val = gb_model.predict(X_val)
    gb_pred_test = gb_model.predict(X_test)
    
    print("\n--- Model 4: Gradient Boosting ---")
    print("  Strategy: Sequential trees, each correcting previous errors")
    gb_mae_val = mean_absolute_error(y_val, gb_pred_val)
    gb_rmse_val = np.sqrt(mean_squared_error(y_val, gb_pred_val))
    gb_mae_test = mean_absolute_error(y_test, gb_pred_test)
    gb_rmse_test = np.sqrt(mean_squared_error(y_test, gb_pred_test))
    print(f"  Validation - MAE: {gb_mae_val:.4f} m, RMSE: {gb_rmse_val:.4f} m")
    print(f"  Test       - MAE: {gb_mae_test:.4f} m, RMSE: {gb_rmse_test:.4f} m")
    
    # =====================================================================
    # COMPARISON SUMMARY
    # =====================================================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 60)
    print(f"{'Model':<25} {'MAE (m)':>10} {'RMSE (m)':>10} {'vs Baseline':>15}")
    print("-" * 60)
    
    models_test = {
        "Naive Baseline": (naive_mae_test, naive_rmse_test),
        "Linear Regression": (lr_mae_test, lr_rmse_test),
        "Random Forest": (rf_mae_test, rf_rmse_test),
        "Gradient Boosting": (gb_mae_test, gb_rmse_test),
    }
    
    for name, (mae, rmse) in models_test.items():
        if name == "Naive Baseline":
            improvement = "-"
        else:
            pct = (1 - mae / naive_mae_test) * 100
            improvement = f"{pct:+.1f}% MAE"
        print(f"{name:<25} {mae:>10.4f} {rmse:>10.4f} {improvement:>15}")
    
    # Find best model
    best_name = min(models_test, key=lambda k: models_test[k][0])
    print(f"\n  [BEST] Best model: {best_name}")
    
    test_predictions_df = pd.DataFrame({
        "observation_date": test_df["date"],
        "target_date": test_df["target_date"],
        "actual_gwl": y_test,
        "naive_pred": naive_pred_test,
        "linear_pred": lr_pred_test,
        "rf_pred": rf_pred_test,
        "gb_pred": gb_pred_test,
    })
    
    results = {
        "models": {
            "naive": {"mae_test": naive_mae_test, "rmse_test": naive_rmse_test,
                     "mae_val": naive_mae_val, "rmse_val": naive_rmse_val,
                     "pred_test": naive_pred_test, "pred_val": naive_pred_val},
            "linear": {"model": lr_model, "scaler": scaler,
                      "mae_test": lr_mae_test, "rmse_test": lr_rmse_test,
                      "mae_val": lr_mae_val, "rmse_val": lr_rmse_val,
                      "pred_test": lr_pred_test, "pred_val": lr_pred_val},
            "rf": {"model": rf_model,
                  "mae_test": rf_mae_test, "rmse_test": rf_rmse_test,
                  "mae_val": rf_mae_val, "rmse_val": rf_rmse_val,
                  "pred_test": rf_pred_test, "pred_val": rf_pred_val,
                  "importances": imp_df},
            "gb": {"model": gb_model,
                  "mae_test": gb_mae_test, "rmse_test": gb_rmse_test,
                  "mae_val": gb_mae_val, "rmse_val": gb_rmse_val,
                  "pred_test": gb_pred_test, "pred_val": gb_pred_val},
        },
        "best_model": best_name,
        "feature_cols": feature_cols,
        "test_predictions_df": test_predictions_df,
    }
    
    return results
