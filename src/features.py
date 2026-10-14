"""
features.py — Feature Engineering for Groundwater Level Forecasting
====================================================================
This module creates the input features that the ML model will use to
predict groundwater levels 7 days into the future.

KEY CONCEPT: SUPERVISED LEARNING FROM TIME SERIES
  In a normal ML problem, you have features (X) and a target (y).
  For time series, we CREATE features from past values:
  
  Example (simplified):
    Date    | GWL   | lag_1  | lag_7  | roll_7 | TARGET (7 days later)
    Jan 1   | -10.0 | -10.1  | -10.5  | -10.3  | -9.8 (Jan 8 value)
    Jan 2   | -9.9  | -10.0  | -10.4  | -10.2  | -9.7 (Jan 9 value)
    
  The model learns: "given these past patterns, what will the level be in 7 days?"

FEATURES WE CREATE:
  1. Lagged values: GWL from 1, 2, 3, 7, 14, 21, 28 days ago
  2. Rolling averages: Mean GWL over last 7, 14, 30 days
  3. Rolling std: Volatility over last 7, 14 days
  4. Rate of change: How fast is the level changing?
  5. Calendar features: Month, day of year (for seasonal patterns)
"""

import pandas as pd
import numpy as np


def create_lag_features(df, lags=[1, 2, 3, 5, 7, 14, 21, 28]):
    """
    Create lagged groundwater level features.
    
    WHAT IS A LAG?
    lag_1 = yesterday's GWL
    lag_7 = GWL from 7 days ago
    lag_28 = GWL from 28 days ago
    
    WHY THESE SPECIFIC LAGS?
    - lag_1 to lag_3: Capture very recent trends (is the level rising or falling?)
    - lag_5, lag_7: Capture weekly patterns
    - lag_14, lag_21: Capture fortnightly/tri-weekly patterns  
    - lag_28: Capture monthly patterns
    
    These give the model a "memory" of what happened in the recent past.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["gwl"].shift(lag)
    return df


def create_rolling_features(df, windows=[7, 14, 30]):
    """
    Create rolling (moving) average and standard deviation features.
    
    WHAT IS A ROLLING AVERAGE?
    roll_mean_7 = Average GWL over the last 7 days
    
    WHY?
    - Individual daily readings are noisy (pumping, weather, sensor noise)
    - A 7-day average captures the underlying trend
    - A 30-day average captures the broader seasonal movement
    - Rolling std captures how "volatile" the level has been recently
    
    IMPORTANT: min_periods ensures we don't compute averages from too few values.
    The window is "backward-looking" only — no future data leakage.
    """
    df = df.copy()
    for w in windows:
        # shift(1) ensures we don't include today's value (avoid data leakage)
        # We only use values up to yesterday
        shifted = df["gwl"].shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(window=w, min_periods=w).std()
    return df


def create_change_features(df):
    """
    Create rate-of-change features.
    
    These tell the model: "How fast is the groundwater level changing?"
    
    - change_1d: Change from yesterday to today (short-term momentum)
    - change_7d: Change over the last week (medium-term trend)
    - change_14d: Change over the last 2 weeks (longer-term trend)
    
    Positive change = water rising (getting shallower)
    Negative change = water falling (getting deeper)
    """
    df = df.copy()
    df["change_1d"] = df["gwl"] - df["gwl"].shift(1)
    df["change_7d"] = df["gwl"] - df["gwl"].shift(7)
    df["change_14d"] = df["gwl"] - df["gwl"].shift(14)
    return df


def create_calendar_features(df):
    """
    Create calendar-based features from the date.
    
    WHY?
    Groundwater levels follow seasonal patterns:
    - Monsoon season (June-September): Water rises due to rainfall recharge
    - Post-monsoon (Oct-Nov): Levels start declining
    - Summer (March-May): Water at its deepest due to extraction + evaporation
    
    We encode this seasonality using:
    - month: Integer 1-12
    - day_of_year: Integer 1-365 (captures position within the annual cycle)
    - sin/cos of day_of_year: Smooth cyclical encoding 
      (so Dec 31 and Jan 1 are numerically close, unlike raw day_of_year)
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # Cyclical encoding — converts day_of_year into smooth sine/cosine waves
    # This helps the model understand that day 365 is close to day 1
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    
    return df


def create_target(df, horizon=7):
    """
    Create the prediction target: GWL 'horizon' days into the future.
    
    target = gwl at (today + 7 days)
    
    IMPORTANT: This is the value we're trying to PREDICT.
    The model will never see this value as an input — only as the answer
    it's trying to learn.
    
    After creating the target, rows near the end of the dataset will have
    NaN targets (because we don't know what happens 7 days after the last
    observation). These rows are used for generating the final forecast,
    not for training.
    """
    df = df.copy()
    df["target"] = df["gwl"].shift(-horizon)
    return df


def build_feature_matrix(daily_df, horizon=7):
    """
    Complete feature engineering pipeline.
    
    Takes the clean daily GWL series and produces:
    - Feature matrix X (all the input columns)
    - Target vector y (GWL 7 days later)
    - A list of feature column names
    
    Rows with NaN features (from lags/rolling at the start) are dropped.
    Rows with NaN targets (at the end) are separated for final forecasting.
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df = daily_df.copy()
    
    # Step 1: Create all features
    print("\nCreating lag features...")
    df = create_lag_features(df)
    
    print("Creating rolling statistics...")
    df = create_rolling_features(df)
    
    print("Creating rate-of-change features...")
    df = create_change_features(df)
    
    print("Creating calendar features...")
    df = create_calendar_features(df)
    
    # Step 2: Create the target (7-day-ahead GWL)
    print(f"Creating target (GWL {horizon} days ahead)...")
    df = create_target(df, horizon=horizon)
    
    # Step 3: Define feature columns
    # These are ALL the columns the model will use as inputs
    feature_cols = [
        # Lagged values
        "lag_1", "lag_2", "lag_3", "lag_5", "lag_7", "lag_14", "lag_21", "lag_28",
        # Rolling statistics
        "roll_mean_7", "roll_mean_14", "roll_mean_30",
        "roll_std_7", "roll_std_14", "roll_std_30",
        # Rate of change
        "change_1d", "change_7d", "change_14d",
        # Calendar
        "month", "day_of_year", "day_sin", "day_cos",
    ]
    
    # Step 4: Add target_date explicitly for auditability
    df["target_date"] = df["date"] + pd.Timedelta(days=horizon)
    
    # Step 5: Separate forecast rows (no target available) from training data
    forecast_rows = df[df["target"].isna()].dropna(subset=feature_cols).copy().reset_index(drop=True)
    trainable = df[df["target"].notna()].dropna(subset=feature_cols + ["gwl"]).copy().reset_index(drop=True)
    
    print(f"\n--- Feature matrix summary ---")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")
    print(f"  Trainable samples (valid features & target): {len(trainable)}")
    print(f"  Operational forecast rows (latest {len(forecast_rows)} days): {len(forecast_rows)}")
    print(f"  Date range for training: {trainable['date'].min().date()} to {trainable['date'].max().date()}")
    
    return trainable, forecast_rows, feature_cols


if __name__ == "__main__":
    # Quick test
    from preprocessing import preprocess_pipeline
    csv_path = r"data\gwl_tel_6_hourly_cgwb_br_2021_2025 (1).csv"
    daily = preprocess_pipeline(csv_path)
    trainable, forecast, features = build_feature_matrix(daily)
    print(f"\nSample row:\n{trainable.iloc[0]}")
