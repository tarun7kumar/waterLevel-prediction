"""
preprocessing.py — Data Loading, Cleaning, and Daily Aggregation
================================================================
This module handles:
  1. Loading the raw CGWB Bihar groundwater CSV
  2. Filtering to a selected station (default: Khagaul I)
  3. Parsing timestamps and sorting chronologically
  4. Aggregating 6-hourly readings to daily means
  5. Handling gaps (missing days) via forward-fill for small gaps
  6. Saving the clean daily time series

WHY DAILY AGGREGATION?
  - Our prediction target is 7 days ahead — daily resolution is natural
  - It smooths out the ~0.3m diurnal pumping/recovery cycle
  - Days missing 1-2 of 4 readings still produce valid daily means
  - Reduces noise without losing the trend signal
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(csv_path):
    """
    Load the raw CSV and parse the essential columns.
    
    Returns a DataFrame with columns: Station, District, Latitude, Longitude,
    RL_MSL, datetime, gwl
    """
    df = pd.read_csv(csv_path)
    
    # The groundwater level column name is long — rename it for convenience
    gwl_col = "Groundwater Level Telemetry 6 Hourly (meter)"
    
    # Parse the timestamp column
    # Format in the CSV: "02-03-2023 00:00" (DD-MM-YYYY HH:MM)
    df["datetime"] = pd.to_datetime(
        df["Data Acquisition Time"], 
        format="%d-%m-%Y %H:%M"
    )
    
    # Keep only the columns we need
    df = df.rename(columns={gwl_col: "gwl"})
    keep_cols = ["Station", "District", "Latitude", "Longitude", "RL_MSL", "datetime", "gwl"]
    df = df[keep_cols].copy()
    
    return df


def filter_station(df, station_name="Khagaul I"):
    """
    Filter to a single monitoring station.
    
    Why filter? Each station has its own groundwater behavior depending on
    geology, depth, local pumping, etc. Mixing stations would confuse the model.
    We pick one station with the best data quality and train a model for it.
    """
    station_df = df[df["Station"] == station_name].copy()
    station_df = station_df.sort_values("datetime").reset_index(drop=True)
    
    print(f"Station: {station_name}")
    print(f"  Records: {len(station_df):,}")
    print(f"  Date range: {station_df['datetime'].min()} to {station_df['datetime'].max()}")
    print(f"  GWL range: {station_df['gwl'].min():.2f} to {station_df['gwl'].max():.2f} meters")
    
    return station_df


def aggregate_to_daily(station_df):
    """
    Convert 6-hourly readings to daily mean groundwater level.
    
    WHY DAILY MEAN (not min, max, or last reading)?
    - Mean captures the average condition of the day
    - Less sensitive to single outlier readings
    - Smooths the diurnal cycle (groundwater naturally fluctuates within a day
      due to pumping, evapotranspiration, and tidal effects)
    
    Returns a DataFrame indexed by date with column 'gwl' (daily mean).
    """
    station_df = station_df.copy()
    station_df["date"] = station_df["datetime"].dt.date
    
    daily = station_df.groupby("date").agg(
        gwl=("gwl", "mean"),           # Daily mean groundwater level
        readings_count=("gwl", "count") # How many readings contributed
    ).reset_index()
    
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    
    print(f"\nDaily aggregation:")
    print(f"  Total days with data: {len(daily)}")
    print(f"  Days with all 4 readings: {(daily['readings_count'] == 4).sum()}")
    print(f"  Days with fewer readings: {(daily['readings_count'] < 4).sum()}")
    print(f"  Min readings in a day: {daily['readings_count'].min()}")
    
    return daily


def fill_gaps(daily_df, max_gap_days=3):
    """
    Create a continuous daily time series by filling small gaps.
    
    WHY FILL GAPS?
    - Our model needs lagged features (yesterday, 2 days ago, etc.)
    - If a day is missing, we can't compute these lags
    - Small gaps (1-3 days) are filled with forward-fill (carry last known value)
    - Larger gaps are left as NaN — rows with NaN features will be dropped later
    
    WHAT IS FORWARD-FILL?
    - If Monday = -10.5 and Tuesday is missing, we set Tuesday = -10.5
    - This assumes groundwater doesn't change dramatically in 1-3 days
    - For groundwater (which changes slowly), this is a reasonable assumption
    
    Args:
        daily_df: DataFrame with 'date' and 'gwl' columns
        max_gap_days: Maximum consecutive missing days to fill (default: 3)
    
    Returns:
        DataFrame with continuous daily index
    """
    # Create a complete date range from first to last observation
    full_range = pd.date_range(
        start=daily_df["date"].min(), 
        end=daily_df["date"].max(), 
        freq="D"
    )
    
    # Set date as index and reindex to the full range
    daily_df = daily_df.set_index("date")
    daily_df = daily_df.reindex(full_range)
    daily_df.index.name = "date"
    
    # Count missing days before filling
    missing_before = daily_df["gwl"].isna().sum()
    total_days = len(daily_df)
    
    # Forward-fill only up to max_gap_days consecutive missing values
    # limit=max_gap_days means: fill at most this many consecutive NaNs
    daily_df["gwl"] = daily_df["gwl"].ffill(limit=max_gap_days)
    
    missing_after = daily_df["gwl"].isna().sum()
    
    print(f"\nGap filling (max {max_gap_days} consecutive days):")
    print(f"  Total days in range: {total_days}")
    print(f"  Missing before fill: {missing_before}")
    print(f"  Missing after fill:  {missing_after}")
    print(f"  Days filled: {missing_before - missing_after}")
    
    # Reset index so 'date' is a column again
    daily_df = daily_df.reset_index()
    
    return daily_df


def preprocess_pipeline(csv_path, station_name="Khagaul I"):
    """
    Complete preprocessing pipeline: load → filter → aggregate → fill gaps.
    
    Returns the clean daily time series ready for feature engineering.
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n--- Loading raw data ---")
    raw = load_raw_data(csv_path)
    print(f"  Total rows loaded: {len(raw):,}")
    print(f"  Unique stations: {raw['Station'].nunique()}")
    
    # Step 2: Filter to selected station
    print(f"\n--- Filtering to station: {station_name} ---")
    station = filter_station(raw, station_name)
    
    # Step 3: Aggregate to daily
    print("\n--- Aggregating to daily means ---")
    daily = aggregate_to_daily(station)
    
    # Step 4: Fill small gaps
    print("\n--- Filling small gaps ---")
    daily_clean = fill_gaps(daily, max_gap_days=3)
    
    # NOTE: We preserve the unbroken daily calendar timeline!
    # Unfillable gaps (>3 days) remain as NaN so that subsequent lag shifts
    # (shift(1), shift(7)) and target shifts (shift(-7)) strictly correspond to
    # true calendar days without silently jumping across missing time intervals.
    # Rows with NaN features/targets will be cleanly dropped in features.py.
    unfilled_days = daily_clean["gwl"].isna().sum()
    
    print(f"\n--- Continuous daily dataset ---")
    print(f"  Total calendar days: {len(daily_clean)}")
    print(f"  Valid observed/filled days: {len(daily_clean) - unfilled_days}")
    print(f"  Unfilled gap days (kept as NaN for calendar integrity): {unfilled_days}")
    print(f"  Date range: {daily_clean['date'].min().date()} to {daily_clean['date'].max().date()}")
    
    return daily_clean


if __name__ == "__main__":
    csv_path = r"data\gwl_tel_6_hourly_cgwb_br_2021_2025 (1).csv"
    daily = preprocess_pipeline(csv_path)
    
    # Save for inspection
    out_path = Path("outputs") / "daily_gwl_khagaul.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
