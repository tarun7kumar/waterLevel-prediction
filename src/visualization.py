"""
visualization.py — Plots and Charts for the Forecasting Project
================================================================
Generates publication-quality visualizations:
  1. Raw time series plot (daily GWL over time)
  2. Actual vs Predicted comparison on test set
  3. Error distribution histogram
  4. Feature importance chart
  5. 7-day forecast plot
  6. Model comparison bar chart
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

# Use a clean, modern style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

FIGURES_DIR = Path("outputs/figures")


def save_fig(fig, name):
    """Save a figure to the outputs/figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_daily_timeseries(daily_df):
    """
    Plot 1: The complete daily groundwater level time series.
    Shows the overall trend and seasonal patterns.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(daily_df["date"], daily_df["gwl"], color="#2563EB", linewidth=1.0, alpha=0.9)
    ax.fill_between(daily_df["date"], daily_df["gwl"], daily_df["gwl"].min() - 0.5,
                     alpha=0.1, color="#2563EB")
    
    ax.set_title("Khagaul I — Daily Mean Groundwater Level (March 2023 – Dec 2025)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Groundwater Level (meters below ground)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.invert_yaxis()  # More negative = deeper, show deeper at bottom
    ax.grid(True, alpha=0.3)
    
    # Add annotation explaining the y-axis
    ax.annotate("↑ Shallower (rising)\n↓ Deeper (falling)", 
                xy=(0.01, 0.95), xycoords="axes fraction",
                fontsize=10, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    return save_fig(fig, "01_daily_timeseries")


def plot_actual_vs_predicted(test_df, results):
    """
    Plot 2: Actual vs Predicted GWL on the test set for all models.
    This is the most important plot — shows how well each model tracks reality.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    models_info = [
        ("Naive Baseline", results["models"]["naive"]["pred_test"], "#94A3B8"),
        ("Linear Regression", results["models"]["linear"]["pred_test"], "#F59E0B"),
        ("Random Forest", results["models"]["rf"]["pred_test"], "#10B981"),
        ("Gradient Boosting", results["models"]["gb"]["pred_test"], "#8B5CF6"),
    ]
    
    dates = test_df["date"].values
    actual = test_df["target"].values
    
    for ax, (name, pred, color) in zip(axes.flat, models_info):
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        
        ax.plot(dates, actual, color="#1E293B", linewidth=1.5, label="Actual", alpha=0.9)
        ax.plot(dates, pred, color=color, linewidth=1.5, label=f"Predicted", alpha=0.8, linestyle="--")
        ax.fill_between(dates, actual, pred, alpha=0.15, color=color)
        
        ax.set_title(f"{name}\nMAE={mae:.3f}m  |  RMSE={rmse:.3f}m", fontweight="bold")
        ax.set_ylabel("GWL (m)")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    
    fig.suptitle("7-Day-Ahead Forecast: Actual vs Predicted (Test Period)", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    return save_fig(fig, "02_actual_vs_predicted")


def plot_error_distribution(test_df, results):
    """
    Plot 3: Distribution of prediction errors for each model.
    Ideally, errors should be centered at 0 with a narrow spread.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    actual = test_df["target"].values
    models_info = [
        ("Linear Regression", results["models"]["linear"]["pred_test"], "#F59E0B"),
        ("Random Forest", results["models"]["rf"]["pred_test"], "#10B981"),
        ("Gradient Boosting", results["models"]["gb"]["pred_test"], "#8B5CF6"),
    ]
    
    for ax, (name, pred, color) in zip(axes, models_info):
        errors = actual - pred
        ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Perfect prediction")
        ax.axvline(errors.mean(), color="black", linestyle="-", linewidth=1.5, label=f"Mean error: {errors.mean():.3f}m")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Prediction Error (meters)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    
    fig.suptitle("Distribution of Prediction Errors (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    return save_fig(fig, "03_error_distribution")


def plot_feature_importance(results):
    """
    Plot 4: Feature importance from Random Forest.
    Shows which features the model relies on most for predictions.
    """
    imp_df = results["models"]["rf"]["importances"].head(15)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
    bars = ax.barh(range(len(imp_df)), imp_df["importance"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest: Feature Importance Ranking", fontweight="bold")
    
    # Add value labels
    for bar, val in zip(bars, imp_df["importance"].values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)
    
    plt.tight_layout()
    return save_fig(fig, "04_feature_importance")


def plot_model_comparison(results):
    """
    Plot 5: Bar chart comparing all models on MAE and RMSE.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = ["Naive\nBaseline", "Linear\nRegression", "Random\nForest", "Gradient\nBoosting"]
    keys = ["naive", "linear", "rf", "gb"]
    colors = ["#94A3B8", "#F59E0B", "#10B981", "#8B5CF6"]
    
    mae_vals = [results["models"][k]["mae_test"] for k in keys]
    rmse_vals = [results["models"][k]["rmse_test"] for k in keys]
    
    # MAE comparison
    bars1 = ax1.bar(names, mae_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_title("Mean Absolute Error (MAE)", fontweight="bold")
    ax1.set_ylabel("MAE (meters)")
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    
    # RMSE comparison
    bars2 = ax2.bar(names, rmse_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_title("Root Mean Squared Error (RMSE)", fontweight="bold")
    ax2.set_ylabel("RMSE (meters)")
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    
    fig.suptitle("Model Performance Comparison (Test Set)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    return save_fig(fig, "05_model_comparison")


def plot_forecast(daily_df, forecast_dates, forecast_values, model_name):
    """
    Plot 6: Final 7-day-ahead forecast from the best model.
    Shows recent history + the predicted future values.
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Show last 60 days of actual data
    recent = daily_df.dropna(subset=["gwl"]).tail(60)
    
    ax.plot(recent["date"], recent["gwl"], color="#1E293B", linewidth=2, 
            label="Observed GWL", marker="o", markersize=3)
    
    # Plot the forecast
    ax.plot(forecast_dates, forecast_values, color="#EF4444", linewidth=2.5,
            label=f"7-Day Forecast ({model_name})", marker="D", markersize=6,
            linestyle="--")
    
    # Connect last observation to first forecast
    last_obs_date = recent["date"].iloc[-1]
    last_obs_val = recent["gwl"].iloc[-1]
    ax.plot([last_obs_date, forecast_dates[0]], [last_obs_val, forecast_values[0]],
            color="#EF4444", linewidth=1.5, linestyle=":", alpha=0.6)
            
    # Shade region if range exists
    if len(forecast_dates) > 1:
        ax.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.1, color="#EF4444",
                   label="Forecast Period")
    else:
        ax.axvline(forecast_dates[0], alpha=0.3, color="#EF4444", linestyle=":",
                   label=f"Forecast Horizon (T+7)")
    
    ax.set_title(f"7-Day Groundwater Level Forecast — Khagaul I Station\n"
                 f"Using {model_name} (Prediction from Latest Observation: {last_obs_date.date()})", 
                 fontweight="bold", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Groundwater Level (meters below ground)")
    ax.legend(loc="upper right", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # Annotate forecast values
    for d, v in zip(forecast_dates, forecast_values):
        ax.annotate(f"{v:.2f}m", (d, v), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9, fontweight="bold",
                    color="#EF4444")
    
    plt.tight_layout()
    return save_fig(fig, "06_seven_day_forecast")


def plot_seasonal_pattern(daily_df):
    """
    Plot 7: Monthly boxplot showing seasonal groundwater patterns.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df = daily_df.copy()
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    
    # Create monthly boxplot
    month_order = range(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    data_by_month = [df[df["month"] == m]["gwl"].values for m in month_order]
    try:
        bp = ax.boxplot(data_by_month, tick_labels=month_names, patch_artist=True,
                        medianprops=dict(color="red", linewidth=2))
    except TypeError:
        bp = ax.boxplot(data_by_month, labels=month_names, patch_artist=True,
                        medianprops=dict(color="red", linewidth=2))
    
    # Color monsoon months differently
    monsoon = [5, 6, 7, 8]  # 0-indexed: Jun, Jul, Aug, Sep
    for i, box in enumerate(bp["boxes"]):
        if (i + 1) in [6, 7, 8, 9]:
            box.set_facecolor("#93C5FD")  # Monsoon - blue
        else:
            box.set_facecolor("#FDE68A")  # Non-monsoon - yellow
    
    ax.set_title("Seasonal Groundwater Level Pattern — Khagaul I", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Groundwater Level (meters below ground)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#93C5FD", label="Monsoon months"),
                       Patch(facecolor="#FDE68A", label="Non-monsoon months")]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    return save_fig(fig, "07_seasonal_pattern")
