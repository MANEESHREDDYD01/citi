# 📂 Imports and Setup
import sys
import os

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# 🛠 Project imports
import src.config as config
from src.citi_interface import (
    get_feature_store,
    load_model_from_local
)

from datetime import datetime, timedelta
import pandas as pd
import pytz
import numpy as np  # ✅ Needed for sin/cos

# ==============================
# 🔑 Hopsworks Connection
# ==============================

# Connect to Feature Store
feature_store = get_feature_store()

# ==============================
# 🚲 Fetch Citi Bike Data (January to March 2025)
# ==============================

# Fixed Start and End
fetch_data_from = pd.Timestamp("2025-01-01 00:00:00", tz="Etc/UTC")
fetch_data_to = pd.Timestamp("2025-03-31 23:59:59", tz="Etc/UTC")

print(f"📅 Fetching Citi Bike data from {fetch_data_from} to {fetch_data_to}...")

feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

# Fetch batch data
ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)

# Filter exact range
ts_data = ts_data[ts_data["hour_ts"].between(fetch_data_from, fetch_data_to)]

# Sort and Reset
ts_data = ts_data.sort_values(["start_station_id", "hour_ts"]).reset_index(drop=True)

print(f"✅ Timeseries data shape after filtering: {ts_data.shape}")

# 🛑 Early Exit if No Data
if ts_data.empty:
    print("⚠️ No Citi Bike data available in Feature View for the selected period. Exiting.")
    sys.exit(0)

# ==============================
# 📦 Load Trained Model
# ==============================

model = load_model_from_local()
print("✅ Model loaded successfully.")

# ==============================
# ⚙️ Prepare Features for Prediction
# ==============================

print("🛠 Creating manual features...")

# Manual time-based features
ts_data["hour_sin"] = np.sin(2 * np.pi * ts_data["hour"] / 24)
ts_data["hour_cos"] = np.cos(2 * np.pi * ts_data["hour"] / 24)

# Drop non-feature columns
non_feature_cols = ["hour_ts", "start_station_name", "time_of_day"]
X = ts_data.drop(columns=non_feature_cols, errors="ignore")

# Define trained feature columns (what the model expects)
trained_features = [
    "hour", "hour_sin", "hour_cos", "day_of_week", "is_holiday_or_weekend",
    "month", "is_peak_hour", "day_of_year", "ride_count_roll3"
] + [f"ride_count_lag_{i}" for i in range(1, 679)] + ["target_ride_count"]

# Fill missing features if any
for col in trained_features:
    if col not in X.columns:
        print(f"⚠️ Missing column: {col}. Filling with 0.")
        X[col] = 0

# Reorder features properly
X = X[trained_features]

print(f"✅ Final feature shape for prediction: {X.shape}")

# ==============================
# 🔮 Predict Ride Counts
# ==============================

# Predict
predictions = model.predict(X)

# ==============================
# 🛠 Build Prediction Results
# ==============================

results = pd.DataFrame()
results["start_station_id"] = ts_data["start_station_id"].values

# ✅ Correct UTC ➔ EST timezone conversion
results["hour_ts_est"] = pd.to_datetime(ts_data["hour_ts"]).dt.tz_convert('America/New_York')

# Predicted ride counts
results["predicted_ride_count"] = predictions

print(f"✅ Predictions completed. Shape: {results.shape}")

# ==============================
# 🚴 Top 5 Stations by Predicted Demand
# ==============================

top_5_locations = results.sort_values("predicted_ride_count", ascending=False).head(5)

print("\n🏆 Top 5 Stations by Predicted Demand:")
print(top_5_locations[["start_station_id", "hour_ts_est", "predicted_ride_count"]])

