# 📂 Imports and Setup
import sys
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pytz

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# 🛠 Project imports
import src.config as config
from src.citi_interface import (
    get_feature_store,
    load_model_from_local
)

# ==============================
# 🔑 Hopsworks Connection
# ==============================

print("🔌 Connecting to Hopsworks...")
feature_store = get_feature_store()

# ==============================
# 🚲 Fetch Citi Bike Data
# ==============================

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
print("Columns in ts_data:", ts_data.columns.tolist())


print(f"✅ Timeseries data shape after filtering: {ts_data.shape}")

if ts_data.empty:
    print("⚠️ No Citi Bike data available in Feature View for the selected period. Exiting.")
    sys.exit(0)

# ==============================
# ⚙️ Create Manual Features Here
# ==============================

print("🛠 Creating manual features...")

# Add manual time-based features
ts_data["hour"] = ts_data["hour_ts"].dt.hour
ts_data["hour_sin"] = np.sin(2 * np.pi * ts_data["hour"] / 24)
ts_data["hour_cos"] = np.cos(2 * np.pi * ts_data["hour"] / 24)
ts_data["day_of_week"] = ts_data["hour_ts"].dt.dayofweek
ts_data["month"] = ts_data["hour_ts"].dt.month
ts_data["day_of_year"] = ts_data["hour_ts"].dt.dayofyear

# Add holiday/weekend feature
from pandas.tseries.holiday import USFederalHolidayCalendar

calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=ts_data["hour_ts"].min(), end=ts_data["hour_ts"].max())
ts_data["is_holiday_or_weekend"] = ts_data["hour_ts"].dt.normalize().isin(holidays) | ts_data["hour_ts"].dt.dayofweek.isin([5, 6])

# Identify peak hours (7-9 AM and 4-7 PM typical)
ts_data["is_peak_hour"] = ts_data["hour"].isin([7,8,9,16,17,18,19])

# Rolling mean (3-hour)
ts_data["ride_count_roll3"] = ts_data["ride_count"].shift(1).rolling(3, min_periods=1).mean()

# Target: 8-hour ahead prediction
ts_data["target_ride_count"] = ts_data["ride_count"].shift(-8)

# ==============================
# ➡️ Generate Lag Features
# ==============================

print("🛠 Generating lag features...")

for lag in range(1, 679):
    ts_data[f"ride_count_lag_{lag}"] = ts_data["ride_count"].shift(lag)

# Drop rows with missing lag features
ts_data = ts_data.dropna().reset_index(drop=True)

print(f"✅ Final dataset shape after feature engineering: {ts_data.shape}")

# ==============================
# 📦 Load Model
# ==============================

print("📦 Loading trained model...")
model = load_model_from_local()
print("✅ Model loaded successfully.")

# ==============================
# 📈 Prepare Features
# ==============================

print("🛠 Preparing feature matrix X...")

non_feature_cols = ["hour_ts", "start_station_name", "time_of_day"]  # remove non-feature cols

trained_features = [
    "hour", "hour_sin", "hour_cos", "day_of_week", "is_holiday_or_weekend",
    "month", "is_peak_hour", "day_of_year", "ride_count_roll3"
] + [f"ride_count_lag_{i}" for i in range(1, 679)] + ["target_ride_count"]

X = ts_data.drop(columns=non_feature_cols, errors="ignore")

for col in trained_features:
    if col not in X.columns:
        print(f"⚠️ Missing column {col}. Filling with 0.")
        X[col] = 0

X = X[trained_features]

print(f"✅ Final feature matrix shape: {X.shape}")

# ==============================
# 🔮 Predict
# ==============================

print("🔮 Predicting ride counts...")
predictions = model.predict(X)

# ==============================
# 📊 Build Prediction Results
# ==============================

results = pd.DataFrame()
results["start_station_id"] = ts_data["start_station_id"].values
results["hour_ts_est"] = pd.to_datetime(ts_data["hour_ts"]).dt.tz_convert('America/New_York')
results["predicted_ride_count"] = predictions

print(f"✅ Predictions completed. Results shape: {results.shape}")

# ==============================
# 🏆 Show Top Stations
# ==============================

top_5_locations = results.sort_values("predicted_ride_count", ascending=False).head(5)

print("\n🏆 Top 5 Stations by Predicted Demand:")
print(top_5_locations[["start_station_id", "hour_ts_est", "predicted_ride_count"]])

