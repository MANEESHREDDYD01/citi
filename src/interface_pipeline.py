# ============================
# 🚴 Citi Bike Interface Pipeline
# ============================

# 📦 Imports
import pandas as pd
from datetime import datetime, timedelta
import pytz

import src.config as config
from src.citi_interface import (
    get_hopsworks_project,
    get_feature_store,
    get_feature_view,
    load_model_from_local,
    get_model_predictions,
)
from src.transform_ts_features_targets import transform_ts_data_into_features_and_targets

# ============================
# 📂 Connect to Hopsworks
# ============================
print("🔌 Connecting to Hopsworks...")
project = get_hopsworks_project()
fs = get_feature_store()

# ============================
# 📅 Define Time Range
# ============================
current_time_utc = pd.Timestamp.now(tz="UTC")

# For now, just fetch entire 2024–early 2025 data — or you can adjust window
print(f"📅 Fetching Citi Bike data from 2025-01-01 to {current_time_utc}...")

# ============================
# 🛒 Fetch Time Series Data
# ============================
fv = get_feature_view(name="citibike_hourly_data_v2", version=1)

try:
    ts_data = fv.get_batch_data()
except Exception as e:
    print(f"❌ Failed to fetch batch data: {e}")
    exit(1)

print(f"✅ Timeseries data shape after filtering: {ts_data.shape}")

if ts_data.shape[0] == 0:
    print("⚠️ No recent Citi Bike data available in Feature View. Exiting gracefully.")
    exit(0)

# ============================
# 🛠 Transform into Features
# ============================
features_targets = transform_ts_data_into_features_and_targets(ts_data)

if features_targets is None:
    print("⚠️ No features generated. Exiting gracefully...")
    exit(0)

features, _ = features_targets

print(f"✅ Feature shape for prediction: {features.shape}")

# ============================
# 🤖 Load Model
# ============================
model = load_model_from_local()
print("✅ Loaded model successfully.")

# ============================
# 🔮 Predict Ride Counts
# ============================
X = features.drop(columns=["hour_ts"], errors="ignore")

print(f"✅ Final feature shape for prediction: {X.shape}")

predictions = model.predict(X)

# ============================
# 🛠 Build Prediction Results
# ============================
results = pd.DataFrame()
results["start_station_id"] = features["start_station_id"].values

# Convert UTC hour_ts to EST
est = pytz.timezone('America/New_York')
results["hour_ts_est"] = pd.to_datetime(features["hour_ts"]).dt.tz_convert(est)

results["predicted_ride_count"] = predictions

print(f"✅ Predictions completed. Shape: {results.shape}")

# ============================
# 🏆 Display Top 10 Predictions
# ============================
top_predictions = results.sort_values(by="predicted_ride_count", ascending=False).head(10)
print("\n🏆 Top 10 Stations by Predicted Demand:")
print(top_predictions)

# (Optionally: Save results to Hopsworks or locally)
