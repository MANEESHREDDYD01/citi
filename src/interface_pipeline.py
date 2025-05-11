# ============================
# 🚴 Citi Bike Interface Pipeline (FINAL)
# ============================

# 📦 Imports
import pandas as pd
import pytz
from datetime import datetime

import src.config as config
from src.citi_interface import (
    get_hopsworks_project,
    get_feature_store,
    load_model_from_local,
)
from src.utils import to_new_york  # ✅ you already had this
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

print(f"📅 Fetching Citi Bike data from 2025-01-01 to {current_time_utc}...")

# ============================
# 🛒 Fetch Raw Feature Group Data
# ============================
try:
    fg = fs.get_feature_group(name="citibike_hourly_data_v2", version=1)
    query = fg.select_all()
    ts_data = query.read()
except Exception as e:
    print(f"❌ Failed to fetch Feature Group data: {e}")
    exit(1)

print(f"✅ Timeseries data shape after fetching: {ts_data.shape}")

if ts_data.shape[0] == 0:
    print("⚠️ No Citi Bike data available in Feature Group. Exiting gracefully.")
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
print("✅ Model loaded successfully.")

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

# (Optional: Save back to Feature Group or storage if needed)
