# src/interface_pipeline.py (FOR CITI BIKE PROJECT)

from datetime import datetime, timedelta
import pandas as pd

import src.config as config
from src.citi_interface import (
    get_feature_store,
    get_model_predictions,
    load_model_from_local,  # ⬅️ your Citi models are local
)
from src.transform_ts_features_targets import transform_ts_data_into_features_and_targets_all_months

# ==============================
# 📅 Set Time Range to Fetch Data
# ==============================

# Get current UTC time
current_date = pd.Timestamp.now(tz="Etc/UTC")

# Fetch data from last 28 days
fetch_data_to = current_date - timedelta(hours=1)
fetch_data_from = current_date - timedelta(days=28)

print(f"📅 Fetching Citi Bike data from {fetch_data_from} to {fetch_data_to}...")

# Connect to Hopsworks Feature Store
feature_store = get_feature_store()

# Read time-series data from Feature View
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)

# Filter strictly within the desired window
ts_data = ts_data[ts_data.hour_ts.between(fetch_data_from, fetch_data_to)]
ts_data = ts_data.sort_values(["start_station_id", "hour_ts"]).reset_index(drop=True)

# No need to tz_localize — already handled earlier if needed

print(f"✅ Time series data shape after filtering: {ts_data.shape}")

# ==============================
# ✨ Transform Data into Features
# ==============================

# (If you have a real transform_ts_data_info_features function, use it)
# For now directly use Citi function
features, _ = transform_ts_data_into_features_and_targets_all_months(
    ts_data,
    window_size=24*28,
    step_size=23
)

print(f"✅ Features created: {features.shape}")

# ==============================
# 🚀 Load Model
# ==============================

model = load_model_from_local()
print("✅ Loaded model from local storage.")

# ==============================
# 🔮 Predict
# ==============================

predictions = get_model_predictions(model, features)

# Assign prediction timestamp
predictions["hour_ts"] = current_date.ceil("h")
print(f"✅ Predictions completed. Shape: {predictions.shape}")

print(predictions.head())

# ==============================
# 💾 Save Predictions to Feature Store
# ==============================

feature_group = feature_store.get_or_create_feature_group(
    name=config.FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="Predictions from Citi Bike LGBM Model",
    primary_key=["start_station_id", "hour_ts"],
    event_time="hour_ts",
)

feature_group.insert(predictions, write_options={"wait_for_job": False})

print("✅ Predictions inserted into Feature Store successfully!")
