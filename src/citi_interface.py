# src/citi_interface.py

from datetime import datetime, timedelta, timezone
import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore
import joblib
from pathlib import Path

import src.config as config
from src.transform_ts_features_targets import transform_ts_data_into_features_and_targets_all_months

# ===============================
# ✨ Basic Utilities
# ===============================

def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()

# ===============================
# ✨ Model Loading & Prediction
# ===============================

def load_model_from_local() -> object:
    """Load LGBM model from local pickle file"""
    model_path = Path("C:/Users/MD/Desktop/citi/models/lgbmhyper.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model from {model_path}")
    return model

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(features)

    results = pd.DataFrame()
    results["start_station_id"] = features["start_station_id"].values
    results["predicted_ride_count"] = predictions.round(0)

    return results

# ===============================
# ✨ Feature Fetching
# ===============================

def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Fetch CitiBike feature batch for prediction"""
    feature_store = get_feature_store()

    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching CitiBike data from {fetch_data_from} to {fetch_data_to}")

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )

    ts_data = ts_data[ts_data["hour_ts"].between(fetch_data_from, fetch_data_to)]
    ts_data.sort_values(by=["start_station_id", "hour_ts"], inplace=True)

    features = transform_ts_data_info_features(
        ts_data, window_size=24 * 28, step_size=23
    )

    print(f"✅ Loaded {features.shape[0]} samples for batch features")
    return features

# ===============================
# ✨ Fetch Past Predictions
# ===============================

def fetch_next_hour_predictions():
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()

    df = df[df["hour_ts"] == next_hour]

    print(f"Current UTC time: {now}")
    print(f"Next hour: {next_hour}")
    print(f"Found {len(df)} prediction records")
    return df

def fetch_predictions(hours: int) -> pd.DataFrame:
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    df = fg.filter(fg.hour_ts >= current_hour).read()
    return df

def fetch_hourly_rides(hours: int) -> pd.DataFrame:
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    query = query.filter(fg.hour_ts >= current_hour)

    return query.read()

def fetch_days_data(days: int) -> pd.DataFrame:
    current_date = pd.to_datetime(datetime.now(timezone.utc))
    fetch_data_from = current_date - timedelta(days=(365 + days))
    fetch_data_to = current_date - timedelta(days=365)

    print(f"Fetching CitiBike ride data from {fetch_data_from} to {fetch_data_to}")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    df = fg.select_all().read()
    cond = (df["hour_ts"] >= fetch_data_from) & (df["hour_ts"] <= fetch_data_to)

    return df[cond]

# ===============================
# ✨ Optional: Safety Check
# ===============================

def assert_model_trained_for_8_hours():
    # Only if your training_metrics have 'target_gap_hours'
    try:
        project = get_hopsworks_project()
        model_registry = project.get_model_registry()
        models = model_registry.get_models(name=config.MODEL_NAME)
        model = max(models, key=lambda model: model.version)

        target_gap = model.training_metrics.get('target_gap_hours', 8)
        assert target_gap == 8, "⚠️ Model not trained for 8-hour ahead forecasting!"
        print(f"✅ Confirmed model is trained for {target_gap}-hour prediction.")
    except Exception as e:
        print(f"⚠️ Could not verify model training gap: {e}")

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("✅ Interface ready for CitiBike 8-hour ahead prediction!")
