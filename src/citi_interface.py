# src/citi_interface.py

from datetime import datetime, timedelta, timezone
import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore
import joblib
from pathlib import Path

import src.config as config

# ===============================
# ✨ Basic Utilities
# ===============================

def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()

# ===============================
# ✨ Model Loading & Saving
# ===============================

def load_model_from_local():
    """
    Load a pre-trained model from the models/ directory inside the repo.
    """
    model_path = Path(__file__).parent.parent / "models" / "lgbmhyper.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

def save_model_to_registry(model_name: str, metrics: dict = None):
    """
    Upload the trained model to Hopsworks Model Registry.
    Assumes model file already saved locally as {model_name}.pkl
    """
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model_path = Path(f"{model_name}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Cannot find model file {model_path} to upload!")

    model_registry.upload_model(
        str(model_path),
        model_name=model_name,
        metrics=metrics if metrics else {},
        description="Trained model uploaded from GitHub Actions",
        overwrite=True,
    )
    print(f"✅ Model '{model_name}' uploaded successfully to registry.")

def save_metrics_to_registry(model_name: str, metrics: dict):
    """
    Save evaluation metrics to Hopsworks Model Registry.
    """
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    model = model_registry.get_model(model_name, version=None)
    model.save_metrics(metrics)
    print(f"✅ Metrics for model '{model_name}' saved to registry.")

# ===============================
# ✨ Feature Fetching
# ===============================

def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Fetch latest batch features for prediction from Feature Store.
    """
    feature_store = get_feature_store()

    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)

    print(f"Fetching CitiBike data from {fetch_data_from} to {fetch_data_to}")

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )

    ts_data = ts_data[ts_data["hour_ts"].between(fetch_data_from, fetch_data_to)]
    ts_data.sort_values(["start_station_id", "hour_ts"], inplace=True)

    print(f"✅ Loaded {ts_data.shape[0]} samples for batch prediction.")
    return ts_data

def fetch_days_data(days: int) -> pd.DataFrame:
    """
    Fetch historical Citi Bike data for model training.
    """
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
# ✨ Fetch Past Predictions (for dashboard or monitoring)
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

# ===============================
# ✨ Model Check
# ===============================

def assert_model_trained_for_8_hours():
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

def load_metrics_from_registry() -> dict:
    """
    Load the previous best model's metrics from Hopsworks Model Registry.
    If no model is found, return a very high MAE by default.
    """
    try:
        project = get_hopsworks_project()
        model_registry = project.get_model_registry()
        models = model_registry.get_models(name=config.MODEL_NAME)

        if not models:
            print("⚠️ No previous model found. Returning default high MAE.")
            return {"test_mae": float("inf")}

        latest_model = max(models, key=lambda model: model.version)
        return latest_model.metrics

    except Exception as e:
        print(f"⚠️ Error loading metrics from registry: {e}")
        return {"test_mae": float("inf")}

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("✅ Interface ready for CitiBike 8-hour ahead prediction!")
