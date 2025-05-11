# src/model_training_pipeline.py

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.citi_interface import (
    get_hopsworks_project,
    get_feature_store,
    load_metrics_from_registry,
    save_model_to_registry,
    save_metrics_to_registry
)
from src.pipeline_util import get_pipeline

# ==============================
# ✨ Connect to Hopsworks
# ==============================

feature_store = get_feature_store()

# ==============================
# 📦 Step 1: Fetch Citi Bike Data
# ==============================

print("📦 Fetching Citi Bike data from feature store...")

feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data()

# ==============================
# 🔄 Step 2: Prepare Features and Targets
# ==============================

print("🔄 Preparing features and targets for training...")

# Only drop truly unnecessary columns (KEEP 'hour_ts')
drop_cols = ["start_station_name", "time_of_day"]
X = ts_data.drop(columns=drop_cols, errors="ignore")

# Features: everything except the target
features = X.drop(columns=["target_ride_count"], errors="ignore")

# Target: label for training
targets = X["target_ride_count"]

print(f"✅ Features shape: {features.shape}, Targets shape: {targets.shape}")

# ==============================
# 🚀 Step 3: Train New Pipeline
# ==============================

print("🚀 Training new Citi Bike model pipeline...")

pipeline = get_pipeline(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipeline.fit(features, targets)

# ==============================
# 🔮 Step 4: Predict and Evaluate
# ==============================

print("🔍 Evaluating model...")

predictions = pipeline.predict(features)

# Compute MAE
test_mae = mean_absolute_error(targets, predictions)
metrics = {"test_mae": test_mae}

print(f"📈 New model MAE: {test_mae:.4f}")

# Load Previous Metrics
previous_metrics = load_metrics_from_registry()
print(f"📉 Previous best MAE: {previous_metrics['test_mae']:.4f}")

# ==============================
# 🛡 Step 5: Model Registration Decision
# ==============================

if test_mae < previous_metrics.get("test_mae", float("inf")):
    print(f"🏆 New model is better! Proceeding to register the model...")

    model_name = config.MODEL_NAME

    # Save and register
    save_model_to_registry(pipeline, model_name=model_name, metrics=metrics)

    # Update metrics
    save_metrics_to_registry(model_name=model_name, metrics=metrics)

    print(f"✅ Model registered successfully!")
else:
    print(f"⚠️ Skipping model registration because the new model is not better.")

