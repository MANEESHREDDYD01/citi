# src/model_training_pipeline.py

import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.pipeline_util import get_pipeline
from src.citi_interface import (
    get_feature_store,
    get_hopsworks_project,
    load_metrics_from_registry,
    save_model_to_registry,
    save_metrics_to_registry,
)
from src.transform_ts_features_targets import transform_ts_data_into_features_and_targets

# ==============================
# 📦 Step 1: Fetch Data from Hopsworks
# ==============================

print("📦 Fetching Citi Bike data from feature store...")

# Connect to Hopsworks
feature_store = get_feature_store()

feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION,
)

ts_data = feature_view.get_batch_data()
from src.citi_interface import get_feature_store
import src.config as config

# Connect to Feature Store
feature_store = get_feature_store()

# Load your Feature View
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

# Fetch a small batch (1 day) to inspect schema
import pandas as pd
from datetime import datetime, timedelta

start_time = pd.Timestamp.now(tz="UTC") - timedelta(days=1)
end_time = pd.Timestamp.now(tz="UTC")

print(f"Fetching Feature View data from {start_time} to {end_time}...")

batch_data = feature_view.get_batch_data(
    start_time=start_time,
    end_time=end_time
)

# Print columns
print("\n✅ Columns in Feature View:")
print(list(batch_data.columns))

# ==============================
# 🔄 Step 2: Transform Timeseries Data
# ==============================

print("🔄 Preparing features and targets for training...")

features, targets = transform_ts_data_into_features_and_targets(ts_data)

# Handle no data case
if features is None or targets is None:
    print("❌ Feature creation failed. Exiting...")
    exit(1)

print(f"✅ Features shape: {features.shape}, Targets shape: {targets.shape}")

# ==============================
# 🚀 Step 3: Train LightGBM Pipeline
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
# 🔍 Step 4: Evaluate
# ==============================

print("🔍 Evaluating model...")

predictions = pipeline.predict(features)
test_mae = mean_absolute_error(targets, predictions)

metrics = load_metrics_from_registry()

print(f"📈 New model MAE: {test_mae:.4f}")
print(f"📉 Previous best MAE: {metrics['test_mae']:.4f}")

# ==============================
# 🛡 Step 5: Register if Better
# ==============================

if test_mae < metrics.get("test_mae", float("inf")):
    print(f"🏆 New model is better! Proceeding to register the model...")

    model_name = config.MODEL_NAME

    # Save model locally
    model_path = config.MODELS_DIR / f"{model_name}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    # Define input/output schema
    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    # Register model in Hopsworks
    model = model_registry.sklearn.create_model(
        name=model_name,
        metrics={"test_mae": test_mae},
        input_example=features.sample(),
        model_schema=model_schema,
        description="Citi Bike Demand Prediction Model"
    )
    model.save(str(model_path))

    print(f"✅ Model registered successfully!")

    # Save metrics separately
    save_metrics_to_registry(model_name, {"test_mae": test_mae})

else:
    print(f"⚠️ Skipping model registration. New model not better.")
