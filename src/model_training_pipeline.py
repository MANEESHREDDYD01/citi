# src/model_training_pipeline.py

import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.citi_interface import (
    get_feature_store,
    load_model_from_local,
    save_model_to_registry,
    save_metrics_to_registry,
    load_metrics_from_registry,
    get_hopsworks_project
)
from src.pipeline_util import get_pipeline

# ==============================
# 📦 Step 1: Fetch Citi Bike Data
# ==============================

print("📦 Fetching Citi Bike data from feature store...")

feature_store = get_feature_store()
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

# Load full timeseries dataset (already processed and feature engineered)
ts_data = feature_view.get_batch_data()

# ==============================
# 🔄 Step 2: Prepare Features and Targets
# ==============================

print("🔄 Preparing features and targets for training...")

# Drop unnecessary columns
drop_cols = ["hour_ts", "start_station_name", "time_of_day"]
X = ts_data.drop(columns=drop_cols, errors="ignore")

# Features = everything except the final label
features = X.drop(columns=["target_ride_count"], errors="ignore")
# Targets = the label we want to predict
targets = X["target_ride_count"]

print(f"✅ Features shape: {features.shape}, Targets shape: {targets.shape}")

# ==============================
# 🚀 Step 3: Train New Model Pipeline
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

predictions = pipeline.predict(features)
test_mae = mean_absolute_error(targets, predictions)

print(f"📈 New model MAE: {test_mae:.4f}")

# Load previous best model metrics
previous_metrics = load_metrics_from_registry()
print(f"📉 Previous best MAE: {previous_metrics['test_mae']:.4f}")

# ==============================
# 🛡 Step 5: Model Registration Decision
# ==============================

if test_mae < previous_metrics.get("test_mae", float("inf")):
    print(f"🏆 New model is better! Proceeding to register the model...")

    # Save the model locally
    model_save_path = config.MODELS_DIR / "lgbm_citibike_model.pkl"
    joblib.dump(pipeline, model_save_path)

    # Prepare input/output schema
    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    # Register model to Hopsworks
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.sklearn.create_model(
        name=config.MODEL_NAME,
        metrics={"test_mae": test_mae},
        input_example=features.sample(),
        model_schema=model_schema,
    )
    model.save(str(model_save_path))

    # Save updated metrics
    save_metrics_to_registry(model_name=config.MODEL_NAME, metrics={"test_mae": test_mae})

    print(f"✅ Model registered successfully in Hopsworks Model Registry!")

else:
    print(f"⚠️ Skipping model registration because the new model is not better.")

