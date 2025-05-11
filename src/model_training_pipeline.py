# src/model_training_pipeline.py (FOR CITI BIKE PROJECT)

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.transform_ts_features_targets import transform_ts_data_into_features_and_targets_all_months
from src.citi_interface import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry
)
from src.pipeline_util import get_pipeline

# ==============================
# 📅 Step 1: Fetch Citi Bike Data
# ==============================

print("📦 Fetching Citi Bike data from feature store...")
ts_data = fetch_days_data(180)  # Fetch last 180 days data

# ==============================
# 🔄 Step 2: Transform into Features and Targets
# ==============================

print("🔄 Transforming timeseries data into features and targets...")

features, targets = transform_ts_data_into_features_and_targets_all_months(
    ts_data, window_size=24 * 28, step_size=23
)

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

predictions = pipeline.predict(features)

test_mae = mean_absolute_error(targets, predictions)
metric = load_metrics_from_registry()

print(f"📈 New model MAE: {test_mae:.4f}")
print(f"📉 Previous best MAE: {metric['test_mae']:.4f}")

# ==============================
# 🛡 Step 5: Model Registration Decision
# ==============================

if test_mae < metric.get("test_mae"):
    print(f"🏆 New model is better! Proceeding to register the model...")
    
    model_path = config.MODELS_DIR / "lgbm_citibike_model.pkl"
    joblib.dump(pipeline, model_path)

    # Define input and output schemas
    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.sklearn.create_model(
        name="citibike_demand_predictor",
        metrics={"test_mae": test_mae},
        input_example=features.sample(),
        model_schema=model_schema,
    )
    model.save(model_path)

    print(f"✅ Model registered successfully in Hopsworks Model Registry!")
else:
    print(f"⚠️ Skipping model registration because the new model is not better.")
