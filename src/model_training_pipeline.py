# src/model_training_pipeline.py (FOR CITI BIKE PROJECT)

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.transform_ts_features_targets import transform_ts_data_into_features_and_targets_all_months
from src.citi_interface import (
    get_feature_store,
    load_model_from_local,
    save_model_to_registry,
    save_metrics_to_registry
)

# Connect to Hopsworks
feature_store = get_feature_store()
from src.pipeline_util import get_pipeline

# ==============================
# 📅 Step 1: Fetch Citi Bike Data
# ==============================

print("📦 Fetching Citi Bike data from feature store...")
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME,
    version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data()

# ==============================
# 🔄 Step 2: Transform into Features and Targets
# ==============================

print("🔄 Transforming timeseries data into features and targets...")

features, targets = transform_ts_data_into_features_and_targets_all_months()

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
