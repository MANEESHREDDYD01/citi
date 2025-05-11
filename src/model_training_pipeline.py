# src/model_training_pipeline.py (FOR CITI BIKE PROJECT)

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path

import src.config as config
from src.citi_interface import (
    get_feature_store,
    load_model_from_local,
    save_model_to_registry,
    save_metrics_to_registry,
    get_hopsworks_project,
    load_metrics_from_registry
)
from src.pipeline_util import get_pipeline

# ==============================
# 📅 Step 1: Load Final Citi Bike Data
# ==============================

print("📦 Loading final processed Citi Bike feature data...")

# Final feature files
final_features_path = Path("C:/Users/MD/Desktop/citi/data/processed/final_features")

df_2024 = pd.read_parquet(final_features_path / "rides_citibike_final_2024_with_lags.parquet")
df_2025 = pd.read_parquet(final_features_path / "rides_citibike_final_2025_with_lags.parquet")

# Combine
df = pd.concat([df_2024, df_2025], axis=0).reset_index(drop=True)

print(f"✅ Combined dataset shape: {df.shape}")

# ==============================
# 🧹 Step 2: Prepare Features and Targets
# ==============================

non_feature_cols = ["hour_ts", "start_station_name", "time_of_day", "start_station_id"]

features = df.drop(columns=non_feature_cols + ["target_ride_count"], errors="ignore")
targets = df["target_ride_count"]

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
