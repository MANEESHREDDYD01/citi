# src/transform_ts_features_targets.py (FINAL FOR HOPSWORKS DATA)

import pandas as pd
from pathlib import Path

# ==========================================
# 1️⃣ Load FINAL full-year processed data
# ==========================================

def transform_ts_data_into_features_and_targets_all_months(
    input_dir="../data/processed/final_features"
):
    input_path = Path(input_dir)

    path_2024 = input_path / "rides_citibike_final_2024_with_lags.parquet"
    path_2025 = input_path / "rides_citibike_final_2025_with_lags.parquet"

    if not path_2024.exists() or not path_2025.exists():
        print("❌ One or both final feature files are missing!")
        return None, None

    print(f"🔵 Loading {path_2024}")
    df_2024 = pd.read_parquet(path_2024)

    print(f"🔵 Loading {path_2025}")
    df_2025 = pd.read_parquet(path_2025)

    # Combine both years
    df = pd.concat([df_2024, df_2025], axis=0, ignore_index=True)
    print(f"✅ Combined data shape: {df.shape}")

    # Check if target exists
    target_col = "target_ride_count"
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return None, None

    features = df.drop(columns=[target_col])
    targets = df[target_col]

    print(f"✅ Features shape: {features.shape}, Targets shape: {targets.shape}")
    return features, targets

# ==========================================
# 2️⃣ In-memory feature creation for prediction
# ==========================================

def transform_ts_data_into_features_and_targets(ts_data):
    """
    Create features and targets dynamically from Hopsworks Feature View data.
    """

    if ts_data.empty:
        print("⚠️ No data provided for feature creation. Returning None.")
        return None, None

    ts_data = ts_data.copy()

    # Manual temporal features (these are safe because hour_ts exists)
    ts_data["hour"] = ts_data["hour_ts"].dt.hour
    ts_data["day_of_week"] = ts_data["hour_ts"].dt.dayofweek
    ts_data["month"] = ts_data["hour_ts"].dt.month

    # No need to create ride_count_roll3 here — it already exists in Hopsworks!!

    if "target_ride_count" not in ts_data.columns:
        print("❌ target_ride_count column missing!")
        return None, None

    # Final feature columns (safe columns that exist)
    feature_columns = [
        "start_station_id", "hour", "day_of_week", "month", "ride_count_roll3"
    ] + [col for col in ts_data.columns if col.startswith("ride_count_lag_")]

    features = ts_data[feature_columns]
    targets = ts_data["target_ride_count"]

    return features, targets

# ==========================================
# 3️⃣ Main Execution for Manual Testing
# ==========================================

if __name__ == "__main__":
    transform_ts_data_into_features_and_targets_all_months()
