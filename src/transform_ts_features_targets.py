# src/transform_ts_features_targets.py (FINAL FOR HOPSWORKS)

import pandas as pd

# ==========================================
# 🚀 Transform directly from Hopsworks Feature View Data
# ==========================================

def transform_ts_data_into_features_and_targets(ts_data):
    """
    Create features and targets dynamically from Hopsworks Feature View data.

    Parameters
    ----------
    ts_data : pd.DataFrame
        The full batch of data fetched from Feature View.

    Returns
    -------
    features : pd.DataFrame
        DataFrame containing input features.
    targets : pd.Series
        Series containing target values (8-hour ahead ride_count).
    """

    if ts_data.empty:
        print("⚠️ No data provided for feature creation. Returning None.")
        return None, None

    ts_data = ts_data.copy()

    # ✅ Manual temporal features
    ts_data["hour"] = ts_data["hour_ts"].dt.hour
    ts_data["day_of_week"] = ts_data["hour_ts"].dt.dayofweek
    ts_data["month"] = ts_data["hour_ts"].dt.month

    # ✅ No need to create ride_count_roll3 again — already exists from Hopsworks!

    # ✅ Check if target column exists
    if "target_ride_count" not in ts_data.columns:
        print("❌ 'target_ride_count' column missing!")
        return None, None

    # ✅ Define final features: only use columns that surely exist
    feature_columns = [
        "start_station_id", "hour", "day_of_week", "month", "ride_count_roll3"
    ] + [col for col in ts_data.columns if col.startswith("ride_count_lag_")]

    # ✅ Prepare features and targets
    features = ts_data[feature_columns]
    targets = ts_data["target_ride_count"]

    print(f"✅ Features ready: {features.shape}, Targets ready: {targets.shape}")
    return features, targets

# ==========================================
# 🧪 Manual Testing Block
# ==========================================

if __name__ == "__main__":
    print("❗ This file is intended to be imported, not run directly.")
