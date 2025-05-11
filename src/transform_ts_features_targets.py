# 04_transform_ts_data_into_features_and_targets_all_months_with_id_with_name.py

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1️⃣ Save monthly parquet feature datasets
# ==========================================
def transform_ts_data_into_features_and_targets_all_months(
    input_dir="../data/processed/timeseries", 
    output_dir="../data/processed/feature_eng_all_id"
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # List all months to process separately
    months = [
        (2024, 1), (2024, 2), (2024, 3), (2024, 4),
        (2024, 5), (2024, 6), (2024, 7), (2024, 8),
        (2024, 9), (2024, 10), (2024, 11), (2024, 12),
        (2025, 1), (2025, 2), (2025, 3)
    ]

    for year, month in months:
        file_path = input_path / f"rides_{year}_{month:02}.parquet"

        if not file_path.exists():
            print(f"⚠️ Skipping {year}-{month:02} (File not found)")
            continue

        print(f"\n🔵 Loading file: {file_path}")
        df = pd.read_parquet(file_path)

        # ✅ Focus only on top 5 busiest stations for this month
        top_station_ids = (
            df.groupby("start_station_id")["ride_count"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        # Find corresponding station names
        id_to_name = df[df["start_station_id"].isin(top_station_ids)].groupby("start_station_id")["start_station_name"].first()

        print("✅ Using top 5 stations for", f"{year}-{month:02}:")
        for station_id in top_station_ids:
            station_name = id_to_name.get(station_id, "Unknown")
            print(f"ID: {station_id} → Station Name: {station_name}")

        # Filter only top stations
        df = df[df["start_station_id"].isin(top_station_ids)].copy()

        # Sort by time
        df = df.sort_values("hour_ts").reset_index(drop=True)

        # 3-hour rolling mean (shifted globally)
        df["ride_count_roll3"] = df["ride_count"].shift(1).rolling(3, min_periods=1).mean()

        # Target variable (8 hours ahead ride_count)
        df["target_ride_count"] = df["ride_count"].shift(-8)

        # Drop missing rows (caused by rolling and shifting)
        df = df.dropna(subset=["ride_count_roll3", "target_ride_count"])

        print(f"✅ Final shape for {year}-{month:02}: {df.shape}")

        # Save final dataset for this month
        final_save_path = output_path / f"citibike_features_targets_8hours_{year}_{month:02}.parquet"
        df.to_parquet(final_save_path, index=False)

        print(f"✅ Saved monthly feature dataset at: {final_save_path}")

# ==========================================
# 2️⃣ In-memory feature creation for inference
# ==========================================
def transform_ts_data_into_features_and_targets(ts_data):
    """
    Create features and targets dynamically from given CitiBike timeseries data.

    Used for inference pipelines (interface_pipeline.py).
    """

    if ts_data.empty:
        print("⚠️ No data provided for feature creation. Returning None.")
        return None, None

    # Copy
    ts_data = ts_data.copy()

    # Create manual temporal features
    ts_data["hour"] = ts_data["hour_ts"].dt.hour
    ts_data["day_of_week"] = ts_data["hour_ts"].dt.dayofweek
    ts_data["month"] = ts_data["hour_ts"].dt.month

    # Rolling feature
    ts_data["ride_count_roll3"] = ts_data["ride_count"].shift(1).rolling(3, min_periods=1).mean()

    # Target feature (8 hours ahead ride_count)
    ts_data["target_ride_count"] = ts_data["ride_count"].shift(-8)

    # Drop rows with missing features/target
    ts_data = ts_data.dropna(subset=["ride_count_roll3", "target_ride_count"])

    # Final feature columns
    feature_columns = [
        "start_station_id", "hour", "day_of_week", "month", "ride_count_roll3"
    ]

    features = ts_data[feature_columns]
    targets = ts_data["target_ride_count"]

    return features, targets

# ==========================================
# 3️⃣ Main Execution
# ==========================================
if __name__ == "__main__":
    transform_ts_data_into_features_and_targets_all_months()
