# src/utils.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import holidays
from typing import Tuple
import pytz

# ========= BASIC HELPERS ==========

def fill_missing_hour_station(df: pd.DataFrame, time_col: str, station_cols: list, count_col: str) -> pd.DataFrame:
    """Fill missing (hour, station) combinations with 0 ride counts."""
    df[time_col] = pd.to_datetime(df[time_col])
    full_hours = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq="H")
    all_stations = df[station_cols].drop_duplicates()

    full_combinations = pd.DataFrame(
        [[hour] + list(station) for hour in full_hours for station in all_stations.values],
        columns=[time_col] + station_cols
    )

    merged = pd.merge(full_combinations, df, on=[time_col] + station_cols, how="left")
    merged[count_col] = merged[count_col].fillna(0).astype(int)
    return merged

def to_new_york(series: pd.Series) -> pd.Series:
    """Ensure datetime is localized to New York timezone."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("America/New_York")
    else:
        return series.dt.tz_localize("UTC", ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert("America/New_York")

def map_time_of_day(hour: int) -> str:
    """Categorize hour into parts of day."""
    if 0 <= hour <= 5:
        return "Night"
    elif 6 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 17:
        return "Afternoon"
    else:
        return "Evening"

# ========= FINAL FEATURE CREATION =========

def create_final_features(
    input_dir: str = r"C:/Users/MD/Desktop/citi/data/processed/feature_eng_all_id", 
    output_dir: str = r"C:/Users/MD/Desktop/citi/data/processed/final_features"
) -> None:
    """Load 8-hour prediction monthly files, merge into 2024 and 2025, add lags, and save."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Scanning {input_path} for 8-hour monthly parquet files...")

    dfs_2024 = []
    dfs_2025 = []

    for file in sorted(input_path.glob("citibike_features_targets_8hours_*.parquet")):
        print(f"🔄 Reading {file.name}")
        try:
            df = pd.read_parquet(file)
            df["hour_ts"] = pd.to_datetime(df["hour_ts"])
            year = df["hour_ts"].dt.year.iloc[0]
            if year == 2024:
                dfs_2024.append(df)
            elif year == 2025:
                dfs_2025.append(df)
            else:
                print(f"⚠️ Skipping unknown year in {file.name}")
        except Exception as e:
            print(f"⚠️ Failed to read {file.name}: {e}")
            continue

    # === Function to process a full year of data ===
    def process_year_data(dfs, year):
        if not dfs:
            print(f"❌ No data found for {year}. Skipping.")
            return

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.sort_values("hour_ts").reset_index(drop=True)

        print(f"\n📋 Merged {year} data shape before cleaning: {merged_df.shape}")

        # 🧹 Drop old lag columns (if they exist)
        lag_cols = [col for col in merged_df.columns if col.startswith("ride_count_lag_")]
        if lag_cols:
            print(f"⚡ Dropping {len(lag_cols)} old lag columns before creating new.")
            merged_df = merged_df.drop(columns=lag_cols)

        # === Now create new lag features
        lag_features = []
        for lag in range(1, 673):
            lag_col = merged_df["ride_count"].shift(lag)
            lag_features.append(lag_col.rename(f"ride_count_lag_{lag}"))
        lag_df = pd.concat(lag_features, axis=1)
        merged_df = pd.concat([merged_df, lag_df], axis=1)

        # 3-hour rolling mean
        merged_df["ride_count_roll3"] = merged_df["ride_count"].shift(1).rolling(3, min_periods=1).mean()

        # 8-hour ahead target
        if "target_ride_count" not in merged_df.columns:
            merged_df["target_ride_count"] = merged_df["ride_count"].shift(-8)

        # Drop missing rows caused by lagging, rolling
        merged_df = merged_df.dropna(subset=[f"ride_count_lag_{lag}" for lag in range(1, 673)] + ["ride_count_roll3", "target_ride_count"])
        merged_df = merged_df.reset_index(drop=True)

        print(f"✅ After dropping missing: {merged_df.shape}")

        # Time-based features
        merged_df["hour"] = merged_df["hour_ts"].dt.hour
        merged_df["hour_sin"] = np.sin(2 * np.pi * merged_df["hour"] / 24)
        merged_df["hour_cos"] = np.cos(2 * np.pi * merged_df["hour"] / 24)
        merged_df["day_of_week"] = merged_df["hour_ts"].dt.dayofweek
        merged_df["month"] = merged_df["hour_ts"].dt.month
        merged_df["day_of_year"] = merged_df["hour_ts"].dt.dayofyear
        merged_df["time_of_day"] = merged_df["hour"].apply(map_time_of_day)

        try:
            us_holidays = holidays.US(years=[year])
        except:
            us_holidays = holidays.US(years=[])

        is_weekend = (merged_df["day_of_week"] >= 5).astype(int)
        is_holiday = merged_df["hour_ts"].dt.date.isin(us_holidays).astype(int)

        merged_df["is_holiday_or_weekend"] = ((is_weekend + is_holiday) >= 1).astype(int)
        merged_df["is_peak_hour"] = merged_df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

        print(f"✅ Time-based features added for {year}. Final shape: {merged_df.shape}")

        # Save the processed dataset
        save_path = output_path / f"rides_citibike_final_{year}_with_lags.parquet"
        merged_df.to_parquet(save_path, index=False)
        print(f"✅ Saved final dataset for {year} at: {save_path}")

    # Process for 2024 and 2025
    process_year_data(dfs_2024, 2024)
    process_year_data(dfs_2025, 2025)

# ========= SPLITTING FUNCTION =========

def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str = "target_ride_count"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split CitiBike data based on hour_ts into training and testing sets.
    """
    if "hour_ts" not in df.columns:
        raise ValueError("Expected a column named 'hour_ts' for time-based splitting.")

    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce")

    if not cutoff_date.tzinfo:
        cutoff_date = pytz.timezone("America/New_York").localize(cutoff_date)

    train_data = df[df["hour_ts"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["hour_ts"] >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    print(f"✅ Split complete: {len(X_train)} train samples, {len(X_test)} test samples")

    return X_train, y_train, X_test, y_test

# ========= MAIN ==========

if __name__ == "__main__":
    create_final_features()
