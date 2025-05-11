import pandas as pd
import numpy as np
from pathlib import Path
import holidays

def fill_missing_hour_station(df, time_col, station_cols, count_col):
    df[time_col] = pd.to_datetime(df[time_col])
    full_hours = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq="H")
    all_stations = df[station_cols].drop_duplicates()

    full_combinations = pd.DataFrame([
        [hour] + list(station) for hour in full_hours for station in all_stations.values
    ], columns=[time_col] + station_cols)

    merged = pd.merge(full_combinations, df, on=[time_col] + station_cols, how="left")
    merged[count_col] = merged[count_col].fillna(0).astype(int)
    return merged

def to_new_york(series):
    if pd.api.types.is_datetime64tz_dtype(series):
        return series.dt.tz_convert("America/New_York")
    else:
        return series.dt.tz_localize("UTC", ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert("America/New_York")

def map_time_of_day(hour):
    if 0 <= hour <= 5:
        return "Night"
    elif 6 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 17:
        return "Afternoon"
    else:
        return "Evening"

def transform_to_timeseries(input_dir="C:/Users/MD/Desktop/citi/data/processed/validated", output_dir="C:/Users/MD/Desktop/citi/data/processed/timeseries"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in sorted(input_path.glob("rides_*.parquet")):
        print(f"\n🔁 Transforming: {file.name}")
        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"⚠️ Failed to read {file.name}: {e}")
            continue
        
        # Convert timestamps safely
        df["started_at"] = to_new_york(pd.to_datetime(df["started_at"], errors="coerce"))
        df["ended_at"] = to_new_york(pd.to_datetime(df["ended_at"], errors="coerce"))

        # Convert start_station_id to int (✅ added here)
        if "start_station_id" in df.columns:
            df["start_station_id"] = pd.to_numeric(df["start_station_id"], errors="coerce").dropna().astype(int)

        # Drop any rows with invalid or missing timestamps
        df = df.dropna(subset=["started_at", "ended_at"])

        # ✅ FINAL DST FIX
        df = df[~((df["started_at"].dt.month == 11) & 
                  (df["started_at"].dt.day == 3) & 
                  (df["started_at"].dt.hour == 1))]

        # Floor to hour
        df["hour_ts"] = df["started_at"].dt.floor("H")
        year_detected = df["hour_ts"].dt.year.max()

        # Create US holidays for detected year
        us_holidays = holidays.US(years=[year_detected])

        # Group
        grouped = df.groupby(["start_station_name", "start_station_id", "hour_ts"]).size().reset_index(name="ride_count")
        grouped = fill_missing_hour_station(grouped, "hour_ts", ["start_station_name", "start_station_id"], "ride_count")
        
        # === Feature Engineering ===
        grouped["hour"] = grouped["hour_ts"].dt.hour
        grouped["hour_sin"] = np.sin(2 * np.pi * grouped["hour"] / 24)
        grouped["hour_cos"] = np.cos(2 * np.pi * grouped["hour"] / 24)
        grouped["day_of_week"] = grouped["hour_ts"].dt.dayofweek
        
        is_weekend = (grouped["day_of_week"] >= 5).astype(int)
        is_holiday = pd.to_datetime(grouped["hour_ts"].dt.date).isin(us_holidays).astype(int)

        grouped["is_holiday_or_weekend"] = ((is_weekend + is_holiday) >= 1).astype(int)
        grouped["month"] = grouped["hour_ts"].dt.month
        grouped["is_peak_hour"] = grouped["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        grouped["day_of_year"] = grouped["hour_ts"].dt.dayofyear
        grouped["time_of_day"] = grouped["hour"].apply(map_time_of_day)
        
        # === Lag Features ===
        grouped = grouped.sort_values(["start_station_name", "hour_ts"])
        grouped["ride_count_lag_1"] = grouped.groupby("start_station_name")["ride_count"].shift(1)
        grouped["ride_count_roll3"] = grouped.groupby("start_station_name")["ride_count"].shift(1).rolling(3).mean()

        # === Remove rows where lag features are missing ===
        grouped = grouped.dropna(subset=["ride_count_lag_1", "ride_count_roll3"])

        # Save cleaned output
        output_file = output_path / file.name
        grouped.to_parquet(output_file, index=False)
        print(f"✅ Saved: {output_file} with {len(grouped)} rows (after cleaning)")

# Run the function
if __name__ == "__main__":
    transform_to_timeseries()
