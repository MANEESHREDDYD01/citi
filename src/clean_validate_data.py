import pandas as pd
from pathlib import Path
from collections import Counter

def validate_and_save_citibike_data(
    input_dir: str = "C:/Users/MD/Desktop/citi/data/processed/monthly",
    output_dir: str = "C:/Users/MD/Desktop/citi/data/processed/validated"
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    station_counter = Counter()

    print("🔍 Scanning all files to find top 5 busiest start_station_ids...")

    for parquet_file in sorted(input_path.glob("rides_20*.parquet")):
        try:
            df = pd.read_parquet(parquet_file, columns=["start_station_id"])
            station_counter.update(df["start_station_id"].dropna().tolist())
        except Exception as e:
            print(f"⚠️ Skipped {parquet_file.name}: {e}")

    if not station_counter:
        print("🚫 No valid data found to compute top stations!")
        return

    top5_station_ids = [station_id for station_id, _ in station_counter.most_common(5)]
    print(f"✅ Top 5 Station IDs: {top5_station_ids}")

    for parquet_file in sorted(input_path.glob("rides_20*.parquet")):
        print(f"\n🔄 Processing: {parquet_file.name}")

        try:
            # Read only important columns first to filter down
            important_cols = [
                "start_station_id", "start_station_name", "start_lat", "start_lng",
                "end_station_id", "end_station_name", "end_lat", "end_lng",
                "started_at", "ended_at"
            ]
            df = pd.read_parquet(parquet_file, columns=important_cols)
        except Exception as e:
            print(f"⚠️ Failed to read {parquet_file.name}: {e}")
            continue

        # Small dataframe after reading only necessary columns
        df = df[df["start_station_id"].isin(top5_station_ids)]

        if df.empty:
            print(f"🚫 No top 5 stations found in {parquet_file.name}")
            continue

        for col in ("started_at", "ended_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = (
                    df[col]
                    .dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
                    .dt.tz_convert("America/New_York")
                )

        if {"started_at", "ended_at"}.issubset(df.columns):
            df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0

        if "duration_min" in df.columns:
            df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 240)]

        drop_cols = ["start_station_id", "end_station_id", "end_lat", "end_lng"]
        if set(drop_cols).issubset(df.columns):
            df = df.dropna(subset=drop_cols)

        if {"start_lat", "start_lng"}.issubset(df.columns):
            df = df[
                df["start_lat"].between(40.4774, 40.9176)
              & df["start_lng"].between(-74.2591, -73.7004)
            ]

        if "started_at" in df.columns:
            df = df.sort_values("started_at")

        output_file = output_path / parquet_file.name
        df.to_parquet(output_file, index=False)
        print(f"✅ Cleaned and saved: {output_file} ({len(df)} rows after filtering)")

if __name__ == "__main__":
    validate_and_save_citibike_data()
