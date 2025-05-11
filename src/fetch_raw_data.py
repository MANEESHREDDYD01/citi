# src/raw.py

import requests
import zipfile
import pandas as pd
import re
from pathlib import Path

# === Part 1: Download and Unzip ===

def download_and_unzip_citibike_files(file_names: list, raw_dir="C:/Users/MD/Desktop/citi/data/raw", unzip_dir="C:/Users/MD/Desktop/citi/data/unzipped"):
    """
    Download Citi Bike tripdata zip files and extract them to a folder.
    """
    base_url = "https://s3.amazonaws.com/tripdata"

    raw_path = Path(raw_dir)
    unzip_path = Path(unzip_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    unzip_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        url = f"{base_url}/{file_name}"
        zip_file_path = raw_path / file_name

        print(f"🔵 Downloading: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(zip_file_path, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded: {zip_file_path}")

            # === Unzip
            print(f"🔵 Unzipping: {zip_file_path}")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            print(f"✅ Unzipped into: {unzip_path}")

        except Exception as e:
            print(f"❌ Failed to process {file_name}. Error: {e}")

# === Part 2: Chunk Read and Save Monthly Parquet ===

def save_monthly_files_with_chunks(unzip_dir="C:/Users/MD/Desktop/citi/data/unzipped", output_dir="C:/Users/MD/Desktop/citi/data/processed/monthly", chunk_size=500_000):
    """
    Read extracted CSV files, group by month, and save each month as a single .parquet file.
    """
    unzip_path = Path(unzip_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    monthly_files = {}

    for file in unzip_path.rglob("*.csv"):
        if file.name.startswith("._"):  # Skip Mac junk files
            continue
        match = re.search(r"(20\d{2})(\d{2})", file.name)
        if match:
            year, month = match.groups()
            key = f"{year}_{month}"
            monthly_files.setdefault(key, []).append(file)

    for key, file_list in monthly_files.items():
        print(f"\n📦 Processing month: {key} with {len(file_list)} files")
        monthly_data = []

        for file in file_list:
            try:
                print(f"  🔄 Reading: {file.name}")
                chunks = pd.read_csv(file, low_memory=False, dtype={"start_station_id": str, "end_station_id": str}, chunksize=chunk_size)
                for chunk in chunks:
                    monthly_data.append(chunk)
            except Exception as e:
                print(f"⚠️ Skipped {file.name} due to error: {e}")

        if monthly_data:
            combined = pd.concat(monthly_data, ignore_index=True)
            output_file = output_path / f"rides_{key}.parquet"
            combined.to_parquet(output_file, index=False)
            print(f"✅ Saved: {output_file} with {len(combined)} rows")
        else:
            print(f"🚫 No valid data for {key}")

# === Part 3: Run All ===

if __name__ == "__main__":
    # Define which files to download
    file_names = [
        "202401-citibike-tripdata.csv.zip",
        "202402-citibike-tripdata.csv.zip",
        "202403-citibike-tripdata.csv.zip",
        "202404-citibike-tripdata.csv.zip",
        "202405-citibike-tripdata.zip",
        "202406-citibike-tripdata.zip",
        "202407-citibike-tripdata.zip",
        "202408-citibike-tripdata.zip",
        "202409-citibike-tripdata.zip",
        "202410-citibike-tripdata.zip",
        "202411-citibike-tripdata.zip",
        "202412-citibike-tripdata.zip",
        "202501-citibike-tripdata.zip",
        "202502-citibike-tripdata.zip",
        "202503-citibike-tripdata.csv.zip"
    ]

    # === First download and unzip
    download_and_unzip_citibike_files(file_names)

    # === Then group by month and save clean parquet
    save_monthly_files_with_chunks()
