# src/app.py

import sys
from pathlib import Path

# Set parent directory dynamically
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
import streamlit as st
import pydeck as pdk

# ======================================
# 1. Load the Citi Bike Feature Store Data
# ======================================

@st.cache_data(ttl=3600)
def load_citibike_data():
    # Example: replace this with your actual feature view loading if needed
    path = parent_dir / "data" / "feature_store" / "citibike_feature_view.parquet"
    if not path.exists():
        st.error(f"Data not found at {path}")
        st.stop()
    df = pd.read_parquet(path)
    return df

# ======================================
# 2. App Layout
# ======================================

st.title("🚴‍♂️ Citi Bike Top 5 Stations - MAE Dashboard (NYC Map)")

# Load Data
data = load_citibike_data()

# Check necessary columns
required_cols = {"start_station_id", "start_station_name", "target_ride_count", "ride_count_roll3"}
if not required_cols.issubset(data.columns):
    st.error(f"Missing columns! Found columns: {list(data.columns)}")
    st.stop()

# Calculate Error
data["mae"] = abs(data["target_ride_count"] - data["ride_count_roll3"])

# Group by Station ID and calculate MAE
mae_by_station = (
    data.groupby(["start_station_id", "start_station_name"])
    .agg(mae=("mae", "mean"))
    .reset_index()
    .sort_values("mae", ascending=True)
)

# Show Top 5 Stations
st.subheader("🏆 Top 5 Stations with Lowest MAE")
top5 = mae_by_station.head(5)
st.dataframe(top5)

# ======================================
# 3. Bonus: NYC Map using Pydeck
# ======================================

# (Optional) Randomly assign lat/lon for stations if missing
if "latitude" not in data.columns or "longitude" not in data.columns:
    import numpy as np
    np.random.seed(42)
    station_locations = {
        sid: (40.7 + np.random.rand() * 0.1, -74.0 + np.random.rand() * 0.1)
        for sid in data["start_station_id"].unique()
    }
    data["latitude"] = data["start_station_id"].map(lambda x: station_locations[x][0])
    data["longitude"] = data["start_station_id"].map(lambda x: station_locations[x][1])

# Merge Top 5 with Locations
top5_map = pd.merge(top5, data[["start_station_id", "latitude", "longitude"]].drop_duplicates(), on="start_station_id")

# Plot Map
st.subheader("🗺️ Top 5 Stations on NYC Map")

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=40.75,
            longitude=-74.0,
            zoom=11,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=top5_map,
                get_position=["longitude", "latitude"],
                get_color="[200, 30, 0, 160]",
                get_radius=300,
            ),
        ],
        tooltip={"text": "{start_station_name}\nMAE: {mae:.2f}"},
    )
)

st.success("✅ Dashboard loaded successfully!")
