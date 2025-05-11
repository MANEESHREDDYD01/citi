# 📍 src/frontend/app.py

import sys
from pathlib import Path

# Set parent directory dynamically for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
import streamlit as st
import pydeck as pdk

from src.inference import fetch_hourly_rides, fetch_predictions

# =============================
# 1. Streamlit Page Setup
# =============================

st.set_page_config(page_title="CitiBike MAE Dashboard", layout="wide")
st.title("🚲 Citi Bike MAE Visualization Dashboard")

# =============================
# 2. Sidebar Options
# =============================

past_hours = st.sidebar.slider(
    "Select number of past hours:",
    min_value=12,
    max_value=24 * 28,
    value=48,
    step=6,
)

st.sidebar.write("---")
st.sidebar.subheader("Visualization Settings")
selected_map_type = st.sidebar.radio(
    "Select Map Style:",
    ["Light", "Dark", "Satellite"],
    index=0,
)

map_styles = {
    "Light": "mapbox://styles/mapbox/light-v9",
    "Dark": "mapbox://styles/mapbox/dark-v9",
    "Satellite": "mapbox://styles/mapbox/satellite-streets-v9",
}

# =============================
# 3. Fetch Data from Feature Store
# =============================

st.write(f"📦 Fetching CitiBike data for the past **{past_hours} hours**...")

rides_df = fetch_hourly_rides(past_hours)
predictions_df = fetch_predictions(past_hours)

if rides_df.empty or predictions_df.empty:
    st.error("❌ No data fetched! Check Hopsworks connection or data availability.")
    st.stop()

# Merge rides with predictions
merged_df = pd.merge(
    rides_df,
    predictions_df,
    on=["start_station_id", "hour_ts"],
    how="inner"
)

# =============================
# 4. Compute MAE by Station
# =============================

if "predicted_demand" not in merged_df.columns or "ride_count_roll3" not in merged_df.columns:
    st.error("❌ Required columns missing! Check raw data and model output.")
    st.stop()

merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["ride_count_roll3"])

# Calculate MAE per station
mae_by_station = merged_df.groupby(["start_station_id", "start_station_name"])["absolute_error"].mean().reset_index()
mae_by_station.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Top 5 stations by MAE
top5_stations = mae_by_station.sort_values("MAE", ascending=False).head(5)

# Display Top 5
st.subheader("🏆 Top 5 Stations with Highest MAE")
st.dataframe(top5_stations)

# =============================
# 5. NYC Coordinates (Hardcoded for Top Stations)
# =============================

HARDCODED_COORDINATES = {
    72: (40.767272, -73.993929),
    79: (40.719115, -74.006666),
    82: (40.711174, -74.000165),
    116: (40.741776, -74.001497),
    83: (40.683826, -73.976323),
}

top5_stations["latitude"] = top5_stations["start_station_id"].map(lambda x: HARDCODED_COORDINATES.get(x, (None, None))[0])
top5_stations["longitude"] = top5_stations["start_station_id"].map(lambda x: HARDCODED_COORDINATES.get(x, (None, None))[1])

top5_stations = top5_stations.dropna(subset=["latitude", "longitude"])

# =============================
# 6. Plot Map with PyDeck
# =============================

st.subheader("🗺️ Interactive NYC MAE Map for Top Stations")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=top5_stations,
    get_position='[longitude, latitude]',
    get_color='[255, 0, 0, 160]',
    get_radius="MAE * 100",  # Bigger MAE -> Bigger Circle
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=40.730610,
    longitude=-73.935242,
    zoom=11,
    pitch=45,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style=map_styles[selected_map_type],
    tooltip={"text": "{start_station_name}\nMAE: {MAE:.2f}"}
)

st.pydeck_chart(deck)

# =============================
# 7. Summary Statistics
# =============================

st.write("---")
st.subheader("📊 Summary Statistics")
st.metric("Average MAE Across Top 5 Stations", value=f"{top5_stations['MAE'].mean():.2f}")
