# src/frontend.py (FOR CITI BIKE PROJECT)

import sys
from pathlib import Path

# Set parent directory dynamically
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.citi_interface import fetch_hourly_rides, fetch_predictions  # Updated to citi_interface

st.title("🚴‍♂️ Citi Bike Demand Prediction - MAE by Hour")

# Sidebar for user input
st.sidebar.header("Settings")

# Set path to lookup CSV (adjust accordingly for Citi Bike station info)
lookup_path = parent_dir / "data" / "citibike-station-lookup.csv"  

# Load station lookup table
lookup_df = pd.read_csv(lookup_path)

# Ensure correct column names
lookup_df.rename(columns={"station_id": "start_station_id", "station_name": "start_station_name"}, inplace=True)

# Dropdown to select station by name
selected_station = st.sidebar.selectbox("Select Start Station", lookup_df["start_station_name"].unique())

# Get the corresponding station ID
selected_station_id = lookup_df.loc[lookup_df["start_station_name"] == selected_station, "start_station_id"].values[0]

# Slider to select past hours
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=12,
    step=1,
)

# Fetch data
st.write(f"📦 Fetching data for {selected_station} (Station ID: {selected_station_id}) for the past {past_hours} hours...")

df_actual = fetch_hourly_rides(past_hours)
df_predicted = fetch_predictions(past_hours)

# Ensure both DataFrames have the required columns
if "start_station_id" in df_actual.columns and "start_station_id" in df_predicted.columns:
    merged_df = pd.merge(df_actual, df_predicted, on=["start_station_id", "hour_ts"], how="inner")
else:
    st.error("❌ Column 'start_station_id' is missing from one of the data sources! Check fetch_hourly_rides and fetch_predictions functions.")
    st.stop()

# Filter data based on selected station
filtered_df = merged_df[merged_df["start_station_id"] == selected_station_id]

# Ensure required columns exist before computing absolute error
if "predicted_ride_count" not in filtered_df.columns or "ride_count" not in filtered_df.columns:
    st.error("❌ Missing required columns in the merged data! Check fetch_predictions and fetch_hourly_rides sources.")
    st.stop()

# Compute absolute error
filtered_df["absolute_error"] = abs(filtered_df["predicted_ride_count"] - filtered_df["ride_count"])

# Group by hour_ts and calculate MAE
mae_by_hour = filtered_df.groupby("hour_ts")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Create a Plotly line plot
fig = px.line(
    mae_by_hour,
    x="hour_ts",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for {selected_station} in the Past {past_hours} Hours",
    labels={"hour_ts": "Start Hour (EST)", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display the plot
st.plotly_chart(fig)

# Show average MAE
st.write(f'📈 Average MAE for {selected_station}: **{mae_by_hour["MAE"].mean():.2f}**')
