# src/app.py

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np
import altair as alt
import config

# Setup import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Hopsworks import
from src.citi_interface import get_feature_store

# =============================
# 📥 Load Citi Bike Data
# =============================

@st.cache_data
def load_citibike_data():
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    df = feature_view.get_batch_data()
    return df

# =============================
# 🎨 Inject Custom CSS and JS
# =============================

custom_css = """
<style>
/* Background Animation */
body {
    background: linear-gradient(-45deg, #d0f0f7, #f0f8ff, #e0ffff, #b0e0e6);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

/* Gradient keyframes */
@keyframes gradientBG {
    0% {background-position:0% 50%}
    50% {background-position:100% 50%}
    100% {background-position:0% 50%}
}

/* Hide extra elements */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Custom Cursor */
* {
    cursor: url('https://cdn-icons-png.flaticon.com/512/3069/3069170.png'), auto;
}

/* Loader Styling */
.loader-container {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    gap: 20px;
    z-index: 10000;
    background: none;
}

.bike {
    font-size: 40px;
    animation: moveBike 4s infinite alternate;
}

.bike.left {
    animation-direction: alternate-reverse;
}

@keyframes moveBike {
    0% { transform: translateX(0px) scale(1); }
    50% { transform: translateX(60px) scale(1.5); }
    100% { transform: translateX(0px) scale(1); }
}
</style>

<!-- Loader Div -->
<div id="loader" class="loader-container">
  <div class="bike">🚲</div>
  <div class="bike left">🚲</div>
  <div class="bike">🚲</div>
  <div class="bike left">🚲</div>
  <div class="bike">🚲</div>
</div>

<script>
// Hide loader when page fully loaded
window.addEventListener('load', () => {
    setTimeout(() => {
        document.getElementById('loader').style.display = 'none';
    }, 1500); // slight delay to finish animation nicely
});
</script>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# =============================
# 🎯 Main App
# =============================

st.title("🚲 Citi Bike Trip Prediction Dashboard (LIVE Updates!)")

# Load data
df = load_citibike_data()

if df is None or df.empty:
    st.error("❌ No data available from feature store!")
    st.stop()

# Prediction error
df["prediction_error"] = abs(df["target_ride_count"] - df["ride_count_roll3"])

# 1. 📈 MAE Over Time
with st.container():
    st.subheader("📈 MAE Over Time")
    mae_time = df.groupby("hour_ts")["prediction_error"].mean().reset_index()
    chart_mae_time = alt.Chart(mae_time).mark_line().encode(
        x="hour_ts:T", y="prediction_error:Q", tooltip=["hour_ts", "prediction_error"]
    ).properties(height=400)
    st.altair_chart(chart_mae_time, use_container_width=True)
    st.caption(f"🕒 Last data update: **{mae_time['hour_ts'].max()}**")

# 2. 🏆 Top Stations Leaderboard
with st.container():
    st.subheader("🏆 Top Stations with Lowest MAE")
    top_n = st.slider("Select Top N Stations", 1, 5, 3)
    top_stations = (df.groupby(["start_station_id", "start_station_name"])
                      .agg(avg_mae=("prediction_error", "mean"))
                      .reset_index()
                      .sort_values("avg_mae"))
    st.dataframe(top_stations.head(top_n), hide_index=True)

# 3. 🗺️ NYC Map of Best Stations
with st.container():
    st.subheader("🗺️ Best Stations on Map (Demo Locations)")
    np.random.seed(42)
    map_data = top_stations.head(top_n).copy()
    map_data["latitude"] = 40.75 + np.random.rand(len(map_data)) * 0.05
    map_data["longitude"] = -73.98 + np.random.rand(len(map_data)) * 0.05

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["longitude", "latitude"],
        get_color="[0, 128, 255, 180]",
        get_radius=400,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=40.75, longitude=-73.97, zoom=11)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state,
                 tooltip={"text": "Station: {start_station_name}\nMAE: {avg_mae:.2f}"})
    st.pydeck_chart(r)

# 4. 📅 MAE by Day of Week
with st.container():
    st.subheader("📅 MAE by Day of Week")
    day_mae = df.groupby("day_of_week")["prediction_error"].mean().reset_index()
    chart_day = alt.Chart(day_mae).mark_bar().encode(
        x="day_of_week:O", y="prediction_error:Q", color="day_of_week:N"
    )
    st.altair_chart(chart_day, use_container_width=True)

# 5. 📆 MAE by Month
with st.container():
    st.subheader("📆 MAE by Month")
    month_mae = df.groupby("month")["prediction_error"].mean().reset_index()
    chart_month = alt.Chart(month_mae).mark_bar(color="orange").encode(
        x="month:O", y="prediction_error:Q"
    )
    st.altair_chart(chart_month, use_container_width=True)

# 6. 🏖 MAE on Holidays vs Weekdays
with st.container():
    st.subheader("🏖 Holidays vs Weekdays MAE")
    holiday_mae = df.groupby("is_holiday_or_weekend")["prediction_error"].mean().reset_index()
    holiday_mae["type"] = holiday_mae["is_holiday_or_weekend"].map({True: "Holiday/Weekend", False: "Weekday"})
    chart_holiday = alt.Chart(holiday_mae).mark_bar().encode(
        x="type:N", y="prediction_error:Q", color="type:N"
    )
    st.altair_chart(chart_holiday, use_container_width=True)

# 7. 🕰️ MAE by Time of Day
with st.container():
    st.subheader("🕰️ MAE by Time of Day")
    timeofday_mae = df.groupby("time_of_day")["prediction_error"].mean().reset_index()
    chart_timeofday = alt.Chart(timeofday_mae).mark_bar(color="purple").encode(
        x="time_of_day:O", y="prediction_error:Q"
    )
    st.altair_chart(chart_timeofday, use_container_width=True)
