import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    # Convert datetime column to actual datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

data_path = os.path.join("data", "preprocesed.csv")
df = load_data(data_path)
if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Filter Settings")

    # Station Selection
    stations = sorted(df['station'].unique())
    selected_station = st.sidebar.selectbox("Select a Station", stations)

    polutants = ["PM25","PM10","SO2","NO2","CO", "O3"]
    selected_polutant = st.sidebar.radio(
        "Select Polutant to Display",
        options=polutants
    )

# Load your saved joblib file
@st.cache_resource
def load_model_data():
    return joblib.load('data/air_quality_models.joblib')

data = load_model_data()
models = data['models']
features = data['features']
pollutants = data['pollutants']

st.title("📅 Air Quality Prediction")

# 1. User Input: Date
selected_date = datetime(2017, 3, 1)
# 2. Realistic Hardcoded Weather Patterns (24 values for each)
# These simulate a standard day: cold morning, hot afternoon, cooling evening.
weather_data_24h = {
    'TEMP': [12.1, 11.5, 11.0, 10.8, 10.5, 11.2, 13.5, 16.0, 18.5, 21.0, 23.5, 25.0, 
             26.5, 27.2, 27.0, 26.5, 24.0, 21.5, 19.0, 17.5, 16.0, 15.0, 14.2, 13.5],
    'PRES': [1015]*24, # Pressure stays relatively stable
    'DEWP': [10.2, 10.1, 10.0, 10.0, 9.8, 9.5, 9.0, 8.5, 8.0, 8.2, 8.5, 9.0, 
             9.5, 10.0, 10.2, 10.5, 11.0, 11.2, 11.5, 11.5, 11.4, 11.2, 11.0, 10.8],
    'RAIN': [0.0]*14 + [0.2, 0.5, 0.1] + [0.0]*7, # Simulated light afternoon shower
    'wd': [2, 2, 2, 3, 3, 3, 4, 4, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 2, 2, 2, 2],
    'WSPM': [0.8, 0.7, 0.6, 0.5, 0.6, 1.2, 1.8, 2.2, 2.5, 2.8, 3.2, 3.5, 
             3.8, 4.0, 3.7, 3.2, 2.8, 2.2, 1.5, 1.2, 1.0, 0.9, 0.8, 0.8]
}

history_df = df[df['station'] == selected_station].sort_values('datetime').tail(24)

# Create a dictionary to track the sliding windows for each pollutant
# This stores the most recent 24 values for every pollutant
pollutant_history = {p: history_df[p].tolist() for p in pollutants}

# Initialize current_state with the correct feature columns
current_state = pd.DataFrame(columns=features)

daily_results = []

# 2. Generate 24-Hour Forecast
for hour in range(24):
    row_data = {}
    
    # A. Inject Hardcoded Weather for the specific hour
    for key in weather_data_24h.keys():
        if key in features:
            row_data[key] = weather_data_24h[key][hour]
    
    if 'hour' in features:
        row_data['hour'] = hour

    # B. Calculate Rolling Features (The "Seed" Data)
    # As per model.py, we use the mean of the previous 'w' hours
    for p in pollutants:
        for w in [6, 24]:
            col_name = f'{p}_roll_mean_{w}'
            if col_name in features:
                # Take the last 'w' values from our history buffer
                recent_values = pollutant_history[p][-w:]
                row_data[col_name] = sum(recent_values) / len(recent_values)

    # Convert row_data to DataFrame for prediction
    current_state = pd.DataFrame([row_data])
    
    # C. Predict all pollutants for this hour
    step_preds = {'Hour': f"{hour:02d}:00"}
    for p in pollutants:
        val = models[p].predict(current_state)[0]
        val = max(0, val) # Prevent negative pollution
        step_preds[p] = val
        
        # D. Update History Buffer for the NEXT hour
        # We append the prediction so it's included in next hour's rolling mean
        pollutant_history[p].append(val)
        pollutant_history[p].pop(0) # Keep the buffer at exactly 24 items
    
    daily_results.append(step_preds)

# 3. Display Results
df_daily = pd.DataFrame(daily_results)

# Metrics and Chart
st.subheader(f"Summary for {selected_station} on {selected_date}")
avg_pm25 = df_daily['PM2.5'].mean()
avg_o3 = df_daily['O3'].mean()

col1, col2 = st.columns(2)
col1.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
col2.metric("Avg Ozone", f"{avg_o3:.1f} µg/m³")


if avg_pm25 < 35:
    st.success("✅ The air quality is expected to be Good today.")
elif avg_pm25 < 75:
    st.warning("⚠️ The air quality may be Moderate.")
else:
    st.error("🚨 Air quality is expected to be Unhealthy.")

st.subheader(f"Hourly Trend: {selected_polutant}")
st.line_chart(df_daily.set_index('Hour')[selected_polutant])

# 4. Process Summary
df_daily = pd.DataFrame(daily_results)

# Calculate means for the metrics
avg_pm25 = df_daily['PM25'].mean()
avg_o3 = df_daily['O3'].mean()
