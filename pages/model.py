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

    polutants = ["PM2.5","PM10","SO2","NO2","CO", "O3"]
    selected_polutant = st.sidebar.radio(
        "Select Polutant to Display",
        options=["PM2.5","SO2","NO2","CO", "O3"]
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

# Generate 24-Hour Forecast
for hour in range(24):
    row_data = {}
    
    # A. Inject Hardcoded Weather for the specific hour
    for key in weather_data_24h.keys():
        if key in features:
            row_data[key] = weather_data_24h[key][hour]
    
    if 'hour' in features:
        row_data['hour'] = hour

    # Calculate Rolling Features
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
    
    # Predict all pollutants for this hour
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

# Display Results
df_daily = pd.DataFrame(daily_results)

def calculate_china_iaqi(concentration, pollutant):
    """Calculates IAQI based on HJ 633-2012 Chinese Standard."""
    # Breakpoints: [Concentration_Low, Concentration_High, IAQI_Low, IAQI_High]
    breakpoints = {
        'PM2.5': [(0, 35, 0, 50), (35, 75, 50, 100), (75, 115, 100, 150), 
                  (115, 150, 150, 200), (150, 250, 200, 300), (250, 350, 300, 400), (350, 500, 400, 500)],
        'PM10':  [(0, 50, 0, 50), (50, 150, 50, 100), (150, 250, 100, 150), 
                  (250, 350, 150, 200), (350, 420, 200, 300), (420, 500, 300, 400), (500, 600, 400, 500)],
        'SO2':   [(0, 50, 0, 50), (50, 150, 50, 100), (150, 475, 100, 150), 
                  (475, 800, 150, 200), (800, 1600, 200, 300), (1600, 2100, 300, 400), (2100, 2620, 400, 500)],
        'NO2':   [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 150), 
                  (180, 280, 150, 200), (280, 565, 200, 300), (565, 750, 300, 400), (750, 940, 400, 500)],
        'CO':    [(0, 2, 0, 50), (2, 4, 50, 100), (4, 14, 100, 150), 
                  (14, 24, 150, 200), (24, 36, 200, 300), (36, 48, 300, 400), (48, 60, 400, 500)],
        'O3':    [(0, 160, 0, 50), (160, 200, 50, 100), (200, 300, 100, 150), 
                  (300, 400, 150, 200), (400, 800, 200, 300), (800, 1000, 300, 400), (1000, 1200, 400, 500)]
    }
    
    if pollutant not in breakpoints:
        return 0
    
    for (c_low, c_high, i_low, i_high) in breakpoints[pollutant]:
        if c_low <= concentration <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
    return 500 # Cap at 500 if above scale


st.subheader(f"Summary for {selected_station} on {selected_date}")


# Note: PM10 = PM2.5 + PM2.5-10
avg_pm25 = df_daily['PM2.5'].mean()
avg_pm10 = (df_daily['PM10']).mean()
avg_so2 = df_daily['SO2'].mean()
avg_no2 = df_daily['NO2'].mean()
avg_co = df_daily['CO'].mean() / 1000.0  # Convert µg/m³ to mg/m³ for CO calculation
max_o3 = df_daily['O3'].max()           # Ozone uses 1-hour max for AQI

# 2. Calculate IAQI for each
iaqis = {
    "PM2.5": calculate_china_iaqi(avg_pm25, 'PM2.5'),
    "PM10": calculate_china_iaqi(avg_pm10, 'PM10'),
    "SO2": calculate_china_iaqi(avg_so2, 'SO2'),
    "NO2": calculate_china_iaqi(avg_no2, 'NO2'),
    "CO": calculate_china_iaqi(avg_co, 'CO'),
    "O3": calculate_china_iaqi(max_o3, 'O3')
}

# 3. Final AQI is the maximum IAQI
aqi_val = max(iaqis.values())

# Display Main Metrics
col1, col2, col3 = st.columns(3)
col1.metric("China AQI", f"{aqi_val:.0f}")
col2.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
col3.metric("Max Ozone", f"{max_o3:.1f} µg/m³")


if aqi_val <= 50:
    st.success(f"✅ **AQI: {aqi_val:.0f} (Excellent)**. The air quality is satisfactory and poses little or no risk.")
elif aqi_val <= 100:
    st.success(f"🟡 **AQI: {aqi_val:.0f} (Good)**. Air quality is acceptable; some pollutants may pose a moderate health concern for sensitive individuals.")
elif aqi_val <= 150:
    st.warning(f"⚠️ **AQI: {aqi_val:.0f} (Lightly Polluted)**. Children and people with respiratory diseases should reduce outdoor exertion.")
elif aqi_val <= 200:
    st.error(f"🚨 **AQI: {aqi_val:.0f} (Moderately Polluted)**. Healthy people may experience symptoms; sensitive groups should avoid outdoor activities.")
elif aqi_val <= 300:
    st.error(f"🛑 **AQI: {aqi_val:.0f} (Heavily Polluted)**. General population should significantly reduce outdoor activities.")
else:
    st.error(f"💀 **AQI: {aqi_val:.0f} (Severely Polluted)**. Everyone should avoid all outdoor activities.")



st.subheader(f"Hourly Trend: {selected_polutant}")
dailygraph_df = df_daily.rename(columns={"PM2.5": "PM25"})
if selected_polutant=="PM2.5":
    selected_polutant="PM25"
st.line_chart(dailygraph_df.set_index('Hour')[selected_polutant])


