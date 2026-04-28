import streamlit as st

st.title("🤖 Model Analysis")

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load your saved joblib file
@st.cache_resource
def load_model_data():
    return joblib.load('data/air_quality_models.joblib')

data = load_model_data()
models = data['models']
features = data['features']
pollutants = data['pollutants']

st.title("📅 Daily Air Quality Planner")

# 1. User Input: Date
selected_date = st.date_input("Select a date for the forecast:", datetime.now())

# 2. Hardcoded Weather (Simulating an API call)
# In the future, you'd replace this with a weather API request for 'selected_date'
hardcoded_weather = {
    'TEMP': 22.5,
    'PRES': 1012.0,
    'DEWP': 12.4,
    'RAIN': 0.0,
    'wd': 2,    # Encoded wind direction
    'WSPM': 1.8
}

if st.button(f"Get Forecast for {selected_date}"):
    # We need a starting point for the rolling features. 
    # For this example, we'll start with "clean air" assumptions (zeros/averages).
    current_state = pd.DataFrame([{f: 0.0 for f in features}], columns=features)
    
    # Update weather features with our hardcoded values
    for key, value in hardcoded_weather.items():
        if key in features:
            current_state[key] = value

    # 3. Generate 24 hours of data
    daily_results = []
    
    for hour in range(24):
        # Update the 'hour' feature if it exists in your training
        if 'hour' in features:
            current_state['hour'] = hour
            
        step_preds = {'Hour': f"{hour}:00"}
        for p in pollutants:
            # Predict
            val = models[p].predict(current_state)[0]
            step_preds[p] = max(0, val) # Ensure no negative pollution
            
            # Recursive Update: Update rolling mean for the next hour
            if f'{p}_roll_mean_6' in features:
                current_state[f'{p}_roll_mean_6'] = val
        
        daily_results.append(step_preds)

    # 4. Process Summary
    df_daily = pd.DataFrame(daily_results)
    avg_pm25 = df_daily['PM2.5'].mean()
    avg_o3 = df_daily['O3'].mean()

    # 5. Display Summary UI
    st.subheader(f"Summary for {selected_date}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
    col2.metric("Avg Ozone", f"{avg_o3:.1f} µg/m³")
    
    # Simple Health Logic
    if avg_pm25 < 35:
        st.success("✅ The air quality is expected to be Good today.")
    else:
        st.warning("⚠️ The air quality may be Moderate to Unhealthy.")

    # Show the trend chart
    st.line_chart(df_daily.set_index('Hour')[['PM2.5', 'O3', 'NO2']])