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
selected_date = st.date_input("Select a date for the forecast:",datetime(2018, 1, 1),min_value=datetime(2018, 1, 1),max_value=datetime(2018, 1, 31))

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

if st.button(f"Get Forecast for {selected_date}"):
    # Starting point for rolling features
    current_state = pd.DataFrame([{f: 0.0 for f in features}], columns=features)
    
    daily_results = []
    
    for hour in range(24):
        # Update weather features from our hourly lists
        for key in weather_data_24h.keys():
            if key in features:
                current_state[key] = weather_data_24h[key][hour]

        # Update the 'hour' feature
        if 'hour' in features:
            current_state['hour'] = hour
            
        step_preds = {'Hour': f"{hour:02d}:00"}
        
        for p in pollutants:
            # Predict using the model for this specific pollutant
            val = models[p].predict(current_state)[0]
            step_preds[p] = max(0, val) # Ensure no negative values
            
            # Recursive Update: Feed prediction back into rolling mean for next hour
            if f'{p}_roll_mean_6' in features:
                current_state[f'{p}_roll_mean_6'] = val
        
        daily_results.append(step_preds)

    # 4. Process Summary
    df_daily = pd.DataFrame(daily_results)
    
    # Calculate means for the metrics
    avg_pm25 = df_daily['PM2.5'].mean()
    avg_o3 = df_daily['O3'].mean()

    # 5. Display Summary UI
    st.subheader(f"Summary for {selected_date}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
    col2.metric("Avg Ozone", f"{avg_o3:.1f} µg/m³")
    
    if avg_pm25 < 35:
        st.success("✅ The air quality is expected to be Good today.")
    elif avg_pm25 < 75:
        st.warning("⚠️ The air quality may be Moderate.")
    else:
        st.error("🚨 Air quality is expected to be Unhealthy.")

    # 6. Show the trend chart
    st.subheader("Hourly Trends")
    st.line_chart(df_daily.set_index('Hour')[['PM2.5', 'O3', 'NO2']])
    
    # Optional: Show the weather inputs used
    with st.expander("View Simulated Weather Data"):
        st.write("Below is the hourly weather data used for this prediction:")
        weather_df = pd.DataFrame(weather_data_24h)
        weather_df.index.name = 'Hour'
        st.dataframe(weather_df)