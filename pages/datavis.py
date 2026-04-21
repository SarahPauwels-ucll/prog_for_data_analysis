import streamlit as st
import pandas as pd
import os

# Helper function to load data
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    # Convert datetime column to actual datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# 1. Load the data
data_path = os.path.join("data", "preprocesed.csv")
df = load_data(data_path)

import pandas as pd
import streamlit as st

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Filter Settings")

    # Station Selection
    stations = sorted(df['station'].unique())
    selected_station = st.sidebar.selectbox("Select a Station", stations)

    # 1. Get the min and max dates from the dataframe
    # We convert to .date() to ensure the slider handles whole days
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    # 2. Use st.sidebar.slider for the range selection
    # Providing a tuple to 'value' creates a range slider
    selected_range = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        
    )

    # --- MAIN PAGE ---
    st.title(f"Dashboard: {selected_station}")

    # Unpack the tuple from the slider
    start_date, end_date = selected_range
    st.markdown(f"Displaying data from **{start_date}** to **{end_date}**")

    # 3. Filter the dataframe
    # We compare the .dt.date component to the slider's date objects
    filtered_df = df[
        (df['station'] == selected_station) & 
        (df['datetime'].dt.date >= start_date) & 
        (df['datetime'].dt.date <= end_date)
    ].sort_values("datetime")

    if not filtered_df.empty:
        # Data Table
        st.subheader("Raw Data")
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data found for the selected station and date range.")
