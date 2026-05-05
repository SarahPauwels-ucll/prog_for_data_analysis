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

    polutants = ["PM2.5","PM10","SO2","NO2","CO", "O3"]
    selected_polutant = st.sidebar.radio(
        "Select Polutant to Display",
        options=polutants
    )

   # --- MAIN PAGE ---
    st.title(f"Dashboard: {selected_station}")

    # Unpack the tuple from the slider
    start_date, end_date = selected_range

    # 3. Filter the dataframe
    selected_df = df[
        (df['station'] == selected_station) & 
        (df['datetime'].dt.date >= start_date) & 
        (df['datetime'].dt.date <= end_date)
    ].sort_values("datetime")

    cols_to_show = ['datetime'] + [selected_polutant]

    filtered_df = selected_df[cols_to_show]

    if not filtered_df.empty:
        # Set index to datetime to allow for resampling
        filtered_df = filtered_df.set_index('datetime')

        # --- NEW LOGIC: Conditional Resampling ---
        days_diff = (end_date - start_date).days
        
        if days_diff > 365:
            st.markdown(f"Displaying **monthly average** from **{start_date}** to **{end_date}** (> 365 days selected)")
            # Resample to Daily ('D') and take the mean
            filtered_df = filtered_df.resample('ME').mean()
        elif days_diff > 120:
            st.markdown(f"Displaying **weekly average** from **{start_date}** to **{end_date}** (> 120 days selected)")
            # Resample to Daily ('D') and take the mean
            filtered_df = filtered_df.resample('W').mean()
        elif days_diff > 14:
            st.markdown(f"Displaying **daily average** from **{start_date}** to **{end_date}** (> 14 days selected)")
            # Resample to Daily ('D') and take the mean
            filtered_df = filtered_df.resample('D').mean()
        
        else:
            st.markdown(f"Displaying **raw data** from **{start_date}** to **{end_date}**")

        # Graph
        st.subheader(f"{selected_polutant} Concentration Over Time")
        filteredgraph_df = filtered_df.rename(columns={"PM2.5": "PM25"})
        st.line_chart(filteredgraph_df)

        
        # Data Table
        st.subheader("Data Table")
        st.dataframe(filtered_df)
    else:
        st.warning("No data found for the selected station and date range.")