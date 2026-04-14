import streamlit as st
from st_pages import Page, add_page_title, show_pages
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Station Data Dashboard", layout="wide")

show_pages(
    [
        Page("streamlit_app.py", "data visualisation", ":books:"),
        Page("pages/model.py", "model", ":robot_face:"),
    ]
)

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

    # Datetime Selection
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    # --- MAIN PAGE ---
    st.title(f"Dashboard: {selected_station}")
    st.markdown(f"Displaying data for **{selected_date}**")

    # Filter the dataframe
    filtered_df = df[
        (df['station'] == selected_station) & 
        (df['datetime'].dt.date == selected_date)
    ].sort_values("datetime")

    if not filtered_df.empty:
        # Data Table
        st.subheader("Raw Data")
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No data found for the selected station and date.")
