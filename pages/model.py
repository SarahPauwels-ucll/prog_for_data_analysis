import streamlit as st
from st_pages import Page, add_page_title, show_pages

show_pages(
    [
        Page("streamlit_app.py", "data visualisation", ":books:"),
        Page("pages/model.py", "model", ":robot_face:"),
    ]
)

st.set_page_config(page_title="Model", layout="wide")

st.title("🤖 Model Analysis")
