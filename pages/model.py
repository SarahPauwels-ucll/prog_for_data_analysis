import streamlit as st

# Define the pages
dashboard_page = st.Page(
    "streamlit_app.py", 
    title="Data Dashboard", 
    icon="📊", 
    default=True
)

model_page = st.Page(
    "pages/model.py", 
    title="Model Prediction", 
    icon="🤖"
)

# Initialize navigation
pg = st.navigation([dashboard_page, model_page])
pg.run()

st.set_page_config(page_title="Model", layout="wide")

st.title("🤖 Model Analysis")
