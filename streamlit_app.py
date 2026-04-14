import streamlit as st
# Define the pages
dashboard_page = st.Page(
    "pages/datavis.py", 
    title="Data Dashboard", 
    icon="📊", 
    default=True
)

model_page = st.Page(
    "pages/model.py", 
    title="Model Prediction", 
    icon="🤖"
)

st.set_page_config(page_title="Station Data Dashboard", layout="wide")

# Initialize navigation
pg = st.navigation([dashboard_page, model_page])

pg.run()
