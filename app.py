import streamlit as st

from common import APP_NAME

st.set_page_config(page_title=APP_NAME, layout="wide")

pg = st.navigation([
    st.Page("views/team_comparison.py", title=APP_NAME, default=True),
    st.Page("views/ml_clustering.py", title="ML Clustering"),
])
pg.run()
