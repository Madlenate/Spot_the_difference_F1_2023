import pandas as pd
import streamlit as st
from pathlib import Path

APP_NAME = "Formula 1 Analytics"
SEASON = 2023
DATA_PATH = Path(__file__).parent / "f1_data_2023" / "lap_features_labeled.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
