import os
import sys

import streamlit as st

# Ensure project root is importable from all page files
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="NBA Win Predictor & Roster Simulator",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/p1_forecast.py",       title="Season Forecast",        icon="🔮"),
    st.Page("pages/p2_simulator.py",      title="Roster Simulator",       icon="🔄"),
    st.Page("pages/p3_shap.py",           title="Why This Prediction",    icon="🧠"),
    st.Page("pages/p4_history.py",        title="Historical Performance", icon="📈"),
    st.Page("pages/p5_team_dashboard.py", title="Team & Roster",          icon="👥"),
    st.Page("pages/p6_model_perf.py",     title="Model Performance",      icon="⚙️"),
]

pg = st.navigation(pages)
pg.run()
