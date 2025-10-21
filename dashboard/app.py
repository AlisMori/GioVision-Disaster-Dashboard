"""
app.py
------
Main Streamlit entry point for the GeoVision Disaster Dashboard.
Uses both dynamic theme (style_config.py) and static CSS (style.css).
"""

import sys
import os
from pathlib import Path

# Fix Streamlit path issue — ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# (optional) confirm it's added
# print("Python path includes:", ROOT)


import streamlit as st
from dashboard.components import overview_tab, trends_tab, comparisons_tab, alerts_tab, impact_tab
from src.utils import style_config

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="GeoVision Disaster Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------
# APPLY GLOBAL STYLES
# ----------------------------

# 1️⃣ Apply dynamic style theme from style_config.py
style_config.apply_streamlit_style()

# 2️⃣ Load static CSS for layout / formatting tweaks
css_path = os.path.join("dashboard", "assets", "css", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ style.css not found — layout styles may not render properly.")

# ----------------------------
# DASHBOARD HEADER
# ----------------------------
st.title("🌍 GeoVision Disaster Dashboard (Prototype)")
st.caption("ICT305 – Data Visualisation and Simulation | Murdoch University, 2025")

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Impact of Natural Disasters", "Trends", "Comparisons", "Alerts"]
)

# ----------------------------
# PAGE ROUTING LOGIC
# ----------------------------
if page == "Overview":
    overview_tab.render()
elif page == "Impact of Natural Disasters":
    impact_tab.render()
elif page == "Trends":
    trends_tab.render()
elif page == "Comparisons":
    comparisons_tab.render()
elif page == "Alerts":
    alerts_tab.render()
else:
    st.warning("This section is under construction.")

# ----------------------------
# FOOTER MESSAGE
# ----------------------------
st.markdown("---")
st.info(
    "💡 This is a placeholder version of the GeoVision Dashboard. "
    "Functionality, data integration, and visuals are under active development."
)
