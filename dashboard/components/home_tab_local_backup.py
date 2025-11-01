# dashboard/components/home_tab.py
import streamlit as st
import datetime

# ===========================
# THEME HELPERS
# ===========================
def section_title(text: str):
    """Theme-aligned section bar (matches app theme)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    """Theme-aligned subsection bar (matches app theme)."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

# ===========================
# MAIN RENDER
# ===========================
def render():
    section_title("GeoVision Disaster Dashboard 🛰️")
    
    subsection_title("Summary Overview")
    st.markdown(
        """
        Welcome to the **GeoVision Disaster Dashboard (GDD)** — an interactive platform
        that visualizes global natural disaster data from **NASA EONET**, **GDACS**, and **EM-DAT**.
        This dashboard was developed by **Team GeoVision** for ICT305 — Data Visualisation and Simulation (Murdoch University, 2025).

        **What this dashboard does**
        - Provides real-time and historical views of natural disasters (floods, storms, earthquakes, wildfires, etc.).  
        - Helps decision-makers and NGOs quickly identify hotspots, track trends, and prioritise response.  
        - Combines interactive maps, time series, and country-level summaries to make insights accessible.

        **Who this is for**
        - Emergency response teams and NGOs for situational awareness.  
        - Researchers and students analysing disaster trends.  
        - Policy makers and planners monitoring risk and impact.

        **Quick guide**
        - Use **Environmental Overview** for global maps and severity.  
        - Use **Impact of Natural Disasters** to dig into human & economic impact.  
        - Use **Alerts** for live GDACS notifications.
        """
    )

    subsection_title("Key Global Indicators (Year-to-Date)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Disasters (YTD)", "124")
    col2.metric("People Affected (YTD)", "2.3M")
    col3.metric("Fatalities (YTD)", "12,300")
    col4.metric("Economic Loss (est.)", "$1.2B")

    st.markdown("---")
    subsection_title("Project & Team")
    st.markdown(
        """
        **Team GeoVision** — Aleena, Fatima, Minal, Alena, and Zhyldyz  
        Course: ICT305 — Data Visualisation and Simulation, Murdoch University (2025)  
        Repo: `GioVision-Disaster-Dashboard` on GitHub.
        """
    )

    st.markdown("---")
    st.caption("Data sources: NASA EONET | GDACS | EM-DAT")
    st.caption("Last updated: " + datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
