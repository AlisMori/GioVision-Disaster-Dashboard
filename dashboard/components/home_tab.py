# dashboard/components/home_tab.py
import streamlit as st
import datetime
import pandas as pd

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
    section_title("GeoVision Disaster Dashboard")
    
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

    # ===========================
    # KEY GLOBAL INDICATORS
    # ===========================
    subsection_title("Key Global Indicators (Year-to-Date)")

    # Load the cleaned dataset (update the path if needed)
    try:
      df = pd.read_csv("data/processed/emdat_cleaned.csv")

    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Please ensure 'data/emdat_cleaned.csv' exists.")
        return

    # Filter for current year
    current_year = datetime.datetime.utcnow().year
    df_ytd = df[df["Start Year"] == current_year]

    # Compute metrics
    total_disasters = len(df_ytd)
    total_affected = df_ytd["Total Affected"].sum()
    total_fatalities = df_ytd["Total Deaths"].sum()
    total_damage = df_ytd["Total Damage ('000 US$)"].sum()

    # Format numbers for display
    def fmt(num):
        if pd.isna(num):
            return "N/A"
        elif num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:,.0f}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Disasters (YTD)", str(total_disasters))
    col2.metric("People Affected (YTD)", fmt(total_affected))
    col3.metric("Fatalities (YTD)", fmt(total_fatalities))
    col4.metric("Economic Loss (est.)", "$" + fmt(total_damage * 1000))  # convert '000 US$ to US$

    # ===========================
    # TEAM INFO
    # ===========================
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
