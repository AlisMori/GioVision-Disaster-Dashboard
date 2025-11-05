# dashboard/components/home_tab.py
import os
import streamlit as st
import datetime
import pandas as pd

# =========================
# RENDERING HELPERS
# =========================
def section_title(text: str):
    """Main section bar (registered by app.py capture)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)


def subsection_title(text: str):
    """Smaller subsection bar."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)


def story_context(text: str):
    st.markdown(f'<div class="gv-context">{text}</div>', unsafe_allow_html=True)


# =========================
# DATA / PATHS
# =========================
EMDAT_PATHS = [
    "data/processed/emdat_cleaned.csv",
    "data/emdat_cleaned.csv",
    "dashboard/data/emdat_cleaned.csv",
    "../data/processed/emdat_cleaned.csv",
]

def read_first_existing_csv(paths):
    """
    Try to read emdat_cleaned.csv from several possible locations.
    Returns a dataframe or None.
    """
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                # file exists but unreadable → try next
                pass
    return None


# ===========================
# MAIN RENDER
# ===========================
def render():
    section_title("GeoVision Disaster Dashboard")
    
    subsection_title("Summary Overview")
    st.markdown(
        """
        Welcome to the **GeoVision Disaster Dashboard (GDD)**, an interactive platform
        that visualizes global natural disaster data from **NASA EONET**, **GDACS**, and **EM-DAT**.
        This dashboard was developed by **Team GeoVision** for ICT305 - Data Visualisation and Simulation (Murdoch University, 2025).

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

    # try all known locations
    df = read_first_existing_csv(EMDAT_PATHS)
    if df is None:
        st.warning("⚠️ Dataset not found. Please ensure EM-DAT is available in one of the known paths.")
        return

    # current year (UTC so it's consistent with the rest of the app)
    current_year = datetime.datetime.utcnow().year

    # Filter for current year
    if "Start Year" in df.columns:
        df_ytd = df[df["Start Year"] == current_year].copy()
    else:
        # fallback if dataset uses Event Date
        if "Event Date" in df.columns:
            df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")
            df_ytd = df[df["Event Date"].dt.year == current_year].copy()
        else:
            df_ytd = df.copy()

    # make sure required columns exist
    if "Total Affected" not in df_ytd.columns:
        df_ytd["No. Injured"] = df_ytd.get("No. Injured", 0).fillna(0)
        df_ytd["No. Affected"] = df_ytd.get("No. Affected", 0).fillna(0)
        df_ytd["No. Homeless"] = df_ytd.get("No. Homeless", 0).fillna(0)
        df_ytd["Total Affected"] = (
            df_ytd["No. Injured"] + df_ytd["No. Affected"] + df_ytd["No. Homeless"]
        )

    if "Total Deaths" not in df_ytd.columns:
        df_ytd["Total Deaths"] = df_ytd.get("Total Deaths", 0).fillna(0)

    if "Total Damage ('000 US$)" not in df_ytd.columns:
        df_ytd["Total Damage ('000 US$)"] = df_ytd.get("Total Damage ('000 US$)", 0).fillna(0)

    # Compute metrics
    total_disasters = len(df_ytd)
    total_affected = df_ytd["Total Affected"].sum()
    total_fatalities = df_ytd["Total Deaths"].sum()
    total_damage_thousands = df_ytd["Total Damage ('000 US$)"].sum()

    # Format numbers for display
    def fmt(num):
        if pd.isna(num):
            return "N/A"
        try:
            num = float(num)
        except Exception:
            return str(num)
        if num >= 1_000_000_000:
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
    # convert '000 US$ → US$
    col4.metric("Economic Loss (est.)", "$" + fmt(total_damage_thousands * 1000))

    # ===========================
    # TEAM INFO
    # ===========================
    st.markdown("---")
    subsection_title("Project & Team")
    st.markdown(
        """
        **Team GeoVision Analytics** : Aleena Fatima, Fatima Faisal, Minal Haque, Alena Bobyleva, and Zhyldyz Kadyrovna Davydova  
        **Course** : ICT305 Data Visualisation and Simulation, Murdoch University (2025)  
        **Repo** : `GioVision-Disaster-Dashboard` on GitHub.
        """
    )

    st.markdown("---")
    st.caption("Data sources: NASA EONET | GDACS | EM-DAT")
    st.caption("Last updated: " + datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
