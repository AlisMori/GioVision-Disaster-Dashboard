# ======================================
# ENVIRONMENTAL OVERVIEW TAB
# ======================================

import os
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.merge_datasets import merge_datasets  # Make sure this exists

# ===========================
# CONSTANTS
# ===========================
DEFAULT_START = 2010
DEFAULT_END = 2025

ALERT_COLORS = {
    "Red":    "#EA6455",
    "Orange": "#EFB369",
    "Green":  "#59B3A9",
    "Unknown":"#8A8A8A",
}

MERGED_PATHS = [
    "data/processed/merged_emdat_eonet.csv",
    "../data/processed/merged_emdat_eonet.csv",
    "../../data/processed/merged_emdat_eonet.csv",
    "dashboard/data/processed/merged_emdat_eonet.csv",
]

# ===========================
# HELPERS
# ===========================
def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_merged(path):
    """Load merged CSV and normalize column names"""
    df = pd.read_csv(path)
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

def _halo_rgba(hex_color: str) -> str:
    base = hex_color.lstrip("#")
    r = int(base[0:2], 16); g = int(base[2:4], 16); b = int(base[4:6], 16)
    return f"rgba({r},{g},{b},0.25)"

def _center_zoom_from_points(lat_series: pd.Series, lon_series: pd.Series):
    lats = pd.to_numeric(lat_series, errors="coerce").dropna()
    lons = pd.to_numeric(lon_series, errors="coerce").dropna()

    if len(lats) == 0 or len(lons) == 0:
        return dict(lat=0, lon=0), 1.3  # global default

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    center = dict(lat=(lat_min + lat_max)/2, lon=(lon_min + lon_max)/2)

    lat_span = max(1e-6, lat_max - lat_min)
    lon_span = max(1e-6, lon_max - lon_min)
    k = 1.4
    zoom_from_lon = math.log2(360.0 / (lon_span * k))
    zoom_from_lat = math.log2(180.0 / (lat_span * k))
    zoom = max(1.0, min(zoom_from_lon, zoom_from_lat))
    zoom = min(8.0, zoom)
    if lon_span < 0.01 and lat_span < 0.01:
        zoom = 5.0
    return center, zoom

def _fmt(dt):
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return "—"

# ===========================
# MAIN RENDER
# ===========================
def render():
    """Render Environmental Overview tab with 3 visuals"""
    
    # Ensure merged CSV exists
    merged_path = _first_existing_path(MERGED_PATHS)
    if not merged_path:
        st.warning("Merged dataset not found. Generating it...")
        merged_path = merge_datasets()  # create it if missing

    df = load_merged(merged_path)

    if df.empty:
        st.warning("Merged dataset is empty.")
        return

    # Sidebar filters
    st.sidebar.header("Environmental Overview Filters")
    years_min = int(df["start year"].min()) if "start year" in df.columns else DEFAULT_START
    years_max = int(df["start year"].max()) if "start year" in df.columns else DEFAULT_END
    years_selected = st.sidebar.slider("Select Year Range", years_min, years_max, (years_min, years_max))
    
    region_list = sorted(df["region"].dropna().unique()) if "region" in df.columns else []
    region_selected = st.sidebar.selectbox("Select Region", ["All Regions"] + region_list)

    # Filter data
    df_filtered = df.copy()
    if "start year" in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered["start year"] >= years_selected[0]) & 
                                  (df_filtered["start year"] <= years_selected[1])]
    if region_selected != "All Regions" and "region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["region"] == region_selected]

    # --------------------------
    # TITLE
    # --------------------------
    section_title("Environmental Overview 🌍")
    st.markdown("Global overview of natural disasters combining EM-DAT & EONET datasets.")

    # --------------------------
    # 1. Interactive Map
    # --------------------------
    subsection_title("Interactive Global Disaster Map")
    if "latitude" in df_filtered.columns and "longitude" in df_filtered.columns:
        df_map = df_filtered.dropna(subset=["latitude","longitude"])
        if not df_map.empty:
            fig = px.scatter_mapbox(
                df_map,
                lat="latitude",
                lon="longitude",
                color="disaster type standardized" if "disaster type standardized" in df_map.columns else "disaster type",
                hover_name="event name" if "event name" in df_map.columns else None,
                hover_data=["country","region","start year"],
                mapbox_style="carto-positron",
                zoom=1,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No geolocation data for the selected filters.")
    else:
        st.info("Latitude/Longitude columns not found in dataset.")

    # --------------------------
    # 2. Total Disasters Over Time
    # --------------------------
    subsection_title("Total Disasters Over Time")
    if "start year" in df_filtered.columns:
        df_yearly = df_filtered.groupby("start year").size().reset_index(name="count")
        if not df_yearly.empty:
            fig_line = px.line(
                df_yearly,
                x="start year",
                y="count",
                markers=True,
                labels={"count":"Number of Disasters"},
                title="Number of Disasters per Year"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data available for this time period.")

    # --------------------------
    # 3. Top 10 Disaster Types
    # --------------------------
    subsection_title("Top 10 Disaster Types")
    type_col = "disaster type standardized" if "disaster type standardized" in df_filtered.columns else "disaster type"
    if type_col in df_filtered.columns:
        df_types = df_filtered[type_col].value_counts().head(10).reset_index()
        df_types.columns = ["Disaster Type","Count"]
        if not df_types.empty:
            fig_bar = px.bar(
                df_types,
                x="Disaster Type",
                y="Count",
                color="Disaster Type",
                color_discrete_sequence=px.colors.sequential.Reds,
                title="Top 10 Disaster Types"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No disaster type data for selected filters.")

    # --------------------------
    # FOOTER
    # --------------------------
    st.markdown("---")
    st.caption("Data source: EM-DAT & NASA EONET | Visualization: GioVision Dashboard")
