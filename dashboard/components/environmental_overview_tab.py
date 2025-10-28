"""
Environmental Overview Tab

Displays key statistics and a high-level overview of global disaster data,
including a styled map consistent with the Alerts tab (Carto-Positron with
halo/ring/main markers).
"""

import os
import math
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly import graph_objects as go

# Import merge function from utils
from src.utils.merge_datasets import merge_datasets

# ===========================
# THEME HELPERS
# ===========================
def _anchor(id_: str):
    """Invisible HTML anchor for smooth scrolling targets."""
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    """Theme-aligned section bar (matches app theme)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    """Theme-aligned subsection bar (matches app theme)."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

def _fmt(dt):
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return "—"

# ===========================
# PALETTE (match app)
# ===========================
ALERT_COLORS = {
    "Red":    "#EA6455",
    "Orange": "#EFB369",
    "Green":  "#59B3A9",
    "Unknown":"#8A8A8A",
}

# ===========================
# MAP HELPERS
# ===========================
def _center_zoom_from_points(lat_series: pd.Series, lon_series: pd.Series):
    """Compute approximate (center, zoom) for Mapbox from bounds."""
    lats = pd.to_numeric(lat_series, errors="coerce").dropna()
    lons = pd.to_numeric(lon_series, errors="coerce").dropna()

    if len(lats) == 0 or len(lons) == 0:
        return dict(lat=0, lon=0), 1.3  # global default

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    center = dict(lat=(lat_min + lat_max) / 2.0, lon=(lon_min + lon_max) / 2.0)

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

def _halo_rgba(hex_color: str) -> str:
    base = hex_color.lstrip("#")
    r = int(base[0:2], 16); g = int(base[2:4], 16); b = int(base[4:6], 16)
    return f"rgba({r},{g},{b},0.25)"

# ===========================
# DATA HELPERS
# ===========================
def load_data():
    merged_path = merge_datasets()
    df = pd.read_csv(merged_path)
    # Normalize column names to lowercase
    df.columns = [col.lower().strip() for col in df.columns]
    return df

# ===========================
# MAIN RENDER
# ===========================
def render():
    df = load_data()
    if df.empty:
        st.warning("No data available for the Environmental Overview tab.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("Environmental Overview Filters")
    years_min = int(df["start year"].min()) if "start year" in df else 2010
    years_max = int(df["start year"].max()) if "start year" in df else 2025
    years_selected = st.sidebar.slider(
        "Select Year Range",
        min_value=years_min,
        max_value=years_max,
        value=(years_min, years_max)
    )

    region_list = sorted(df["region"].dropna().unique()) if "region" in df.columns else []
    region_selected = st.sidebar.selectbox("Select Region", ["All Regions"] + region_list)

    # Filter data
    df_filtered = df.copy()
    if "start year" in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered["start year"] >= years_selected[0]) & 
            (df_filtered["start year"] <= years_selected[1])
        ]
    if region_selected != "All Regions" and "region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["region"] == region_selected]

    # Title
    section_title("Environmental Overview 🌍")
    st.markdown(
        "This section provides a global overview of natural disasters "
        "between 2010–2025, combining data from EM-DAT and NASA EONET."
    )

    # -------------------------
    # 1. Interactive Map
    # -------------------------
    subsection_title("Interactive Global Disaster Map")
    df_map = df_filtered.dropna(subset=["latitude", "longitude"]) if "latitude" in df_filtered and "longitude" in df_filtered else pd.DataFrame()
    if not df_map.empty:
        fig_map = go.Figure()
        main_size, ring_size, halo_size = 11, 14, 26
        for lvl in df_map.get("disaster type standardized", "Unknown").fillna("Unknown").unique():
            sub = df_map[df_map["disaster type standardized"].fillna("Unknown") == lvl]
            color_hex = ALERT_COLORS.get(lvl, ALERT_COLORS["Unknown"])

            # Halo
            fig_map.add_trace(go.Scattermapbox(
                lat=sub["latitude"], lon=sub["longitude"], mode="markers",
                marker=dict(size=halo_size, color=[_halo_rgba(color_hex)] * len(sub)), hoverinfo="skip", showlegend=False
            ))

            # Ring
            fig_map.add_trace(go.Scattermapbox(
                lat=sub["latitude"], lon=sub["longitude"], mode="markers",
                marker=dict(size=ring_size, color="white", symbol="circle"), hoverinfo="skip", showlegend=False
            ))

            # Prepare hover info
            sub["location_display"] = sub["country"].fillna("—") + " / " + sub.get("region", pd.Series(["—"]*len(sub))).fillna("—")
            sub["date_display"] = sub["event date"].apply(_fmt) if "event date" in sub.columns else "—"
            sub["people_affected"] = sub.get("total affected", pd.Series([0]*len(sub))).apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
            sub["displaced"] = sub.get("no. homeless", pd.Series([0]*len(sub))).apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
            sub["deaths"] = sub.get("total deaths", pd.Series([0]*len(sub))).apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
            sub["economic_damage"] = sub.get("total damage ('000 us$')", pd.Series([0]*len(sub))).apply(lambda x: f"${x/1000:,.1f}K" if pd.notna(x) and x > 0 else "—")
            sub["severity_level"] = sub.get("alert level", pd.Series(["Unknown"]*len(sub)))
            sub["data_source"] = sub.get("source", pd.Series(["Unknown"]*len(sub)))
            sub["event_id"] = sub.get("id", pd.Series(["—"]*len(sub)))

            # Main
            fig_map.add_trace(go.Scattermapbox(
                lat=sub["latitude"],
                lon=sub["longitude"],
                mode="markers",
                marker=dict(size=main_size, color=color_hex, symbol="circle"),
                name=str(lvl),
                customdata=sub[[
                    "disaster type standardized",
                    "location_display",
                    "severity_level",
                    "date_display",
                    "people_affected",
                    "displaced",
                    "deaths",
                    "economic_damage",
                    "data_source",
                    "event_id"
                ]].astype(str),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Location: %{customdata[1]}<br>"
                    "Severity: %{customdata[2]}<br>"
                    "Date: %{customdata[3]}<br>"
                    "People Affected: %{customdata[4]}<br>"
                    "Displaced: %{customdata[5]}<br>"
                    "Deaths: %{customdata[6]}<br>"
                    "Economic Damage: %{customdata[7]}<br>"
                    "Data Source: %{customdata[8]}<br>"
                    "Event ID: %{customdata[9]}<extra></extra>"
                )
            ))

        center, zoom = _center_zoom_from_points(df_map["latitude"], df_map["longitude"])
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=560,
            hoverlabel=dict(font_size=14),
            legend_title_text="Disaster Type",
            uirevision=True,
            mapbox=dict(style="carto-positron", center=center, zoom=zoom),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No geolocation data available for the selected filters.")

    # -------------------------
    # 2. Total Disasters Over Time
    # -------------------------
    subsection_title("Total Disasters Over Time")
    if "start year" in df_filtered.columns:
        df_yearly = df_filtered.groupby("start year").size().reset_index(name="count")
        if not df_yearly.empty:
            fig_line = px.line(
                df_yearly,
                x="start year",
                y="count",
                markers=True,
                labels={"start year": "Year", "count": "Number of Disasters"},
                title="Number of Disasters per Year"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data for selected years.")

    # -------------------------
    # 3. Top 10 Disaster Types
    # -------------------------
    subsection_title("Top 10 Disaster Types")
    type_col = "disaster type standardized" if "disaster type standardized" in df_filtered.columns else "disaster type"
    if type_col in df_filtered.columns:
        df_types = df_filtered[type_col].value_counts().head(10).reset_index()
        df_types.columns = ["Disaster Type", "Count"]
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

    # Footer
    st.markdown("---")
    st.caption("Data source: EM-DAT & NASA EONET | Visualization: GioVision Dashboard")
