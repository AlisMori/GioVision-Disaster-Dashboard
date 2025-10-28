
# ======================================
# ENVIRONMENTAL OVERVIEW TAB
# ======================================

import os
import math
"""
Environmental Overview Tab

Displays key statistics and a high-level overview of global disaster data,
including a styled map consistent with the Alerts tab (Carto-Positron with
halo/ring/main markers).
"""

import os
import math
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.merge_datasets import merge_datasets  # Make sure this exists

# TODO implement the environmental overview page

# the following code is just the example of the analyses
# please do not make the same analyses here, but if you can it would be better if you used the same style for the map for the consistency 

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
# MAP HELPERS (same style as Alerts)
# ===========================
def _center_zoom_from_points(lat_series: pd.Series, lon_series: pd.Series):
    """Compute approximate (center, zoom) for Mapbox from bounds (no fitbounds)."""
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
# MAIN RENDER
# ===========================
def render():
    """Render the Environmental Overview tab."""
    # ---- OVERVIEW ----
    _anchor("sec-env-overview")
    section_title("Overview")
    st.markdown(
        "This section presents high-level environmental and disaster context, including a global map "
        "styled consistently with the Alerts page. Replace the placeholder points with your own summary "
        "locations (e.g., country centroids, recent hotspots, or representative monitoring sites)."
    )

    # ---- KEY STATS (placeholders) ----
    st.markdown("---")
    subsection_title("Key Statistics (Placeholder)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Countries Covered", "190+")
    with c2:
        st.metric("Events (YTD)", "2,430")
    with c3:
        st.metric("People Affected (YTD)", "18.2M")
    with c4:
        st.metric("Red-Level Share", "12%")

    # ---- MAP (styled like Alerts, but with neutral data) ----
    st.markdown("---")
    _anchor("sec-env-map")
    section_title("Global Situational Map")

    # Placeholder points: change this to your aggregated/overview dataset.
    # Required columns: Latitude, Longitude
    # Optional columns: Level (Red/Orange/Green/Unknown), Name, Country, Start, End, Severity
    overview_points = pd.DataFrame(
        [
            {"Name": "Andes Cluster",  "Country": "Peru",     "Latitude": -13.5, "Longitude": -71.9, "Level": "Orange", "Start": "2025-10-01", "End": None,        "Severity": "Moderate"},
            {"Name": "Mediterranean",  "Country": "Italy",    "Latitude":  41.9, "Longitude":  12.5, "Level": "Green",  "Start": "2025-09-20", "End": "2025-10-05","Severity": "Low"},
            {"Name": "South Asia Hub", "Country": "India",    "Latitude":  22.8, "Longitude":  78.9, "Level": "Red",    "Start": "2025-10-15", "End": None,        "Severity": "High"},
            {"Name": "SEA Watch",      "Country": "Philippines","Latitude": 12.9, "Longitude": 121.8, "Level": "Orange", "Start": "2025-10-18", "End": None,        "Severity": "Elevated"},
            {"Name": "East Africa",    "Country": "Kenya",    "Latitude":  -0.0, "Longitude":  37.9, "Level": "Unknown","Start": None,         "End": None,        "Severity": "—"},
        ]
    )

    # Build hover HTML similar to Alerts
    dfm = overview_points.copy()
    dfm["_start_dt"] = pd.to_datetime(dfm.get("Start"), errors="coerce", utc=True)
    dfm["_end_dt"]   = pd.to_datetime(dfm.get("End"),   errors="coerce", utc=True)

    name     = dfm.get("Name", "Location").fillna("Location")
    country  = dfm.get("Country", "—").fillna("—")
    level    = dfm.get("Level", "Unknown").fillna("Unknown")
    severity = dfm.get("Severity", "—").fillna("—")

    dfm["hover"] = (
        "<b>" + name + "</b><br>"
        + "Level: " + level + "<br>"
        + "Country: " + country + "<br>"
        + "Start: " + dfm["_start_dt"].map(_fmt) + "<br>"
        + "End: "   + dfm["_end_dt"].map(_fmt) + "<br>"
        + "Severity: " + severity
    )

    # Create figure with halo + ring + main markers, grouped by Level
    fig_map = go.Figure()
    main_size, ring_size, halo_size = 11, 14, 26

    for lvl in dfm.get("Level", "Unknown").fillna("Unknown").unique():
        sub = dfm[dfm["Level"].fillna("Unknown") == lvl]
        color_hex = ALERT_COLORS.get(lvl, ALERT_COLORS["Unknown"])

        # Halo (faint colored circle)
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
            marker=dict(size=halo_size, color=[_halo_rgba(color_hex)] * len(sub), opacity=1.0),
            hoverinfo="skip", showlegend=False,
        ))

        # Ring (white)
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
            marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ))

        # Main marker (solid)
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
            marker=dict(size=main_size, color=color_hex, opacity=0.95, symbol="circle"),
            name=str(lvl),
            customdata=sub[["hover"]],
            hovertemplate="%{customdata[0]}<extra></extra>",
        ))

    center, zoom = _center_zoom_from_points(dfm["Latitude"], dfm["Longitude"])

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
        hoverlabel=dict(font_size=16),
        legend_title_text="Level",
        uirevision=True,
        mapbox=dict(style="carto-positron", center=center, zoom=zoom),
    )

    st.caption(
        "Zoom and pan to explore context locations. Map styling mirrors the Alerts tab: "
        "Carto-Positron base, with halo/ring/main markers; levels follow the app palette."
    )
    st.plotly_chart(
        fig_map, use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )

    # ---- NEXT SECTIONS (placeholders) ----
    st.markdown("---")
    subsection_title("Global Notes (Placeholder)")
    st.markdown(
        "- Data coverage varies by source/time; interpret counts carefully.\n"
        "- Consider normalizing by population or exposure where appropriate.\n"
        "- Integrate climate or seasonal signals in future iterations."
    )

    st.markdown("---")
    st.caption("Sources: Your integrated feeds (e.g., EM-DAT, GDACS, EONET).")
