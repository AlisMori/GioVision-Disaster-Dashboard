"""
disaster_analysis.py
--------------------
Disaster Analysis page using EM-DAT.

Defaults:
- Scope: Global
- Time range: 2022–2025 (changeable in the sidebar + per-visual)
- Dataset: ../data/processed/emdat_cleaned.csv

Visuals:
1) Choropleth: total number of disasters per country (hover + country zoom + type filter)
2) Top-10 disasters by frequency (Bar / Pie with 'Others' aggregation)
3) Stacked area timeline of disasters (by Disaster Type, Top-5 + "Others")
4) Severity analysis (single disaster type; numeric parsing; compare across Countries or Years)
5) Density map & heatmap: historical concentration (zoomable, type filter, new color scale)
6) Calendar heatmap (Year–Month) + optional Lat/Lon density heatmap

Style:
- gv-section-title / gv-subsection-title bars
- In-page anchors ?page=Analysis&section=...
- Carto-Positron maps; cohesive blues for categories; effective gradient for intensities
"""

from __future__ import annotations
import os
import sys
import math
import re
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly import graph_objects as go

# =========================
# CONFIG / PATHS (portable)
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

EMDAT_PATHS = [
    "data/processed/emdat_cleaned.csv",       # run from project root
    "../data/processed/emdat_cleaned.csv",    # run from dashboard/
    "../../data/processed/emdat_cleaned.csv", # run from dashboard/components/
    "dashboard/data/processed/emdat_cleaned.csv",
]

def _first_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            pass
    return None

DEFAULT_START = 2022
DEFAULT_END   = 2025

# Palettes
TYPE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]  # more distinct than before
AREA_TOP5_PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]  # for stacked area top5
OTHERS_COLOR = "#9e9e9e"

INTENSITY_SCALE = "YlOrRd"    # choropleth / calendar
HEAT_SCALE_MAP  = "Inferno"   # map density heat (new: calmer lows, stronger highs)
HEAT_SCALE_XY   = "Viridis"   # 2D lat/lon heat

REQUIRED_COLS = [
    "DisNo.", "Event Name", "Country", "Region", "Location",
    "Start Year", "Start Month", "Start Day", "Event Date",
    "Disaster Type", "Disaster Type Standardized", "Latitude", "Longitude",
    "Total Deaths", "No. Injured", "No. Affected", "No. Homeless",
    "Total Affected", "Total Damage ('000 US$)"
]

# =========================
# THEME HELPERS
# =========================
def _anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

# Unified Plotly config (hover/zoom on by default)
PLOTLY_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d","select2d"], "scrollZoom": True}

# =========================
# DATA LOADING / PREP
# =========================
@st.cache_data(show_spinner=False)
def load_emdat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Dates / numeric
    if "Event Date" in df.columns:
        df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")
    if "Start Year" in df.columns:
        df["Start Year"] = pd.to_numeric(df["Start Year"], errors="coerce")
    for c in ["Start Month", "Start Day"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Totals if missing
    if "Total Affected" not in df.columns:
        for c in ["Total Deaths", "No. Injured", "No. Affected", "No. Homeless"]:
            if c not in df.columns: df[c] = 0
        df["Total Affected"] = df["No. Injured"].fillna(0) + df["No. Affected"].fillna(0) + df["No. Homeless"].fillna(0)

    # Coerce lat/lon
    for c in ["Latitude", "Longitude"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean a Severity column if it exists (text → integers where possible)
    if "Severity" in df.columns:
        df["Severity_raw"] = df["Severity"].astype(str)
        df["Severity"] = pd.to_numeric(
            df["Severity_raw"].str.extract(r"(-?\d+)", expand=False), errors="coerce"
        )

    return df

def _apply_filters(df: pd.DataFrame, years: Tuple[int,int], country: str, dtype: str) -> pd.DataFrame:
    lo, hi = years
    base = df.copy()

    # Time (Start Year preferred, fallback Event Date)
    if "Start Year" in base.columns and base["Start Year"].notna().any():
        base = base[(base["Start Year"] >= lo) & (base["Start Year"] <= hi)]
    elif "Event Date" in base.columns and base["Event Date"].notna().any():
        base = base[(base["Event Date"].dt.year >= lo) & (base["Event Date"].dt.year <= hi)]

    # Country
    if country and country != "Global":
        base = base[base["Country"] == country]

    # Type
    if dtype and dtype != "All":
        base = base[base["Disaster Type"] == dtype]

    return base

def _country_list(df: pd.DataFrame) -> List[str]:
    countries = sorted([c for c in df["Country"].dropna().astype(str).unique().tolist() if c.strip()])
    return ["Global"] + countries

def _type_list(df: pd.DataFrame) -> List[str]:
    dtypes = sorted([t for t in df["Disaster Type"].dropna().astype(str).unique().tolist() if t.strip()])
    return ["All"] + dtypes

def _center_zoom_from_points(lat: pd.Series, lon: pd.Series) -> Tuple[dict, float]:
    lats = pd.to_numeric(lat, errors="coerce").dropna()
    lons = pd.to_numeric(lon, errors="coerce").dropna()
    if len(lats) == 0 or len(lons) == 0:
        return dict(lat=0, lon=0), 1.3
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    center = dict(lat=(lat_min + lat_max) / 2.0, lon=(lon_min + lon_max) / 2.0)
    lat_span = max(1e-6, lat_max - lat_min)
    lon_span = max(1e-6, lon_max - lon_min)
    k = 1.4
    zoom_from_lon = math.log2(360.0 / (lon_span * k))
    zoom_from_lat = math.log2(180.0 / (lat_span * k))
    zoom = max(1.0, min(zoom_from_lon, zoom_from_lat))
    return center, min(7.5, zoom)

def _top_n_with_others(df: pd.DataFrame, label_col: str, n: int = 5) -> pd.DataFrame:
    if df.empty: return df
    counts = df.groupby(label_col, as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
    counts = counts.sort_values("Count", ascending=False)
    if len(counts) <= n: return counts
    top = counts.head(n).copy()
    others = counts.iloc[n:]["Count"].sum()
    top = pd.concat([top, pd.DataFrame({label_col: ["Others"], "Count": [others]})], ignore_index=True)
    return top

# =========================
# PAGE RENDER
# =========================
def render():
    # Sidebar (global filters drive defaults of every visual)
    st.sidebar.header("Analysis Filters")

    emdat_used_path = _first_existing_path(EMDAT_PATHS)
    if not emdat_used_path:
        st.sidebar.error(
            "Could not find `emdat_cleaned.csv` in:\n"
            "- data/processed/\n- ../data/processed/\n- ../../data/processed/\n- dashboard/data/processed/\n"
        )
        st.stop()

    df = load_emdat(emdat_used_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.sidebar.error(f"EM-DAT file is missing columns: {', '.join(missing)}")
        st.stop()

    # Year bounds
    if "Start Year" in df and df["Start Year"].notna().any():
        min_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().min())
        max_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().max())
    else:
        min_year, max_year = 1970, 2025

    years_global = st.sidebar.slider(
        "Year range (global)",
        min_value=min_year, max_value=max_year,
        value=(max(DEFAULT_START, min_year), min(DEFAULT_END, max_year)), step=1
    )
    country_global = st.sidebar.selectbox("Country (global)", _country_list(df), index=0)
    type_global    = st.sidebar.selectbox("Disaster Type (global)", _type_list(df), index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Jump to section")
    anchor_map = {
        "Overview": "sec-da-overview",
        "Map: Total Disasters per Country": "sec-da-map-country",
        "Top-10 by Frequency": "sec-da-top10",
        "Stacked Area Timeline": "sec-da-timeline",
        "Severity Analysis": "sec-da-severity",
        "Concentration Heat & Map": "sec-da-concentration",
        "Calendar / Lat-Lon Heat": "sec-da-calendar",
    }
    sec_choice = st.sidebar.radio("", list(anchor_map.keys()), index=0)
    st.sidebar.markdown(f"[Go ▶](#{anchor_map[sec_choice]})", unsafe_allow_html=True)

    # ----------------- Overview -----------------
    _anchor("sec-da-overview")
    section_title("Disaster Analysis (EM-DAT)")
    st.markdown(
        f"Global defaults apply unless overridden in each visual. Current global: "
        f"**{years_global[0]}–{years_global[1]}**, **{country_global}**, **{type_global}**."
    )
    try:
        st.caption(f"EM-DAT file: `{os.path.relpath(emdat_used_path)}`")
    except Exception:
        st.caption(f"EM-DAT file: `{emdat_used_path}`")

    # 1) Choropleth
    st.markdown("---")
    _anchor("sec-da-map-country")
    section_title("Map: Total Disasters per Country")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years1 = st.slider("Year range", min_year, max_year, value=years_global, key="years_1")
        with c2:
            country1 = st.selectbox("Country", _country_list(df), index=_country_list(df).index(country_global), key="country_1")
        with c3:
            type1 = st.selectbox("Disaster Type", _type_list(df), index=_type_list(df).index(type_global), key="type_1")

    # Apply exactly the chosen scope (no out-of-range or different country leaks)
    d1 = _apply_filters(df, years1, "Global" if country1 == "Global" else country1, type1)
    if country1 == "Global":
        agg1 = d1.groupby("Country", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Events"})
    else:
        # For single country, show its value and keep choropleth zoomed
        agg1 = d1.groupby("Country", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Events"})
        agg1 = agg1[agg1["Country"] == country1]

    if agg1.empty:
        st.info("No data for the selected filters.")
    else:
        fig_chor = px.choropleth(
            agg1, locations="Country", locationmode="country names",
            color="Events", color_continuous_scale=INTENSITY_SCALE
        )
        fig_chor.update_layout(
            coloraxis_colorbar=dict(title="Total Events"),
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", fitbounds="locations")
        )
        st.plotly_chart(fig_chor, use_container_width=True, config=PLOTLY_CFG)
        st.caption("Hover a country to see total events. Global filters and the selectors above fully control the scope.")

    # 2) Top-10 frequency
    st.markdown("---")
    _anchor("sec-da-top10")
    section_title("Top-10 Disasters by Frequency")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years2 = st.slider("Year range", min_year, max_year, value=years_global, key="years_2")
        with c2:
            country2 = st.selectbox("Country", _country_list(df), index=_country_list(df).index(country_global), key="country_2")
        with c3:
            chart2 = st.radio("Chart", ["Bar", "Pie"], horizontal=True, key="chart_2")

    d2 = _apply_filters(df, years2, country2, "All")
    freq = d2.groupby("Disaster Type", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
    freq = freq.sort_values("Count", ascending=False)

    # Ensure we only show data in scope; then top10 + Others
    if len(freq) > 10:
        top10 = freq.head(10).copy()
        others_count = int(freq["Count"].iloc[10:].sum())
        top10 = pd.concat([top10, pd.DataFrame({"Disaster Type": ["Others"], "Count": [others_count]})], ignore_index=True)
    else:
        top10 = freq

    if top10.empty:
        st.info("No data for the selected filters.")
    else:
        if chart2 == "Bar":
            fig_bar = px.bar(
                top10, x="Count", y="Disaster Type", orientation="h",
                color="Disaster Type", color_discrete_sequence=TYPE_PALETTE, text="Count"
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, bargap=0.25, showlegend=False)
            fig_bar.update_traces(textposition="outside", cliponaxis=False, hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>")
            st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CFG)
        else:
            fig_pie = px.pie(
                top10, names="Disaster Type", values="Count", hole=0.3,
                color="Disaster Type", color_discrete_sequence=TYPE_PALETTE
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG)

    # 3) Stacked area timeline (Top-5 + Others)
    st.markdown("---")
    _anchor("sec-da-timeline")
    section_title("Stacked Area Timeline (by Disaster Type – Top-5 + Others)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            years3 = st.slider("Year range", min_year, max_year, value=years_global, key="years_3")
        with c2:
            country3 = st.selectbox("Country", _country_list(df), index=_country_list(df).index(country_global), key="country_3")

    d3 = _apply_filters(df, years3, country3, "All")
    # Year-Month index
    if "Event Date" in d3.columns and d3["Event Date"].notna().any():
        ym = d3["Event Date"].dt.to_period("M").astype(str)
    else:
        sy = pd.to_numeric(d3["Start Year"], errors="coerce")
        sm = pd.to_numeric(d3["Start Month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
        ym = pd.to_datetime(dict(year=sy, month=sm, day=1), errors="coerce").dt.to_period("M").astype(str)

    d3a = d3.assign(YearMonth=ym)

    # Build Top-5 + Others labeling
    top_counts = _top_n_with_others(d3a, "Disaster Type", n=5)
    top_labels = set(top_counts["Disaster Type"].tolist())
    d3a["Type_6"] = d3a["Disaster Type"].where(d3a["Disaster Type"].isin(top_labels), "Others")

    area = d3a.groupby(["YearMonth", "Type_6"], as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
    if area.empty:
        st.info("No data for the selected filters.")
    else:
        area["YearMonth_dt"] = pd.to_datetime(area["YearMonth"], errors="coerce")
        area = area.sort_values("YearMonth_dt")
        # color map: five distinct + grey for Others
        unique_types = [t for t in area["Type_6"].unique() if t != "Others"]
        color_map = {t: AREA_TOP5_PALETTE[i % len(AREA_TOP5_PALETTE)] for i, t in enumerate(sorted(unique_types))}
        color_map["Others"] = OTHERS_COLOR

        fig_area = px.area(area, x="YearMonth_dt", y="Count", color="Type_6", color_discrete_map=color_map)
        fig_area.update_layout(
            xaxis_title="Date", yaxis_title="Number of Events",
            legend_title="Disaster Type (Top-5 + Others)", hovermode="x unified"
        )
        st.plotly_chart(fig_area, use_container_width=True, config=PLOTLY_CFG)

    # 4) Severity analysis (single disaster type, cleaned numeric, compare across Countries/Years)
    st.markdown("---")
    _anchor("sec-da-severity")
    section_title("Severity Analysis (per Disaster Type)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years4 = st.slider("Year range", min_year, max_year, value=years_global, key="years_4")
        with c2:
            # select exactly one type to avoid cross-type comparison
            types_all = [t for t in _type_list(df) if t != "All"]
            dtype4 = st.selectbox("Disaster Type (required)", types_all, index=types_all.index(type_global) if (type_global in types_all) else 0)
        with c3:
            compare_dim = st.radio("Compare by", ["Country","Year"], horizontal=True, key="sev_dim")

    # Choose metric: If 'Severity' exists use it; else default to Total Affected (cleaned numeric)
    metric_candidates = ["Severity", "Total Deaths", "No. Injured", "No. Affected", "Total Affected", "Total Damage ('000 US$)"]
    existing_metrics = [m for m in metric_candidates if m in df.columns]
    metric4 = st.selectbox("Metric", existing_metrics, index=existing_metrics.index("Severity") if "Severity" in existing_metrics else existing_metrics.index("Total Affected"))

    d4 = _apply_filters(df, years4, "Global", dtype4).copy()
    # Strictly keep only rows in scope; parse numeric for text metrics
    if metric4 == "Severity":
        if "Severity" in d4.columns:
            # already parsed during load; ensure numeric
            d4["Severity"] = pd.to_numeric(d4["Severity"], errors="coerce")
    else:
        d4[metric4] = pd.to_numeric(d4[metric4], errors="coerce")
    d4 = d4.dropna(subset=[metric4])

    if d4.empty:
        st.info("No data for the selected filters / selections.")
    else:
        # build comparison key
        if compare_dim == "Country":
            key = "Country"
        else:
            if "Event Date" in d4.columns and d4["Event Date"].notna().any():
                d4["Year"] = d4["Event Date"].dt.year
            else:
                d4["Year"] = pd.to_numeric(d4["Start Year"], errors="coerce")
            key = "Year"

        tab_box, tab_violin = st.tabs(["Box Plot", "Violin Plot"])
        with tab_box:
            fig_box = px.box(
                d4, x=key, y=metric4, points="suspectedoutliers",
                color=key if d4[key].nunique() <= 12 else None
            )
            fig_box.update_layout(xaxis_title=key, yaxis_title=metric4, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CFG)
        with tab_violin:
            fig_vio = px.violin(
                d4, x=key, y=metric4, box=True, points=False,
                color=key if d4[key].nunique() <= 12 else None
            )
            fig_vio.update_layout(xaxis_title=key, yaxis_title=metric4, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True, config=PLOTLY_CFG)

    # 5) Concentration heat & map (density + scatter; new color scale; zoom + hover)
    st.markdown("---")
    _anchor("sec-da-concentration")
    section_title("Concentration Heat & Map (Historical)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years5 = st.slider("Year range", min_year, max_year, value=years_global, key="years_5")
        with c2:
            country5 = st.selectbox("Country", _country_list(df), index=_country_list(df).index(country_global), key="country_5")
        with c3:
            type5 = st.selectbox("Disaster Type", _type_list(df), index=_type_list(df).index(type_global), key="type_5")

    d5 = _apply_filters(df, years5, country5, type5).dropna(subset=["Latitude", "Longitude"])
    if d5.empty:
        st.info("No geolocated events for the selected filters.")
    else:
        center, zoom = _center_zoom_from_points(d5["Latitude"], d5["Longitude"])
        fig_density = px.density_mapbox(
            d5, lat="Latitude", lon="Longitude", radius=18,
            center=center, zoom=zoom, mapbox_style="carto-positron",
            color_continuous_scale=HEAT_SCALE_MAP
        )
        fig_density.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540)
        # Add points with hover
        fig_scatter = go.Figure(fig_density)
        fig_scatter.add_trace(go.Scattermapbox(
            lat=d5["Latitude"], lon=d5["Longitude"], mode="markers",
            marker=dict(size=7, opacity=0.85),
            text=d5["Event Name"],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.2f}, Lon: %{lon:.2f}<extra></extra>"
        ))
        st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CFG)
        st.caption("Soft colors = fewer events; stronger colors = higher concentration. Zoom and pan to inspect clusters.")

    # 6) Calendar / Lat-Lon heat
    st.markdown("---")
    _anchor("sec-da-calendar")
    section_title("Temporal & Spatial Heat")

    tab_cal, tab_xy = st.tabs(["Calendar Heatmap (Year–Month)", "Lat/Lon Density Heat"])

    with tab_cal:
        with st.expander("Selections (this visual)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                years6 = st.slider("Year range", min_year, max_year, value=years_global, key="years_6")
            with c2:
                country6 = st.selectbox("Country", _country_list(df), index=_country_list(df).index(country_global), key="country_6")

        d6 = _apply_filters(df, years6, country6, "All")
        if "Event Date" in d6.columns and d6["Event Date"].notna().any():
            ym6 = d6["Event Date"].dt.to_period("M").astype(str)
        else:
            sy = pd.to_numeric(d6["Start Year"], errors="coerce")
            sm = pd.to_numeric(d6["Start Month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
            ym6 = pd.to_datetime(dict(year=sy, month=sm, day=1), errors="coerce").dt.to_period("M").astype(str)

        cal = d6.assign(YearMonth=ym6)
        cal["Year"] = pd.to_numeric(cal["YearMonth"].str[:4], errors="coerce")
        cal["Month"] = pd.to_numeric(cal["YearMonth"].str[5:7], errors="coerce")
        heat = cal.groupby(["Year", "Month"], as_index=False)["DisNo."].count().rename(columns={"DisNo.":"Count"})
        heat = heat.dropna(subset=["Year","Month"])
        if heat.empty:
            st.info("No data for the selected filters.")
        else:
            heat = heat.sort_values(["Year", "Month"])
            pivot = heat.pivot(index="Year", columns="Month", values="Count").fillna(0)
            fig_heat = px.imshow(pivot, color_continuous_scale=INTENSITY_SCALE, aspect="auto", origin="lower")
            fig_heat.update_layout(coloraxis_colorbar=dict(title="Events"))
            st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CFG)

    with tab_xy:
        with st.expander("Selections (this visual)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                years7 = st.slider("Year range", min_year, max_year, value=years_global, key="years_7")
            with c2:
                country7 = st.selectbox("Country", _country_list(df), index=_country_list(df).index(country_global), key="country_7")
            with c3:
                type7 = st.selectbox("Disaster Type", _type_list(df), index=_type_list(df).index(type_global), key="type_7")

        d7 = _apply_filters(df, years7, country7, type7).dropna(subset=["Latitude", "Longitude"])
        if d7.empty:
            st.info("No geolocated data for the selected filters.")
        else:
            fig_xy = px.density_heatmap(
                d7, x="Longitude", y="Latitude", nbinsx=40, nbinsy=40, color_continuous_scale=HEAT_SCALE_XY
            )
            fig_xy.update_layout(xaxis_title="Longitude", yaxis_title="Latitude", coloraxis_colorbar=dict(title="Density"))
            st.plotly_chart(fig_xy, use_container_width=True, config=PLOTLY_CFG)

    # Footer
    st.markdown("---")
    st.caption("Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
