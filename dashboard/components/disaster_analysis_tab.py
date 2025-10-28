"""
disaster_analysis.py
--------------------
Disaster Analysis page using EM-DAT.

Global controls (sidebar):
- Year range (global)
- Region (global)
- Country (global; constrained by Region + Year range)

Each visual:
- Defaults to the global controls
- Local selectors (if any) are constrained to EXISTING values under that visual's scope
- Changing sidebar controls immediately affects ALL visuals

Style:
- gv-section-title / gv-subsection-title bars
- In-page anchors ?page=Analysis&section=...
- Carto-Positron maps; cohesive single-hue theme (reds)
"""

from __future__ import annotations
import os
import sys
import math
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

DEFAULT_START = 2022
DEFAULT_END   = 2025
TYPE_COL = "Disaster Type"

# =========================
# THEME — clean reds (no purple/brown)
# =========================
RED_PALETTE = ["#FEE5D9", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
TYPE_PALETTE      = RED_PALETTE
AREA_TOP5_PALETTE = ["#CB181D", "#EF3B2C", "#FB6A4A", "#FC9272", "#A50F15"]
OTHERS_COLOR      = "#9E9E9E"

INTENSITY_SCALE   = "Reds"   # choropleth / calendar heat
HEAT_SCALE_MAP    = "Reds"   # density mapbox
HEAT_SCALE_XY     = "Reds"   # 2D density heat
LINE_COLOR        = "#CB181D"
LINE_ACCENT       = "#A50F15"
BAR_COLOR         = "#EF3B2C"

REQUIRED_COLS = [
    "DisNo.", "Event Name", "Country", "Region", "Location",
    "Start Year", "Start Month", "Start Day", "Event Date",
    "Disaster Type", "Disaster Type Standardized", "Latitude", "Longitude",
    "Total Deaths", "No. Injured", "No. Affected", "No. Homeless",
    "Total Affected", "Total Damage ('000 US$)"
]

# Plotly configs
PLOTLY_CFG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "scrollZoom": True,
}
PLOTLY_CFG_NOZOOM = {
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "zoom", "pan", "zoomIn2d", "zoomOut2d",
        "autoScale2d", "resetScale2d",
        "select2d", "lasso2d",
        "zoom2d"
    ],
}

# =========================
# THEME HELPERS
# =========================
def _anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

# =========================
# DATA LOADING / PREP
# =========================
def _first_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            pass
    return None

def _first_country_only(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "Unknown"
    s = s.replace(" & ", ",").replace(" and ", ",").replace("/", ",").replace(";", ",")
    return s.split(",")[0].strip() or "Unknown"

@st.cache_data(show_spinner=False)
def load_emdat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Dates / numeric parsing
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

    # Optional Severity cleaning
    if "Severity" in df.columns:
        df["Severity_raw"] = df["Severity"].astype(str)
        df["Severity"] = pd.to_numeric(
            df["Severity_raw"].str.extract(r"(-?\d+)", expand=False), errors="coerce"
        )

    # Epicenter-only country (display)
    if "Country" in df.columns:
        df["Country_all"] = df["Country"].astype(str)
        df["Country"] = df["Country_all"].apply(_first_country_only)

    return df

# ---------- availability helpers ----------
def _filter_by_years(df: pd.DataFrame, years: Tuple[int,int]) -> pd.DataFrame:
    lo, hi = years
    if "Start Year" in df.columns and df["Start Year"].notna().any():
        return df[(df["Start Year"] >= lo) & (df["Start Year"] <= hi)]
    if "Event Date" in df.columns and df["Event Date"].notna().any():
        return df[(df["Event Date"].dt.year >= lo) & (df["Event Date"].dt.year <= hi)]
    return df

def _available_regions(df: pd.DataFrame, years: Tuple[int,int]) -> List[str]:
    d = _filter_by_years(df, years)
    regs = sorted([r for r in d["Region"].dropna().astype(str).unique() if r.strip()])
    return ["All Regions"] + regs

def _available_countries(df: pd.DataFrame, years: Tuple[int,int], region: str) -> List[str]:
    d = _filter_by_years(df, years)
    if region and region != "All Regions":
        d = d[d["Region"] == region]
    countries = sorted([c for c in d["Country"].dropna().astype(str).unique() if c.strip()])
    return ["Global"] + countries

def _available_types(df: pd.DataFrame, years: Tuple[int,int], region: str, country: str, type_col: str) -> List[str]:
    d = _filter_by_years(df, years)
    if region and region != "All Regions":
        d = d[d["Region"] == region]
    if country and country != "Global":
        d = d[d["Country"] == country]
    if type_col not in d.columns:
        return ["All"]
    dtypes = sorted([t for t in d[type_col].dropna().astype(str).unique() if t.strip()])
    return ["All"] + dtypes if dtypes else ["All"]

def _coerce_choice(current: Optional[str], options: List[str], fallback_idx: int = 0) -> str:
    if not options:
        return ""
    if current in options:
        return current
    return options[min(fallback_idx, len(options)-1)]

def _apply_scope(df: pd.DataFrame, years: Tuple[int,int], region: str, country: str) -> pd.DataFrame:
    d = _filter_by_years(df, years)
    if region and region != "All Regions":
        d = d[d["Region"] == region]
    if country and country != "Global":
        d = d[d["Country"] == country]
    return d

def constrained_type_selector(
    df: pd.DataFrame,
    label_key_prefix: str,
    years_sel: Tuple[int,int],
    region_sel: str,
    country_sel: str,
    type_col: str,
) -> str:
    type_options = _available_types(df, years_sel, region_sel, country_sel, type_col)
    key_state = f"{label_key_prefix}_type"
    prev = st.session_state.get(key_state, "All")
    coerced = _coerce_choice(prev, type_options, 0)
    return st.selectbox("Disaster Type", type_options, index=type_options.index(coerced), key=key_state)

# ---- Geo helpers ----
REGION_BBOX = {
    "Africa":   (-35.0, 37.5,  -20.0,  52.0),
    "Asia":     ( -5.0, 82.0,  25.0, 180.0),
    "Europe":   ( 35.0, 72.5, -25.0,  45.0),
    "Americas": (-57.0, 72.0, -170.0, -30.0),
    "Oceania":  (-50.0, 10.0,  105.0, 180.0),
}

def _clip_to_region_bbox(d: pd.DataFrame, region: str) -> pd.DataFrame:
    if not region or region == "All Regions" or region not in REGION_BBOX:
        return d
    lat_min, lat_max, lon_min, lon_max = REGION_BBOX[region]
    dd = d.copy()
    dd["Latitude"]  = pd.to_numeric(dd["Latitude"], errors="coerce")
    dd["Longitude"] = pd.to_numeric(dd["Longitude"], errors="coerce")
    return dd[(dd["Latitude"].between(lat_min, lat_max)) & (dd["Longitude"].between(lon_min, lon_max))]

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

# =========================
# PAGE RENDER
# =========================
def render():
    # ---------- Load data ----------
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

    # ---------- after: df = load_emdat(emdat_used_path) ----------
    # Country centroids from all geocoded rows in the dataset (median to resist outliers)
    country_geo_centroids = (
        df.dropna(subset=["Latitude", "Longitude"])
        .groupby("Country", as_index=False)[["Latitude", "Longitude"]]
        .median()
        .rename(columns={"Latitude": "CentroidLat", "Longitude": "CentroidLon"})
    )

    def _region_centroid(region: str) -> tuple[float, float]:
        """Return simple centroid of region bbox as a fallback."""
        if region in REGION_BBOX:
            lat_min, lat_max, lon_min, lon_max = REGION_BBOX[region]
            return ( (lat_min + lat_max)/2.0, (lon_min + lon_max)/2.0 )
        # world fallback
        return (0.0, 0.0)

    # ---------- Dataset bounds ----------
    if "Start Year" in df and df["Start Year"].notna().any():
        min_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().min())
        max_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().max())
    else:
        min_year, max_year = 1970, 2025

    # ---------- Global state ----------
    if "glob_years" not in st.session_state:
        st.session_state["glob_years"] = (max(DEFAULT_START, min_year), min(DEFAULT_END, max_year))
    if "glob_region" not in st.session_state:
        st.session_state["glob_region"] = "All Regions"
    if "glob_country" not in st.session_state:
        st.session_state["glob_country"] = "Global"

    # ---------- Sidebar (GLOBAL CONTROLS) ----------
    st.sidebar.header("Analysis Filters")

    years_global = st.sidebar.slider(
        "Year range (global)",
        min_value=min_year,
        max_value=max_year,
        value=(
            max(min_year, st.session_state["glob_years"][0]),
            min(max_year, st.session_state["glob_years"][1]),
        ),
        step=1,
        key="__years_slider__",
    )
    st.session_state["glob_years"] = years_global

    region_options = _available_regions(df, st.session_state["glob_years"])
    region_val = _coerce_choice(st.session_state["glob_region"], region_options, 0)
    region_global = st.sidebar.selectbox(
        "Region (global)",
        options=region_options,
        index=region_options.index(region_val),
        key="__region_select__",
    )
    st.session_state["glob_region"] = region_global

    country_options = _available_countries(df, st.session_state["glob_years"], st.session_state["glob_region"])
    country_val = _coerce_choice(st.session_state["glob_country"], country_options, 0)
    country_global = st.sidebar.selectbox(
        "Country (global)",
        options=country_options,
        index=country_options.index(country_val),
        key="__country_select__",
    )
    st.session_state["glob_country"] = country_global

    # ---------- Overview ----------
    _anchor("sec-da-overview")
    section_title("Overview")
    st.markdown(
        "This page explores EM-DAT disaster records across time and space. "
        "Use the global filters in the sidebar; each visual also offers scoped selectors."
    )

    # short wrapper to pass df consistently
    def type_sel(prefix: str, years_sel: Tuple[int,int], region_sel: str, country_sel: str) -> str:
        return constrained_type_selector(df, prefix, years_sel, region_sel, country_sel, TYPE_COL)

    # ======================
    # 1) Choropleth (total per country) — now with Disaster Type filter
    # ======================
    st.markdown("---")
    _anchor("sec-da-map-country")
    section_title("Map: Total Disasters per Country")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            years1 = st.slider(
                "Year range",
                min_year, max_year,
                value=st.session_state["glob_years"],
                key="years_1",
            )
        with c2:
            region_opts_1 = _available_regions(df, years1)
            region1 = st.selectbox(
                "Region",
                options=region_opts_1,
                index=region_opts_1.index(_coerce_choice(st.session_state["glob_region"], region_opts_1, 0)),
                key="region_1",
            )
        with c3:
            country_opts_1 = _available_countries(df, years1, region1)
            country1 = st.selectbox(
                "Country",
                options=country_opts_1,
                index=country_opts_1.index(_coerce_choice(st.session_state["glob_country"], country_opts_1, 0)),
                key="country_1",
            )
        with c4:
            # Disaster Type constrained by (years1, region1, country1)
            type1 = constrained_type_selector(
                df, "map1", years1, region1, country1, TYPE_COL
            )

    # Apply scope + (optional) type filter
    d1 = _apply_scope(df, years1, region1, country1)
    if type1 and type1 != "All" and TYPE_COL in d1.columns:
        d1 = d1[d1[TYPE_COL] == type1]

    # Aggregate to country
    agg1 = d1.groupby("Country", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Events"})
    if country1 != "Global":
        agg1 = agg1[agg1["Country"] == country1]

    if not agg1.empty:
        fig_chor = px.choropleth(
            agg1, locations="Country", locationmode="country names",
            color="Events", color_continuous_scale=INTENSITY_SCALE
        )
        fig_chor.update_layout(
            coloraxis_colorbar=dict(title="Total Events"),
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", fitbounds="locations")
        )
        st.plotly_chart(fig_chor, use_container_width=True, config=PLOTLY_CFG)
        st.caption("Filtered by year/region/country and disaster type.")

    # ======================
    # 2) Top-10 frequency (with Others hover) — ONE-ROW SELECTORS
    # ======================
    st.markdown("---")
    _anchor("sec-da-top10")
    section_title("Top-10 Disasters by Frequency")

    with st.expander("Selections (this visual)", expanded=True):
        # Keep all three controls on one row
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            years2 = st.slider(
                "Year range",
                min_year, max_year,
                value=st.session_state["glob_years"],
                key="years_2",
            )

        with c2:
            region_opts_2 = _available_regions(df, years2)
            region2 = st.selectbox(
                "Region",
                options=region_opts_2,
                index=region_opts_2.index(_coerce_choice(st.session_state["glob_region"], region_opts_2, 0)),
                key="region_2",
            )

        with c3:
            country_opts_2 = _available_countries(df, years2, region2)
            country2 = st.selectbox(
                "Country",
                options=country_opts_2,
                index=country_opts_2.index(_coerce_choice(st.session_state["glob_country"], country_opts_2, 0)),
                key="country_2",
            )

    # ---- Data prep
    d2 = _apply_scope(df, years2, region2, country2)
    freq = (
        d2.groupby(TYPE_COL, as_index=False)["DisNo."]
        .count()
        .rename(columns={"DisNo.": "Count"})
    ).sort_values("Count", ascending=False)

    if freq.empty:
        st.info("No data for the selected filters.")
    else:
        # Build Top-10 + Others and a readable hover for the remainder
        top10 = freq.head(10).copy()
        others_detail = "—"
        if len(freq) > 10:
            tail = freq.iloc[10:].copy().sort_values("Count", ascending=False)
            others_count = int(tail["Count"].sum())
            items = [f"{t}: {int(c)}" for t, c in zip(tail[TYPE_COL], tail["Count"])]
            if len(items) > 15:
                items = items[:15] + [f"+{len(tail)-15} more"]
            others_detail = "; ".join(items) if items else "—"
            top10 = pd.concat(
                [top10, pd.DataFrame({TYPE_COL: ["Others"], "Count": [others_count]})],
                ignore_index=True
            )

        # Attach hover text for Others row
        top10["customdata"] = np.where(
            top10[TYPE_COL] == "Others",
            "<br><i>Others:</i> " + others_detail,
            ""
        )

        # ---- Tabs: Bar / Pie
        tab_bar, tab_pie = st.tabs(["Bar Chart", "Pie Chart"])

        # BAR
        with tab_bar:
            fig_bar = px.bar(
                top10,
                x="Count",
                y=TYPE_COL,
                orientation="h",
                color=TYPE_COL,
                color_discrete_sequence=TYPE_PALETTE,
                text="Count",
            )
            fig_bar.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Count: %{x:,}%{customdata}<extra></extra>",
                customdata=top10["customdata"],
            )
            fig_bar.update_layout(
                yaxis={"categoryorder": "total ascending"},
                bargap=0.25,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

        # PIE
        with tab_pie:
            fig_pie = px.pie(
                top10,
                names=TYPE_COL,
                values="Count",
                hole=0.3,
                color=TYPE_COL,
                color_discrete_sequence=TYPE_PALETTE,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}%{customdata}<extra></extra>",
                customdata=top10["customdata"],
            )
            st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

    # ======================
    # 3) Stacked area timeline (Top-5 + Others with hover)
    # ======================
    st.markdown("---")
    _anchor("sec-da-timeline")
    section_title("Stacked Area Timeline (by Disaster Type – Top-5 + Others)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years3 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_3")
        with c2:
            region_opts_3 = _available_regions(df, years3)
            region3 = st.selectbox("Region", options=region_opts_3,
                                   index=region_opts_3.index(_coerce_choice(st.session_state["glob_region"], region_opts_3, 0)),
                                   key="region_3")
        with c3:
            country_opts_3 = _available_countries(df, years3, region3)
            country3 = st.selectbox("Country", options=country_opts_3,
                                    index=country_opts_3.index(_coerce_choice(st.session_state["glob_country"], country_opts_3, 0)),
                                    key="country_3")

    d3 = _apply_scope(df, years3, region3, country3)

    if "Event Date" in d3.columns and d3["Event Date"].notna().any():
        ym = d3["Event Date"].dt.to_period("M").astype(str)
    else:
        sy = pd.to_numeric(d3["Start Year"], errors="coerce")
        sm = pd.to_numeric(d3["Start Month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
        ym = pd.to_datetime(dict(year=sy, month=sm, day=1), errors="coerce").dt.to_period("M").astype(str)

    d3a = d3.assign(YearMonth=ym)

    def _top_n_with_others(df_in: pd.DataFrame, label_col: str, n: int = 5) -> pd.DataFrame:
        if df_in.empty: return df_in
        counts = df_in.groupby(label_col, as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
        counts = counts.sort_values("Count", ascending=False)
        if len(counts) <= n: return counts
        top = counts.head(n).copy()
        others = counts.iloc[n:]["Count"].sum()
        top = pd.concat([top, pd.DataFrame({label_col: ["Others"], "Count": [others]})], ignore_index=True)
        return top

    top_counts = _top_n_with_others(d3a, TYPE_COL, n=5)
    top_labels = set(top_counts[TYPE_COL].tolist())
    d3a["Type_6"] = d3a[TYPE_COL].where(d3a[TYPE_COL].isin(top_labels), "Others")

    # Build Others breakdown per YearMonth for hover
    others_map = {}
    for ym_key, slice_df in d3a.groupby("YearMonth"):
        small = slice_df[~slice_df[TYPE_COL].isin(top_labels)]
        if small.empty:
            others_map[ym_key] = "—"
            continue
        counts = (small.groupby(TYPE_COL)["DisNo."].count()
                        .sort_values(ascending=False))
        items = [f"{t}: {int(c)}" for t, c in counts.items()]
        if len(items) > 15:
            items = items[:15] + [f"+{len(counts)-15} more"]
        others_map[ym_key] = "; ".join(items) if items else "—"

    area = d3a.groupby(["YearMonth", "Type_6"], as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
    if not area.empty:
        area["YearMonth_dt"] = pd.to_datetime(area["YearMonth"], errors="coerce")
        area = area.sort_values("YearMonth_dt")
        unique_types = [t for t in area["Type_6"].unique() if t != "Others"]
        color_map = {t: AREA_TOP5_PALETTE[i % len(AREA_TOP5_PALETTE)] for i, t in enumerate(sorted(unique_types))}
        color_map["Others"] = OTHERS_COLOR

        # Attach customdata (only for Others)
        area["customdata"] = np.where(
            area["Type_6"] == "Others",
            area["YearMonth"].map(lambda y: "<br><i>Others:</i> " + others_map.get(y, "—")),
            ""
        )

        fig_area = px.area(area, x="YearMonth_dt", y="Count", color="Type_6", color_discrete_map=color_map)
        fig_area.update_traces(
            hovertemplate="<b>%{x|%Y-%m}</b><br>%{fullData.name}: %{y:,}%{customdata}<extra></extra>"
        )
        fig_area.update_layout(xaxis_title="Date", yaxis_title="Number of Events",
                               legend_title="Disaster Type (Top-5 + Others)", hovermode="x unified")
        st.plotly_chart(fig_area, use_container_width=True, config=PLOTLY_CFG)

    # ======================
    # 4) Yearly Distribution (Counts)
    # ======================
    st.markdown("---")
    _anchor("sec-da-year-dist")
    section_title("Yearly Distribution (Counts)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            yearsY = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_yearly")
        with c2:
            region_opts_Y = _available_regions(df, yearsY)
            regionY = st.selectbox("Region", options=region_opts_Y,
                                   index=region_opts_Y.index(_coerce_choice(st.session_state["glob_region"], region_opts_Y, 0)),
                                   key="region_yearly")
        with c3:
            country_opts_Y = _available_countries(df, yearsY, regionY)
            countryY = st.selectbox("Country", options=country_opts_Y,
                                    index=country_opts_Y.index(_coerce_choice(st.session_state["glob_country"], country_opts_Y, 0)),
                                    key="country_yearly")
        with c4:
            typeY = type_sel("yearly", yearsY, regionY, countryY)

    dY = _apply_scope(df, yearsY, regionY, countryY)
    if typeY and typeY != "All" and TYPE_COL in dY.columns:
        dY = dY[dY[TYPE_COL] == typeY]

    if "Event Date" in dY.columns and dY["Event Date"].notna().any():
        dY["Year"] = dY["Event Date"].dt.year
    else:
        dY["Year"] = pd.to_numeric(dY["Start Year"], errors="coerce")

    year_counts = (
        dY.dropna(subset=["Year"])
          .groupby("Year", as_index=False)["DisNo."].count()
          .rename(columns={"DisNo.": "Count"})
          .sort_values("Year")
    )

    if not year_counts.empty:
        full_years = pd.DataFrame({"Year": list(range(yearsY[0], yearsY[1] + 1))})
        year_counts = full_years.merge(year_counts, on="Year", how="left").fillna({"Count": 0})

        tab_line, tab_bar = st.tabs(["Line Chart", "Bar Chart"])
        with tab_line:
            fig_year_line = px.line(year_counts, x="Year", y="Count", markers=True,
                                    color_discrete_sequence=[LINE_COLOR])
            fig_year_line.update_layout(xaxis_title="Year", yaxis_title="Number of Events",
                                        hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_year_line, use_container_width=True, config=PLOTLY_CFG_NOZOOM)
        with tab_bar:
            fig_year_bar = px.bar(year_counts, x="Year", y="Count", text="Count",
                                  color_discrete_sequence=[BAR_COLOR])
            fig_year_bar.update_traces(textposition="outside", cliponaxis=False)
            fig_year_bar.update_layout(xaxis_title="Year", yaxis_title="Number of Events",
                                       showlegend=False, bargap=0.2, hovermode="x unified")
            st.plotly_chart(fig_year_bar, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

    # ======================
    # 4b) Yearly Distribution by Type (Top-5 + Others, with "Others" hover) — NO 100% TAB
    # ======================
    st.markdown("---")
    _anchor("sec-da-year-bytype")
    section_title("Yearly Distribution by Type (Top-5 + Others)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            yearsYT = st.slider("Year range", min_year, max_year,
                                value=st.session_state["glob_years"], key="years_yearly_types")
        with c2:
            region_opts_YT = _available_regions(df, yearsYT)
            regionYT = st.selectbox(
                "Region", options=region_opts_YT,
                index=region_opts_YT.index(_coerce_choice(st.session_state["glob_region"], region_opts_YT, 0)),
                key="region_yearly_types",
            )
        with c3:
            country_opts_YT = _available_countries(df, yearsYT, regionYT)
            countryYT = st.selectbox(
                "Country", options=country_opts_YT,
                index=country_opts_YT.index(_coerce_choice(st.session_state["glob_country"], country_opts_YT, 0)),
                key="country_yearly_types",
            )

    # ---- Scope (no type filter here) ----
    dYT = _apply_scope(df, yearsYT, regionYT, countryYT).copy()

    # ---- Year derivation (robust) ----
    if "Event Date" in dYT.columns and dYT["Event Date"].notna().any():
        dYT["Year"] = pd.to_datetime(dYT["Event Date"], errors="coerce").dt.year
    else:
        dYT["Year"] = pd.to_numeric(dYT.get("Start Year"), errors="coerce")

    dYT = dYT.dropna(subset=["Year"])
    dYT["Year"] = dYT["Year"].astype(int)

    # ---- Guard
    if TYPE_COL not in dYT.columns:
        st.warning(f"Column `{TYPE_COL}` not found; cannot build type-based distribution.")
    else:
        # A) Decide Top-N
        TOP_N = 5
        totals_scoped = (
            dYT.groupby(TYPE_COL, as_index=False)["DisNo."].count()
            .rename(columns={"DisNo.": "TotalCount"})
            .sort_values("TotalCount", ascending=False)
        )
        top_types = totals_scoped[TYPE_COL].head(TOP_N).tolist()

        # B) Year×Type counts (complete scaffold)
        full_years = pd.DataFrame({"Year": list(range(yearsYT[0], yearsYT[1] + 1))})
        grp_year_type = (
            dYT.groupby(["Year", TYPE_COL], as_index=False)["DisNo."].count()
            .rename(columns={"DisNo.": "Count"})
        )
        all_types = sorted(dYT[TYPE_COL].dropna().unique().tolist())
        scaffold = full_years.assign(key=1).merge(pd.DataFrame({TYPE_COL: all_types, "key": 1}), on="key").drop(columns="key")
        ytc_full = scaffold.merge(grp_year_type, on=["Year", TYPE_COL], how="left").fillna({"Count": 0})
        ytc_full["Count"] = ytc_full["Count"].astype(int)

        # C) Collapse to Others + per-year hover
        def _others_detail_for_year(df_year: pd.DataFrame) -> str:
            small = df_year[~df_year[TYPE_COL].isin(top_types)].copy()
            if small.empty:
                return "—"
            small = small[small["Count"] > 0].sort_values("Count", ascending=False)
            if small.empty:
                return "—"
            lines = [f"{t}: {int(c)}" for t, c in zip(small[TYPE_COL], small["Count"])]
            if len(lines) > 15:
                lines = lines[:15] + [f"+{len(small)-15} more"]
            return "; ".join(lines)

        others_hover_map = {int(y): _others_detail_for_year(ytc_full[ytc_full["Year"] == y]) for y in full_years["Year"]}

        ytc_top = ytc_full[ytc_full[TYPE_COL].isin(top_types)].copy()
        ytc_others = (
            ytc_full[~ytc_full[TYPE_COL].isin(top_types)]
            .groupby("Year", as_index=False)["Count"].sum()
            .assign(**{TYPE_COL: "Others"})
        )
        ytc_others["OthersDetail"] = ytc_others["Year"].map(others_hover_map)

        ytc_plot = pd.concat([ytc_top, ytc_others], ignore_index=True)

        type_order = top_types + (["Others"] if "Others" in ytc_plot[TYPE_COL].unique() else [])
        ytc_plot[TYPE_COL] = pd.Categorical(ytc_plot[TYPE_COL], categories=type_order, ordered=True)
        ytc_plot = ytc_plot.sort_values(["Year", TYPE_COL])

        # Tabs: Stacked / Grouped (NO 100%)
        tab_stack, tab_group = st.tabs(["Stacked Bars", "Grouped Bars"])
        common_orders = {"category_orders": {TYPE_COL: type_order}}
        common_colors = {"color_discrete_sequence": TYPE_PALETTE}

        ytc_plot["customdata"] = np.where(
            ytc_plot[TYPE_COL].astype(str) == "Others",
            ytc_plot["Year"].map(lambda y: "<br><i>Others:</i> " + others_hover_map.get(int(y), "—")),
            ""
        )

        with tab_stack:
            fig_stack = px.bar(
                ytc_plot,
                x="Year", y="Count",
                color=TYPE_COL,
                **common_orders, **common_colors,
            )
            fig_stack.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Count: %{y:,}<br>"
                    "<b>%{fullData.name}</b>%{customdata}"
                    "<extra></extra>"
                ),
                customdata=ytc_plot["customdata"],
            )
            fig_stack.update_layout(
                barmode="stack",
                xaxis_title="Year", yaxis_title="Number of Events",
                legend_title="Disaster Type", hovermode="x unified",
                bargap=0.15,
            )
            st.plotly_chart(fig_stack, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

        with tab_group:
            fig_group = px.bar(
                ytc_plot,
                x="Year", y="Count",
                color=TYPE_COL,
                **common_orders, **common_colors,
            )
            fig_group.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Count: %{y:,}<br>"
                    "<b>%{fullData.name}</b>%{customdata}"
                    "<extra></extra>"
                ),
                customdata=ytc_plot["customdata"],
            )
            fig_group.update_layout(
                barmode="group",
                xaxis_title="Year", yaxis_title="Number of Events",
                legend_title="Disaster Type", hovermode="x unified",
                bargap=0.20,
            )
            st.plotly_chart(fig_group, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

    # ======================
    # 5) Concentration Heat & Map — scoped selectors (years→region→country→type)
    # ======================
    st.markdown("---")
    _anchor("sec-da-concentration")
    section_title("Concentration Heat & Map (Historical)")

    # Aesthetic red gradient
    RED_HEAT = [
        (0.00, "#fde0dd"),
        (0.20, "#fcbba1"),
        (0.40, "#fc9272"),
        (0.60, "#fb6a4a"),
        (0.80, "#de2d26"),
        (1.00, "#a50f15"),
    ]

    # ---------- helpers JUST for this visual ----------
    def _expand_multi_country(df_in: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df_in.iterrows():
            raw = str(row.get("Country", "") or "")
            splits = (raw.replace("&", ",").replace("/", ",").replace(" and ", ",").split(","))
            splits = [c.strip() for c in splits if c.strip()]
            if not splits:
                splits = [raw.strip() or "Unknown"]
            for c in splits:
                r = row.copy()
                r["Country"] = c
                rows.append(r)
        return pd.DataFrame(rows) if rows else df_in.copy()

    @st.cache_data(show_spinner=False)
    def _country_centroids_for_map(src: pd.DataFrame) -> pd.DataFrame:
        base = src.dropna(subset=["Latitude", "Longitude"]).copy()
        base["Latitude"]  = pd.to_numeric(base["Latitude"], errors="coerce")
        base["Longitude"] = pd.to_numeric(base["Longitude"], errors="coerce")
        base = base.dropna(subset=["Latitude", "Longitude"])
        cents = (
            base.groupby("Country", as_index=False)[["Latitude", "Longitude"]]
                .median()
                .rename(columns={"Latitude":"CentroidLat","Longitude":"CentroidLon"})
        )
        return cents

    def _prep_mappable(df_src: pd.DataFrame, years: tuple[int,int]) -> pd.DataFrame:
        """Filter by years, expand multi-country, fill coords with centroids; drop rows still missing coords."""
        d = _filter_by_years(df_src, years)
        d = _expand_multi_country(d)

        cents = _country_centroids_for_map(df_src)   # use whole dataset’s geocoded points for stable centroids
        d = d.merge(cents, how="left", on="Country")

        d["Latitude"]  = pd.to_numeric(d.get("Latitude"), errors="coerce")
        d["Longitude"] = pd.to_numeric(d.get("Longitude"), errors="coerce")
        d["Latitude"]  = d["Latitude"].combine_first(d["CentroidLat"])
        d["Longitude"] = d["Longitude"].combine_first(d["CentroidLon"])
        d = d.dropna(subset=["Latitude", "Longitude"])
        return d

    def _options_for_region(dmap: pd.DataFrame) -> list[str]:
        regs = sorted([r for r in dmap.get("Region", pd.Series([], dtype=str)).dropna().astype(str).unique() if r.strip()])
        return ["All Regions"] + regs

    def _options_for_country(dmap: pd.DataFrame, region: str) -> list[str]:
        d = dmap if region == "All Regions" else dmap[dmap["Region"] == region]
        countries = sorted([c for c in d.get("Country", pd.Series([], dtype=str)).dropna().astype(str).unique() if c.strip()])
        return ["Global"] + countries

    def _options_for_type(dmap: pd.DataFrame, region: str, country: str, type_col: str) -> list[str]:
        d = dmap
        if region != "All Regions":
            d = d[d["Region"] == region]
        if country != "Global":
            d = d[d["Country"] == country]
        if type_col not in d.columns:
            return ["All"]
        types = sorted([t for t in d[type_col].dropna().astype(str).unique() if t.strip()])
        return ["All"] + types

    # ---------- UI: build options from a mappable base FIRST ----------
    with st.expander("Selections (this visual only)", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            years5 = st.slider("Year range", min_year, max_year,
                            value=st.session_state["glob_years"], key="years_5")

        # Build a mappable dataset *for these years only*
        d5_base = _prep_mappable(df, years5)

        with c2:
            region_opts_5 = _options_for_region(d5_base)
            region5 = st.selectbox(
                "Region",
                options=region_opts_5,
                index=region_opts_5.index(_coerce_choice(st.session_state.get("conc_region", "All Regions"), region_opts_5, 0)),
                key="region_5",
            )
            st.session_state["conc_region"] = region5

        with c3:
            # Countries available under the chosen region (and years)
            country_opts_5 = _options_for_country(d5_base, region5)
            country5 = st.selectbox(
                "Country",
                options=country_opts_5,
                index=country_opts_5.index(_coerce_choice(st.session_state.get("conc_country", "Global"), country_opts_5, 0)),
                key="country_5",
            )
            st.session_state["conc_country"] = country5

        with c4:
            # Types available after years + region + country (and only those with events)
            type_opts_5 = _options_for_type(d5_base, region5, country5, TYPE_COL)
            type5 = st.selectbox(
                "Disaster Type",
                options=type_opts_5,
                index=type_opts_5.index(_coerce_choice(st.session_state.get("conc_type", "All"), type_opts_5, 0)),
                key="type_5",
            )
            st.session_state["conc_type"] = type5

        with c5:
            weight_by = st.selectbox(
                "Weight by",
                ["Frequency", "Total Affected", "Total Deaths"],
                index=0,
                help="Frequency counts events. Others weight density by impact (winsorized at 95th percentile).",
                key="weight_5",
            )

    # ---------- Apply the chosen scope on the already-mappable base ----------
    d5 = d5_base.copy()
    if region5 != "All Regions":
        d5 = d5[d5["Region"] == region5]
    if country5 != "Global":
        d5 = d5[d5["Country"] == country5]
    if type5 != "All" and TYPE_COL in d5.columns:
        d5 = d5[d5[TYPE_COL] == type5]

    # Clip to region box for a tidy frame
    d5 = _clip_to_region_bbox(d5, region5)

    if d5.empty:
        # Show nothing if truly empty (you also won’t get empty choices thanks to option building above)
        st.info("No mappable events for the selected filters.")
    else:
        # Hover text (no nulls)
        if "Event Date" in d5.columns and d5["Event Date"].notna().any():
            d5["Year"] = pd.to_datetime(d5["Event Date"], errors="coerce").dt.year
        else:
            d5["Year"] = pd.to_numeric(d5.get("Start Year", np.nan), errors="coerce")

        safe_name    = d5.get("Event Name", pd.Series(index=d5.index, dtype=object)).fillna("").astype(str).str.strip()
        safe_type    = d5.get(TYPE_COL,    pd.Series(index=d5.index, dtype=object)).fillna("Event").astype(str)
        safe_country = d5.get("Country",   pd.Series(index=d5.index, dtype=object)).fillna("—").astype(str)
        safe_year    = d5["Year"].fillna("").astype("Int64").astype(str).replace("<NA>", "")

        d5["Hover"] = np.where(
            safe_name.eq(""),
            safe_type + " — " + safe_country + np.where(safe_year.eq(""), "", " (" + safe_year + ")"),
            safe_name + np.where(safe_year.eq(""), "", " (" + safe_year + ")")
        )

        # Weights
        z = None
        if weight_by != "Frequency":
            col = "Total Affected" if weight_by == "Total Affected" else "Total Deaths"
            if col in d5.columns:
                z_raw = pd.to_numeric(d5[col], errors="coerce").fillna(0)
                cap = z_raw.quantile(0.95) if len(z_raw) else 0
                z = (np.clip(z_raw, 0, cap) / cap) if cap > 0 else np.ones(len(d5))
            else:
                z = np.ones(len(d5))

        # Centering logic: world by default, closer when country pinned
        if country5 == "Global":
            center = dict(lat=10.0, lon=10.0)
            zoom = 1.35
        else:
            lat_series = pd.to_numeric(d5["Latitude"], errors="coerce").dropna()
            lon_series = pd.to_numeric(d5["Longitude"], errors="coerce").dropna()
            if len(lat_series) and len(lon_series):
                center = dict(lat=float(lat_series.mean()), lon=float(lon_series.mean()))
                zoom = 3.6
            else:
                center = dict(lat=10.0, lon=10.0)
                zoom = 1.35

        # Tighter kernel → clearer concentration (hexbin feel)
        radius_px = 18

        fig_density = px.density_mapbox(
            d5,
            lat="Latitude", lon="Longitude",
            z=z, radius=radius_px,
            center=center, zoom=zoom,
            mapbox_style="carto-positron",
            color_continuous_scale=RED_HEAT,
        )
        fig_density.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540)

        fig_scatter = go.Figure(fig_density)
        fig_scatter.add_trace(go.Scattermapbox(
            lat=d5["Latitude"], lon=d5["Longitude"], mode="markers",
            marker=dict(size=4, opacity=0.75, color="#7a0f0f"),
            text=d5["Hover"], hovertemplate="%{text}<extra></extra>",
        ))
        st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CFG)

        cap_note = "" if weight_by == "Frequency" else " (winsorized at 95th percentile)"
        st.caption("Heat shows concentration (light → dark red). Missing coordinates replaced with country centroids" + cap_note + ".")

    # ======================
    # 6) Temporal & Spatial Heat (calendar never disappears)
    # ======================
    st.markdown("---")
    _anchor("sec-da-calendar")
    section_title("Temporal & Spatial Heat")

    tab_cal, tab_xy = st.tabs(["Calendar Heatmap (Year–Month)", "Lat/Lon Density Heat"])

    with tab_cal:
        with st.expander("Selections (this visual)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                years6 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_6")
            with c2:
                region_opts_6 = _available_regions(df, years6)
                region6 = st.selectbox("Region", options=region_opts_6,
                                       index=region_opts_6.index(_coerce_choice(st.session_state["glob_region"], region_opts_6, 0)),
                                       key="region_6")
            with c3:
                country_opts_6 = _available_countries(df, years6, region6)
                country6 = st.selectbox("Country", options=country_opts_6,
                                        index=country_opts_6.index(_coerce_choice(st.session_state["glob_country"], country_opts_6, 0)),
                                        key="country_6")
            with c4:
                type6 = type_sel("cal", years6, region6, country6)

        d6 = _apply_scope(df, years6, region6, country6)
        if type6 and type6 != "All" and TYPE_COL in d6.columns:
            d6 = d6[d6[TYPE_COL] == type6]

        if "Event Date" in d6.columns and d6["Event Date"].notna().any():
            ym6 = d6["Event Date"].dt.to_period("M").astype(str)
        else:
            sy = pd.to_numeric(d6["Start Year"], errors="coerce")
            sm = pd.to_numeric(d6["Start Month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
            ym6 = pd.to_datetime(dict(year=sy, month=sm, day=1), errors="coerce").dt.to_period("M").astype(str)

        cal = d6.assign(YearMonth=ym6)
        cal["Year"] = pd.to_numeric(cal["YearMonth"].str[:4], errors="coerce")
        cal["Month"] = pd.to_numeric(cal["YearMonth"].str[5:7], errors="coerce")

        # Build a full grid of selected years × 1..12 months
        years_span = list(range(years6[0], years6[1] + 1))
        months_span = list(range(1, 13))
        grid = pd.MultiIndex.from_product([years_span, months_span], names=["Year", "Month"]).to_frame(index=False)

        heat = (cal.dropna(subset=["Year","Month"])
                    .groupby(["Year", "Month"], as_index=False)["DisNo."].count()
                    .rename(columns={"DisNo.":"Count"}))

        heat_full = grid.merge(heat, on=["Year","Month"], how="left").fillna({"Count": 0})
        pivot = heat_full.pivot(index="Year", columns="Month", values="Count").reindex(index=years_span, columns=months_span, fill_value=0)

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=INTENSITY_SCALE,
            aspect="auto",
            origin="lower",
            zmin=0
        )
        fig_heat.update_layout(
            coloraxis_colorbar=dict(title="Events"),
            xaxis_title="Month",
            yaxis_title="Year",
        )
        st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CFG)

    with tab_xy:
        with st.expander("Selections (this visual)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                years7 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_7")
            with c2:
                region_opts_7 = _available_regions(df, years7)
                region7 = st.selectbox("Region", options=region_opts_7,
                                       index=region_opts_7.index(_coerce_choice(st.session_state["glob_region"], region_opts_7, 0)),
                                       key="region_7")
            with c3:
                country_opts_7 = _available_countries(df, years7, region7)
                country7 = st.selectbox("Country", options=country_opts_7,
                                        index=country_opts_7.index(_coerce_choice(st.session_state["glob_country"], country_opts_7, 0)),
                                        key="country_7")
            with c4:
                type7 = type_sel("xy", years7, region7, country7)

        d7 = _apply_scope(df, years7, region7, country7)
        if type7 and type7 != "All" and TYPE_COL in d7.columns:
            d7 = d7[d7[TYPE_COL] == type7]
        d7 = d7.dropna(subset=["Latitude", "Longitude"])

        if d7.empty:
            st.info("No geocoded events for the selected filters.")
        else:
            fig_xy = px.density_heatmap(d7, x="Longitude", y="Latitude", nbinsx=40, nbinsy=40,
                                        color_continuous_scale=HEAT_SCALE_XY)
            fig_xy.update_layout(xaxis_title="Longitude", yaxis_title="Latitude",
                                 coloraxis_colorbar=dict(title="Density"))
            st.plotly_chart(fig_xy, use_container_width=True, config=PLOTLY_CFG)

    # Footer
    st.markdown("---")
    st.caption("Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
