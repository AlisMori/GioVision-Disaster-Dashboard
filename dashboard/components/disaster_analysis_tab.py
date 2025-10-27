"""
disaster_analysis.py
--------------------
Disaster Analysis page using EM-DAT.

Global controls (sidebar):
- Year range (global)
- Region (global)
- Country (global; constrained by Region + Year range)
- (No Disaster Type in the sidebar by design)

Each visual:
- Defaults to the global controls
- Local selectors (if any) are constrained to EXISTING values under that visual's scope
- Changing sidebar controls immediately affects ALL visuals

Style:
- gv-section-title / gv-subsection-title bars
- In-page anchors ?page=Analysis&section=...
- Carto-Positron maps; cohesive blues for categories; effective gradient for intensities
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

# Palettes
BLUE_PALETTE = [
    "#dceef7", "#b5d7ee", "#8ec0e5", "#67a9dc",
    "#4192d1", "#2c7fb8", "#1865ab", "#0b4f8a",
]
TYPE_PALETTE = BLUE_PALETTE  # use singular blue variations everywhere types are categorical
AREA_TOP5_PALETTE = ["#2c7fb8", "#4192d1", "#67a9dc", "#8ec0e5", "#0b4f8a"]  # stacked area top5 blues
OTHERS_COLOR = "#9e9e9e"

INTENSITY_SCALE = "YlOrRd"    # choropleth / calendar
HEAT_SCALE_MAP  = "Inferno"   # map density heat
HEAT_SCALE_XY   = "Viridis"   # 2D lat/lon heat

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
# For Bar/Pie: remove zoom/pan tools (keep hover)
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
    """
    Return only the epicenter (first) country from a possibly multi-country string.
    Examples:
      "China, Japan" -> "China"
      "Turkey/Syria" -> "Turkey"
      "France & Spain" -> "France"
      "Iran and Iraq" -> "Iran"
    """
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

    # Clean a Severity column if it exists (text → integers where possible)
    if "Severity" in df.columns:
        df["Severity_raw"] = df["Severity"].astype(str)
        df["Severity"] = pd.to_numeric(
            df["Severity_raw"].str.extract(r"(-?\d+)", expand=False), errors="coerce"
        )

    # --- Epicenter country normalization (in-memory only) ---
    if "Country" in df.columns:
        df["Country_all"] = df["Country"].astype(str)  # preserve original
        df["Country"] = df["Country_all"].apply(_first_country_only)

    return df

# ---------- availability helpers (ONLY valid options based on current scope) ----------
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

def _available_types(df: pd.DataFrame, years: Tuple[int,int], region: str, country: str) -> List[str]:
    d = _filter_by_years(df, years)
    if region and region != "All Regions":
        d = d[d["Region"] == region]
    if country and country != "Global":
        d = d[d["Country"] == country]
    dtypes = sorted([t for t in d["Disaster Type"].dropna().astype(str).unique() if t.strip()])
    return ["All"] + dtypes

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

# ---- Geo helpers for concentration map ----
REGION_BBOX = {
    "Africa":   (-35.0, 37.5,  -20.0,  52.0),
    "Asia":     ( -5.0, 82.0,  25.0, 180.0),  # includes ME & SE Asia
    "Europe":   ( 35.0, 72.5, -25.0,  45.0),
    "Americas": (-57.0, 72.0, -170.0, -30.0), # N+S America
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

    # ---------- Dataset bounds ----------
    if "Start Year" in df and df["Start Year"].notna().any():
        min_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().min())
        max_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().max())
    else:
        min_year, max_year = 1970, 2025

    # ---------- Global state (single source of truth) ----------
    if "glob_years" not in st.session_state:
        st.session_state["glob_years"] = (max(DEFAULT_START, min_year), min(DEFAULT_END, max_year))
    if "glob_region" not in st.session_state:
        st.session_state["glob_region"] = "All Regions"
    if "glob_country" not in st.session_state:
        st.session_state["glob_country"] = "Global"

    # ---------- Sidebar (GLOBAL CONTROLS) ----------
    st.sidebar.header("Analysis Filters")

    # Year range
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

    # Region options based on years
    region_options = _available_regions(df, st.session_state["glob_years"])
    region_val = _coerce_choice(st.session_state["glob_region"], region_options, 0)
    region_global = st.sidebar.selectbox(
        "Region (global)",
        options=region_options,
        index=region_options.index(region_val),
        key="__region_select__",
    )
    st.session_state["glob_region"] = region_global

    # Country options based on years + region
    country_options = _available_countries(df, st.session_state["glob_years"], st.session_state["glob_region"])
    country_val = _coerce_choice(st.session_state["glob_country"], country_options, 0)
    country_global = st.sidebar.selectbox(
        "Country (global)",
        options=country_options,
        index=country_options.index(country_val),
        key="__country_select__",
    )
    st.session_state["glob_country"] = country_global

    # ---------- Section jump ----------
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

    # ---------- Overview ----------
    _anchor("sec-da-overview")
    section_title("Disaster Analysis (EM-DAT)")
    st.markdown(
        f"Global scope: **{st.session_state['glob_years'][0]}–{st.session_state['glob_years'][1]}**, "
        f"**{st.session_state['glob_region']}**, **{st.session_state['glob_country']}**."
    )
    try:
        st.caption(f"EM-DAT file: `{os.path.relpath(emdat_used_path)}`")
    except Exception:
        st.caption(f"EM-DAT file: `{emdat_used_path}`")

    # Helper to build constrained selector for each visual
    def constrained_type_selector(label_key_prefix: str, years_sel: Tuple[int,int], region_sel: str, country_sel: str) -> str:
        type_options = _available_types(df, years_sel, region_sel, country_sel)
        key_state = f"{label_key_prefix}_type"
        prev = st.session_state.get(key_state, "All")
        coerced = _coerce_choice(prev, type_options, 0)
        val = st.selectbox("Disaster Type", type_options, index=type_options.index(coerced), key=key_state)
        return val

    # 1) Choropleth (total per country)
    st.markdown("---")
    _anchor("sec-da-map-country")
    section_title("Map: Total Disasters per Country")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
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

    d1 = _apply_scope(df, years1, region1, country1)
    if country1 == "Global":
        agg1 = d1.groupby("Country", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Events"})
    else:
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
        st.caption("Hover a country to see total events. All selectors here are constrained to valid choices.")

    # 2) Top-10 frequency
    st.markdown("---")
    _anchor("sec-da-top10")
    section_title("Top-10 Disasters by Frequency")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            years2 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_2")
            region_opts_2 = _available_regions(df, years2)
            region2 = st.selectbox(
                "Region",
                options=region_opts_2,
                index=region_opts_2.index(_coerce_choice(st.session_state["glob_region"], region_opts_2, 0)),
                key="region_2",
            )
        with c2:
            country_opts_2 = _available_countries(df, years2, region2)
            country2 = st.selectbox(
                "Country",
                options=country_opts_2,
                index=country_opts_2.index(_coerce_choice(st.session_state["glob_country"], country_opts_2, 0)),
                key="country_2",
            )
        chart2 = st.radio("Chart", ["Bar", "Pie"], horizontal=True, key="chart_2")

    d2 = _apply_scope(df, years2, region2, country2)
    freq = d2.groupby("Disaster Type", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
    freq = freq.sort_values("Count", ascending=False)

    if freq.empty:
        st.info("No data for the selected filters.")
    else:
        if len(freq) > 10:
            top10 = freq.head(10).copy()
            others_count = int(freq["Count"].iloc[10:].sum())
            top10 = pd.concat([top10, pd.DataFrame({"Disaster Type": ["Others"], "Count": [others_count]})], ignore_index=True)
        else:
            top10 = freq

        if chart2 == "Bar":
            fig_bar = px.bar(
                top10, x="Count", y="Disaster Type", orientation="h",
                color="Disaster Type", color_discrete_sequence=TYPE_PALETTE, text="Count"
            )
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                bargap=0.25,
                showlegend=False
            )
            fig_bar.update_traces(textposition="outside", cliponaxis=False, hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>")
            st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CFG_NOZOOM)
        else:
            fig_pie = px.pie(
                top10, names="Disaster Type", values="Count", hole=0.3,
                color="Disaster Type", color_discrete_sequence=TYPE_PALETTE
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

    # 3) Stacked area timeline (Top-5 + Others)
    st.markdown("---")
    _anchor("sec-da-timeline")
    section_title("Stacked Area Timeline (by Disaster Type – Top-5 + Others)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years3 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_3")
        with c2:
            region_opts_3 = _available_regions(df, years3)
            region3 = st.selectbox(
                "Region",
                options=region_opts_3,
                index=region_opts_3.index(_coerce_choice(st.session_state["glob_region"], region_opts_3, 0)),
                key="region_3",
            )
        with c3:
            country_opts_3 = _available_countries(df, years3, region3)
            country3 = st.selectbox(
                "Country",
                options=country_opts_3,
                index=country_opts_3.index(_coerce_choice(st.session_state["glob_country"], country_opts_3, 0)),
                key="country_3",
            )

    d3 = _apply_scope(df, years3, region3, country3)

    # Year-Month index
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

    top_counts = _top_n_with_others(d3a, "Disaster Type", n=5)
    top_labels = set(top_counts["Disaster Type"].tolist())
    d3a["Type_6"] = d3a["Disaster Type"].where(d3a["Disaster Type"].isin(top_labels), "Others")

    area = d3a.groupby(["YearMonth", "Type_6"], as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Count"})
    if area.empty:
        st.info("No data for the selected filters.")
    else:
        area["YearMonth_dt"] = pd.to_datetime(area["YearMonth"], errors="coerce")
        area = area.sort_values("YearMonth_dt")
        unique_types = [t for t in area["Type_6"].unique() if t != "Others"]
        color_map = {t: AREA_TOP5_PALETTE[i % len(AREA_TOP5_PALETTE)] for i, t in enumerate(sorted(unique_types))}
        color_map["Others"] = OTHERS_COLOR

        fig_area = px.area(area, x="YearMonth_dt", y="Count", color="Type_6", color_discrete_map=color_map)
        fig_area.update_layout(
            xaxis_title="Date", yaxis_title="Number of Events",
            legend_title="Disaster Type (Top-5 + Others)", hovermode="x unified"
        )
        st.plotly_chart(fig_area, use_container_width=True, config=PLOTLY_CFG)

    # 4) Yearly Distribution (Counts)
    # 4) Yearly Distribution (Counts)
    st.markdown("---")
    _anchor("sec-da-year-dist")
    section_title("Yearly Distribution (Counts)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            yearsY = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_yearly")
        with c2:
            region_opts_Y = _available_regions(df, yearsY)
            regionY = st.selectbox(
                "Region",
                options=region_opts_Y,
                index=region_opts_Y.index(_coerce_choice(st.session_state["glob_region"], region_opts_Y, 0)),
                key="region_yearly",
            )
        with c3:
            country_opts_Y = _available_countries(df, yearsY, regionY)
            countryY = st.selectbox(
                "Country",
                options=country_opts_Y,
                index=country_opts_Y.index(_coerce_choice(st.session_state["glob_country"], country_opts_Y, 0)),
                key="country_yearly",
            )
        with c4:
            typeY = constrained_type_selector("yearly", yearsY, regionY, countryY)

    dY = _apply_scope(df, yearsY, regionY, countryY)
    if typeY and typeY != "All":
        dY = dY[dY["Disaster Type"] == typeY]

    # Determine Year column
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

    if year_counts.empty:
        st.info("No data for the selected filters.")
    else:
        # Ensure continuous years in the selected range (optional: fill zeros)
        full_years = pd.DataFrame({"Year": list(range(yearsY[0], yearsY[1] + 1))})
        year_counts = full_years.merge(year_counts, on="Year", how="left").fillna({"Count": 0})

        fig_year = px.bar(
            year_counts, x="Year", y="Count",
            text="Count",
            color_discrete_sequence=["#2c7fb8"],  # single blue
        )
        fig_year.update_traces(textposition="outside", cliponaxis=False)
        fig_year.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Events",
            showlegend=False,
            bargap=0.2,
            hovermode="x unified",
        )
        st.plotly_chart(fig_year, use_container_width=True, config=PLOTLY_CFG_NOZOOM)


   # 5) Concentration heat & map (density + scatter)
    st.markdown("---")
    _anchor("sec-da-concentration")
    section_title("Concentration Heat & Map (Historical)")

    with st.expander("Selections (this visual)", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            years5 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_5")
        with c2:
            region_opts_5 = _available_regions(df, years5)
            region5 = st.selectbox(
                "Region",
                options=region_opts_5,
                index=region_opts_5.index(_coerce_choice(st.session_state["glob_region"], region_opts_5, 0)),
                key="region_5",
            )
        with c3:
            country_opts_5 = _available_countries(df, years5, region5)
            country5 = st.selectbox(
                "Country",
                options=country_opts_5,
                index=country_opts_5.index(_coerce_choice(st.session_state["glob_country"], country_opts_5, 0)),
                key="country_5",
            )
        with c4:
            type5 = constrained_type_selector("conc", years5, region5, country5)
        with c5:
            radius5 = st.slider("Density radius", 6, 50, 24, step=2, help="Increase to show larger/smoother density areas")
        with c6:
            # NEW: include Frequency as default, plus weighted options
            weight_by = st.selectbox(
                "Weight by",
                ["Frequency", "Total Affected", "Total Deaths"],
                index=0,
                help="Frequency counts events. Other options weight density by impact (winsorized at 95th percentile).",
            )

    d5 = _apply_scope(df, years5, region5, country5)
    if type5 and type5 != "All":
        d5 = d5[d5["Disaster Type"] == type5]
    d5 = d5.dropna(subset=["Latitude", "Longitude"])
    d5 = _clip_to_region_bbox(d5, region5)

    # Density weights
    z = None  # default None => px.density_mapbox counts points (Frequency)
    if weight_by != "Frequency":
        col = "Total Affected" if weight_by == "Total Affected" else "Total Deaths"
        if col in d5.columns:
            z_raw = pd.to_numeric(d5[col], errors="coerce").fillna(0)
            cap = z_raw.quantile(0.95) if len(z_raw) else 0
            if cap > 0:
                z = np.clip(z_raw, 0, cap) / cap
            else:
                z = np.ones(len(d5))  # fallback if all zeros
        else:
            z = np.ones(len(d5))      # fallback if column missing

    if d5.empty:
        st.info("No geolocated events for the selected filters.")
    else:
        center, zoom = _center_zoom_from_points(d5["Latitude"], d5["Longitude"])
        fig_density = px.density_mapbox(
            d5,
            lat="Latitude",
            lon="Longitude",
            z=z,  # None => frequency; array => weighted
            radius=int(radius5),
            center=center,
            zoom=zoom,
            mapbox_style="carto-positron",
            color_continuous_scale=HEAT_SCALE_MAP
        )
        fig_density.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540)

        # scatter layer for hovers
        fig_scatter = go.Figure(fig_density)
        fig_scatter.add_trace(go.Scattermapbox(
            lat=d5["Latitude"], lon=d5["Longitude"], mode="markers",
            marker=dict(size=7, opacity=0.85, color="#0b4f8a"),
            text=d5["Event Name"],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.2f}, Lon: %{lon:.2f}<extra></extra>"
        ))
        st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CFG)

        cap_note = "" if weight_by == "Frequency" else " (winsorized at 95th percentile)"
        st.caption(f"Density weighted by **{weight_by}**{cap_note}. Increase radius for larger/smoother blobs.")


    # 6) Calendar / Lat-Lon heat
    st.markdown("---")
    _anchor("sec-da-calendar")
    section_title("Temporal & Spatial Heat")

    tab_cal, tab_xy = st.tabs(["Calendar Heatmap (Year–Month)", "Lat/Lon Density Heat"])

    with tab_cal:
        with st.expander("Selections (this visual)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                years6 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_6")
            with c2:
                region_opts_6 = _available_regions(df, years6)
                region6 = st.selectbox(
                    "Region",
                    options=region_opts_6,
                    index=region_opts_6.index(_coerce_choice(st.session_state["glob_region"], region_opts_6, 0)),
                    key="region_6",
                )
            with c3:
                country_opts_6 = _available_countries(df, years6, region6)
                country6 = st.selectbox(
                    "Country",
                    options=country_opts_6,
                    index=country_opts_6.index(_coerce_choice(st.session_state["glob_country"], country_opts_6, 0)),
                    key="country_6",
                )

        d6 = _apply_scope(df, years6, region6, country6)
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
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                years7 = st.slider("Year range", min_year, max_year, value=st.session_state["glob_years"], key="years_7")
            with c2:
                region_opts_7 = _available_regions(df, years7)
                region7 = st.selectbox(
                    "Region",
                    options=region_opts_7,
                    index=region_opts_7.index(_coerce_choice(st.session_state["glob_region"], region_opts_7, 0)),
                    key="region_7",
                )
            with c3:
                country_opts_7 = _available_countries(df, years7, region7)
                country7 = st.selectbox(
                    "Country",
                    options=country_opts_7,
                    index=country_opts_7.index(_coerce_choice(st.session_state["glob_country"], country_opts_7, 0)),
                    key="country_7",
                )
            with c4:
                type7 = constrained_type_selector("xy", years7, region7, country7)

        d7 = _apply_scope(df, years7, region7, country7)
        if type7 and type7 != "All":
            d7 = d7[d7["Disaster Type"] == type7]
        d7 = d7.dropna(subset=["Latitude", "Longitude"])

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

