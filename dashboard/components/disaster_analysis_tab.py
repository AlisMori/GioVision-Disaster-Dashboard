# dashboard/components/disaster_analysis_tab.py

"""
Disaster Analysis page using EM-DAT.

Teacher changes (2025-10-31) — Finalized:
- RIGHT floating (fixed) global filter panel that stays visible while scrolling
- Year/Region/Country react on FIRST try (no rerun warnings)
- Professional sections & titles; context lines 6–15 words, slightly larger
- Distinguishable palettes for category visuals (pie / stacked / grouped / area)
- Lat/Lon gap-fill once (country medians), reused by maps
"""

from __future__ import annotations
import os, sys
from typing import Tuple, List, Optional, Dict

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly import graph_objects as go
from plotly.colors import qualitative as q  # categorical palettes

# =========================
# CONFIG / PATHS (portable)
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

EMDAT_PATHS = [
    "data/processed/emdat_cleaned.csv",        # run from project root
    "../data/processed/emdat_cleaned.csv",     # run from dashboard/
    "../../data/processed/emdat_cleaned.csv",  # run from dashboard/components/
    "dashboard/data/processed/emdat_cleaned.csv",
]

DEFAULT_START = 2022
DEFAULT_END   = 2025

TYPE_COL = "Disaster Type"

REQUIRED_COLS = [
    "DisNo.", "Event Name", "Country", "Region", "Location",
    "Start Year", "Start Month", "Start Day", "Event Date",
    "Disaster Type", "Disaster Type Standardized",
    "Latitude", "Longitude",
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
        "zoom","pan","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","select2d","lasso2d","zoom2d"
    ],
}

# =========================
# THEME
# =========================
# Single-hue scales for intensity
PALETTE_COUNTRY_CHORO = "Blues"
PALETTE_CONC_MAP      = "Oranges"
PALETTE_CALENDAR      = "Greens"

# Distinct categorical palettes (readable)
PALETTE_TOPN_BARS     = q.Safe
PALETTE_TOPN_PIE      = q.Set3
PALETTE_STACKED_AREA  = q.Vivid
PALETTE_YEAR_BYTYPE   = q.Bold
PALETTE_YEAR_LINE     = ["#2563EB"]
PALETTE_YEAR_BAR      = ["#3B82F6"]
OTHERS_COLOR          = "#9E9E9E"

# =========================
# RENDERING HELPERS
# =========================
def _anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

def story_context(text: str):
    st.markdown(f'<div class="gv-context">{text}</div>', unsafe_allow_html=True)

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

    # Ensure totals
    if "Total Affected" not in df.columns:
        for c in ["Total Deaths", "No. Injured", "No. Affected", "No. Homeless"]:
            if c not in df.columns:
                df[c] = 0
        df["Total Affected"] = (
            df["No. Injured"].fillna(0) + df["No. Affected"].fillna(0) + df["No. Homeless"].fillna(0)
        )

    # Coerce lat/lon
    for c in ["Latitude", "Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional Severity cleaning
    if "Severity" in df.columns:
        df["Severity_raw"] = df["Severity"].astype(str)
        df["Severity"] = pd.to_numeric(
            df["Severity_raw"].str.extract(r"(-?\d+)", expand=False),
            errors="coerce"
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

def _coerce_choice(current: Optional[str], options: List[str], fallback_idx: int = 0) -> str:
    if not options: return ""
    if current in options: return current
    return options[min(fallback_idx, len(options)-1)]

def _apply_scope(df: pd.DataFrame, years: Tuple[int,int], region: str, country: str) -> pd.DataFrame:
    d = _filter_by_years(df, years)
    if region and region != "All Regions":
        d = d[d["Region"] == region]
    if country and country != "Global":
        d = d[d["Country"] == country]
    return d

def constrained_type_selector(
    df: pd.DataFrame, label_key_prefix: str, years_sel: Tuple[int,int], region_sel: str, country_sel: str, type_col: str
) -> str:
    d = _apply_scope(df, years_sel, region_sel, country_sel)
    if type_col not in d.columns:
        return "All"
    opts = ["All"] + sorted([t for t in d[type_col].dropna().astype(str).unique() if t.strip()])
    key_state = f"{label_key_prefix}_type"
    prev = st.session_state.get(key_state, "All")
    coerced = _coerce_choice(prev, opts, 0)
    return st.selectbox("Disaster Type", opts, index=opts.index(coerced), key=key_state)

# ---- Geo helpers ----
REGION_BBOX = {
    "Africa":   (-35.0, 37.5, -20.0, 52.0),
    "Asia":     ( -5.0, 82.0,  25.0, 180.0),
    "Europe":   ( 35.0, 72.5, -25.0,  45.0),
    "Americas": (-57.0, 72.0,-170.0, -30.0),
    "Oceania":  (-50.0, 10.0, 105.0, 180.0),
}
def _clip_to_region_bbox(d: pd.DataFrame, region: str) -> pd.DataFrame:
    if not region or region == "All Regions" or region not in REGION_BBOX:
        return d
    lat_min, lat_max, lon_min, lon_max = REGION_BBOX[region]
    dd = d.copy()
    dd["Latitude"]  = pd.to_numeric(dd["Latitude"], errors="coerce")
    dd["Longitude"] = pd.to_numeric(dd["Longitude"], errors="coerce")
    return dd[(dd["Latitude"].between(lat_min, lat_max)) & (dd["Longitude"].between(lon_min, lon_max))]

# =========================
# STATE SEEDING (no “second try” + no rerun)
# =========================
def _coerce_into(options: List[str], value: Optional[str], fallback_idx: int = 0) -> str:
    if not options:
        return ""
    if value in options:
        return value
    return options[min(fallback_idx, len(options)-1)]

def _seed_filter_keys(df: pd.DataFrame, min_year: int, max_year: int):
    """
    Ensure widget keys exist and are valid BEFORE we read them.
    Returns: (current_years, current_region, region_options, country_options)
    """
    gy_default = (max(DEFAULT_START, min_year), min(DEFAULT_END, max_year))
    st.session_state.setdefault("glob_years", gy_default)
    st.session_state.setdefault("glob_region", "All Regions")
    st.session_state.setdefault("glob_country", "Global")

    st.session_state.setdefault("__years_slider__", st.session_state["glob_years"])

    current_years  = st.session_state["__years_slider__"]
    region_options = _available_regions(df, current_years)

    st.session_state.setdefault("__region_select__", _coerce_into(region_options, st.session_state["glob_region"], 0))
    current_region = _coerce_into(region_options, st.session_state["__region_select__"], 0)
    st.session_state["__region_select__"] = current_region

    country_options = _available_countries(df, current_years, current_region)
    st.session_state.setdefault("__country_select__", _coerce_into(country_options, st.session_state["glob_country"], 0))
    st.session_state["__country_select__"] = _coerce_into(country_options, st.session_state["__country_select__"], 0)

    # mirror to globals so visuals always read current values
    st.session_state["glob_years"]   = st.session_state["__years_slider__"]
    st.session_state["glob_region"]  = st.session_state["__region_select__"]
    st.session_state["glob_country"] = st.session_state["__country_select__"]

    return current_years, current_region, region_options, country_options


# =========================
# PAGE RENDER
# =========================
def render():
    # 1) Scoped CSS: RIGHT column sticky with internal scroll, narrower width
        # 1) Scoped CSS: RIGHT column sticky with internal scroll, 15% narrower
    st.markdown(
        """
        <style>
          /* Let sticky work inside Streamlit's layout */
          .main .block-container { overflow: visible !important; }

          /* Scope only to this page */
          #da-sticky-scope #da-sticky-panel {
              /* the panel sits INSIDE the right column and becomes sticky at its own start */
              position: sticky;
              top: 96px;                     /* start sticking just below your header/banner */
              align-self: flex-start;        /* keep the panel aligned to the top of its column */
              z-index: 2;                    /* ensure it stays above nearby charts/toolbars */
            }

          #da-sticky-scope #da-sticky-panel .panel-inner{
              background:#fff;
              border:1px solid rgba(0,0,0,0.06);
              border-radius:12px;
              /* ↓ 15% narrower than previous clamp(210px, 17vw, 240px) */
              width: clamp(180px, 14.5vw, 204px);
              max-height: calc(100vh - 140px);      /* viewport minus top offset */
              overflow: auto;                        /* scroll inside the panel */
              padding: 10px 12px;
              box-shadow: 0 2px 10px rgba(0,0,0,0.06);
          }

          /* Optional: subtle heading style inside panel */
          #da-sticky-scope #da-sticky-panel .panel-title{
              font-weight: 700;
              margin: 0 0 6px 2px;
          }

          /* Safety on small screens */
          @media (max-width: 1100px){
            #da-sticky-scope #da-sticky-panel { position: static; }
            #da-sticky-scope #da-sticky-panel .panel-inner{
              width: 100%;
              max-height: none;
            }
          }

          /* Context line style */
          .gv-context { font-size: 0.95rem; color:#3f3f46; margin: 2px 0 10px 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # 2) Load data
    emdat_used_path = _first_existing_path(EMDAT_PATHS)
    if not emdat_used_path:
        st.error(
            "Could not find emdat_cleaned.csv in:\n"
            "- data/processed/\n- ../data/processed/\n- ../../data/processed/\n- dashboard/data/processed/\n"
        )
        st.stop()

    df = load_emdat(emdat_used_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"EM-DAT file is missing columns: {', '.join(missing)}")
        st.stop()

    # 3) Dataset bounds
    if "Start Year" in df and df["Start Year"].notna().any():
        min_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().min())
        max_year = int(pd.to_numeric(df["Start Year"], errors="coerce").dropna().max())
    else:
        min_year, max_year = 1970, 2025

    # 4) Seed state
    st.session_state.setdefault("glob_years", (max(DEFAULT_START, min_year), min(DEFAULT_END, max_year)))
    st.session_state.setdefault("glob_region", "All Regions")
    st.session_state.setdefault("glob_country", "Global")

    # 5) Prime widget keys (first-try ready)
    current_years, current_region, region_options, country_options = _seed_filter_keys(df, min_year, max_year)

    # 6) Sticky scope + layout
    st.markdown('<div id="da-sticky-scope">', unsafe_allow_html=True)
    left, right = st.columns([7, 2], gap="large")

    # ---------- RIGHT: Sticky Panel WITH FILTERS INSIDE ----------
    with right:
        st.markdown('<div id="da-sticky-panel"><div class="panel-inner">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Analysis Filters</div>', unsafe_allow_html=True)

        # Recompute options based on current slider before dependent widgets
        new_years = st.slider(
            "Year range",
            min_value=min_year, max_value=max_year,
            value=current_years, step=1, key="__years_slider__"
        )
        current_years = new_years

        region_options = _available_regions(df, current_years)
        region_val = _coerce_choice(st.session_state.get("__region_select__", "All Regions"), region_options, 0)
        new_region = st.selectbox(
            "Region", options=region_options, index=region_options.index(region_val), key="__region_select__"
        )

        country_options = _available_countries(df, current_years, new_region)
        country_val = _coerce_choice(st.session_state.get("__country_select__", "Global"), country_options, 0)
        new_country = st.selectbox(
            "Country", options=country_options, index=country_options.index(country_val), key="__country_select__"
        )

        # Mirror to globals (so visuals pick them up in the same run)
        st.session_state["glob_years"]   = current_years
        st.session_state["glob_region"]  = new_region
        st.session_state["glob_country"] = new_country

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ---------- LEFT: Visuals ----------
    years   = st.session_state["glob_years"]
    region  = st.session_state["glob_region"]
    country = st.session_state["glob_country"]

    scoped = _apply_scope(df, years, region, country).copy()

    # Year helper column
    if "Event Date" in scoped.columns and scoped["Event Date"].notna().any():
        scoped["Year"] = pd.to_datetime(scoped["Event Date"], errors="coerce").dt.year
    else:
        scoped["Year"] = pd.to_numeric(scoped.get("Start Year"), errors="coerce")

    # Fill coordinates by country medians (for maps)
    geo_base = df.dropna(subset=["Latitude","Longitude"]).copy()
    geo_base["Latitude"]  = pd.to_numeric(geo_base["Latitude"], errors="coerce")
    geo_base["Longitude"] = pd.to_numeric(geo_base["Longitude"], errors="coerce")
    geo_base = geo_base.dropna(subset=["Latitude","Longitude"])
    country_centroids = (
        geo_base.groupby("Country", as_index=False)[["Latitude","Longitude"]]
        .median()
        .rename(columns={"Latitude":"CentroidLat","Longitude":"CentroidLon"})
    )
    scoped = scoped.merge(country_centroids, on="Country", how="left")
    scoped["Latitude"]  = pd.to_numeric(scoped.get("Latitude"), errors="coerce").combine_first(scoped["CentroidLat"])
    scoped["Longitude"] = pd.to_numeric(scoped.get("Longitude"), errors="coerce").combine_first(scoped["CentroidLon"])

    # ---------- helper stats ----------
    def _top_country(agg_df: pd.DataFrame) -> str:
        if agg_df.empty: return ""
        r = agg_df.sort_values("Events", ascending=False).head(1)
        return str(r.iloc[0]["Country"]) if not r.empty else ""

    def _top_type(df_in: pd.DataFrame) -> str:
        if TYPE_COL not in df_in.columns or df_in.empty: return ""
        s = df_in.groupby(TYPE_COL)["DisNo."].count().sort_values(ascending=False)
        return str(s.index[0]) if len(s) else ""

    # ---------- page ----------
    with left:
        _anchor("sec-da-overview")
        section_title("Disaster Analysis")
        st.markdown(
            "Exploratory view of EM-DAT disaster records across time and space. "
            "Use the filter panel on the right to control all visuals."
        )
        total_events = int(scoped["DisNo."].count())
        peak_year = ""
        if scoped["Year"].notna().any():
            ycnt = scoped.dropna(subset=["Year"]).groupby("Year")["DisNo."].count()
            if not ycnt.empty: peak_year = int(ycnt.idxmax())
        top_type_all = _top_type(scoped)
        story_context(
            f"Showing {total_events:,} events; peak in {peak_year} — top type: {top_type_all}."
            if total_events else "No events match the current filters."
        )

        def type_sel(prefix: str) -> str:
            return constrained_type_selector(df, prefix, years, region, country, TYPE_COL)

        # ======================
        # A — Geographic Overview
        # ======================
        st.markdown("---")
        section_title("Geographic Overview")

        # A1) Choropleth — Country totals (with local Disaster Type filter)
        _anchor("sec-da-map-country")
        subsection_title("Choropleth — Total Disasters per Country")
        type_choro = type_sel("choro")
        d1 = scoped.copy()
        if type_choro and type_choro != "All" and TYPE_COL in d1.columns:
            d1 = d1[d1[TYPE_COL] == type_choro]

        agg1 = d1.groupby("Country", as_index=False)["DisNo."].count().rename(columns={"DisNo.": "Events"})
        tc = _top_country(agg1)
        story_context(
            f"Highest totals in {tc} over {years[0]}–{years[1]} (type: {type_choro})."
            if tc else f"Country totals across {years[0]}–{years[1]} (type: {type_choro})."
        )
        if not agg1.empty:
            fig_chor = px.choropleth(
                agg1, locations="Country", locationmode="country names",
                color="Events", color_continuous_scale=PALETTE_COUNTRY_CHORO
            )
            fig_chor.update_layout(
                coloraxis_colorbar=dict(title="Total Events"),
                geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", fitbounds="locations")
            )
            st.plotly_chart(fig_chor, use_container_width=True, config=PLOTLY_CFG)

        # A2) Spatial Concentration — Density & Points
        _anchor("sec-da-concentration")
        subsection_title("Spatial Concentration — Density & Event Points")
        type5 = type_sel("conc")
        d5 = scoped.copy()
        if type5 and type5 != "All" and TYPE_COL in d5.columns:
            d5 = d5[d5[TYPE_COL] == type5]
        d5 = _clip_to_region_bbox(d5, region)
        story_context("Hotspots cluster within selected scope; darker means higher concentration.")
        z = np.ones(len(d5))
        if d5.empty:
            st.info("No mappable events for the selected filters.")
        else:
            if country == "Global":
                center, zoom = dict(lat=10.0, lon=10.0), 1.35
            else:
                lat_series = pd.to_numeric(d5["Latitude"], errors="coerce").dropna()
                lon_series = pd.to_numeric(d5["Longitude"], errors="coerce").dropna()
                if len(lat_series) and len(lon_series):
                    center = dict(lat=float(lat_series.mean()), lon=float(lon_series.mean()))
                    zoom = 3.6
                else:
                    center, zoom = dict(lat=10.0, lon=10.0), 1.35

            fig_density = px.density_mapbox(
                d5, lat="Latitude", lon="Longitude", z=z, radius=18,
                center=center, zoom=zoom, mapbox_style="carto-positron",
                color_continuous_scale=PALETTE_CONC_MAP,
            )
            fig_density.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520)
            fig_scatter = go.Figure(fig_density)
            safe_name = d5.get("Event Name", pd.Series(index=d5.index, dtype=object)).fillna("").astype(str).str.strip()
            safe_type = d5.get(TYPE_COL, pd.Series(index=d5.index, dtype=object)).fillna("Event").astype(str)
            safe_country = d5.get("Country", pd.Series(index=d5.index, dtype=object)).fillna("—").astype(str)
            safe_year = pd.to_numeric(d5.get("Year"), errors="coerce").astype("Int64").astype(str).replace("<NA>", "")
            hover = np.where(
                safe_name.eq(""),
                safe_type + " — " + safe_country + np.where(safe_year.eq(""), "", " (" + safe_year + ")"),
                safe_name + np.where(safe_year.eq(""), "", " (" + safe_year + ")")
            )
            fig_scatter.add_trace(go.Scattermapbox(
                lat=d5["Latitude"], lon=d5["Longitude"], mode="markers",
                marker=dict(size=4, opacity=0.75, color="#6B7280"),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))
            st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CFG)
            st.caption("Heat shows concentration (light → dark). Missing coordinates are filled by country medians.")

        # ======================
        # B — Disasters Distribution
        # ======================
        st.markdown("---")
        section_title("Disasters Distribution")

        _anchor("sec-da-top10")
        subsection_title("Top-10 Disaster Types by Frequency")
        d2 = scoped.copy()
        freq = (
            d2.groupby(TYPE_COL, as_index=False)["DisNo."]
            .count()
            .rename(columns={"DisNo.": "Count"})
            .sort_values("Count", ascending=False)
        )
        if freq.empty:
            st.info("No data for the selected filters.")
        else:
            top10 = freq.head(10).copy()
            if len(freq) > 10:
                others_count = int(freq.iloc[10:]["Count"].sum())
                top10 = pd.concat(
                    [top10, pd.DataFrame({TYPE_COL: ["Others"], "Count": [others_count]})],
                    ignore_index=True
                )
            dom = str(top10.iloc[0][TYPE_COL]) if not top10.empty else ""
            story_context(f"Frequency is dominated by {dom} within current selection.")

            tab_bar, tab_pie = st.tabs(["Bar Chart", "Pie Chart"])

            with tab_bar:
                fig_bar = px.bar(
                    top10, x="Count", y=TYPE_COL, orientation="h",
                    color=TYPE_COL, color_discrete_sequence=PALETTE_TOPN_BARS, text="Count"
                )
                fig_bar.update_traces(textposition="outside", cliponaxis=False,
                                      hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>")
                fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, bargap=0.25, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

            with tab_pie:
                fig_pie = px.pie(
                    top10, names=TYPE_COL, values="Count", hole=0.3,
                    color=TYPE_COL, color_discrete_sequence=PALETTE_TOPN_PIE
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                      hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>")
                st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

        # ======================
        # C — Temporal Patterns
        # ======================
        st.markdown("---")
        section_title("Temporal Patterns")

        # C1) Stacked Area Timeline (Top-5 + Others)
        _anchor("sec-da-timeline")
        subsection_title("Disaster Types Over Time — Stacked Area (Top-5 + Others)")
        d3 = scoped.copy()
        if "Event Date" in d3.columns and d3["Event Date"].notna().any():
            ym = d3["Event Date"].dt.to_period("M").astype(str)
        else:
            sy = pd.to_numeric(d3["Start Year"], errors="coerce")
            sm = pd.to_numeric(d3["Start Month"], errors="coerce").fillna(1).astype(int).clip(1,12)
            ym = pd.to_datetime(dict(year=sy, month=sm, day=1), errors="coerce").dt.to_period("M").astype(str)
        d3a = d3.assign(YearMonth=ym)

        counts = (d3a.groupby(TYPE_COL, as_index=False)["DisNo."].count()
                  .rename(columns={"DisNo.":"Count"})).sort_values("Count", ascending=False)
        top_labels = set(counts[TYPE_COL].head(5).tolist())
        d3a["Type_6"] = d3a[TYPE_COL].where(d3a[TYPE_COL].isin(top_labels), "Others")
        area = (d3a.groupby(["YearMonth","Type_6"], as_index=False)["DisNo."]
                .count().rename(columns={"DisNo.":"Count"}))
        story_context("Top five types reveal shifting activity across months.")
        if not area.empty:
            area["YearMonth_dt"] = pd.to_datetime(area["YearMonth"], errors="coerce")
            area = area.sort_values("YearMonth_dt")
            uniq = [t for t in area["Type_6"].unique() if t != "Others"]
            color_map = {t: PALETTE_STACKED_AREA[i % len(PALETTE_STACKED_AREA)] for i, t in enumerate(sorted(uniq))}
            color_map["Others"] = OTHERS_COLOR
            fig_area = px.area(area, x="YearMonth_dt", y="Count", color="Type_6", color_discrete_map=color_map)
            fig_area.update_traces(hovertemplate="<b>%{x|%Y-%m}</b><br>%{fullData.name}: %{y:,}<extra></extra>")
            fig_area.update_layout(xaxis_title="Date", yaxis_title="Events", legend_title="Type", hovermode="x unified")
            st.plotly_chart(fig_area, use_container_width=True, config=PLOTLY_CFG)

        # C2) Yearly Distribution (Counts)
        _anchor("sec-da-year-dist")
        subsection_title("Yearly Counts — Line & Bars")
        dY = scoped.copy()
        type_year = type_sel("year")
        if type_year and type_year != "All" and TYPE_COL in dY.columns:
            dY = dY[dY[TYPE_COL] == type_year]
        story_context("Line shows annual totals; bars confirm counts for chosen scope.")
        year_counts = (
            dY.dropna(subset=["Year"])
              .groupby("Year", as_index=False)["DisNo."].count()
              .rename(columns={"DisNo.":"Count"})
              .sort_values("Year")
        )
        if not year_counts.empty:
            full_years = pd.DataFrame({"Year": list(range(years[0], years[1] + 1))})
            year_counts = full_years.merge(year_counts, on="Year", how="left").fillna({"Count": 0})

            tab_line, tab_bar = st.tabs(["Line Chart", "Bar Chart"])
            with tab_line:
                fig_year_line = px.line(year_counts, x="Year", y="Count", markers=True,
                                        color_discrete_sequence=PALETTE_YEAR_LINE)
                fig_year_line.update_layout(xaxis_title="Year", yaxis_title="Events",
                                            hovermode="x unified", showlegend=False)
                st.plotly_chart(fig_year_line, use_container_width=True, config=PLOTLY_CFG_NOZOOM)
            with tab_bar:
                fig_year_bar = px.bar(year_counts, x="Year", y="Count", text="Count",
                                      color_discrete_sequence=PALETTE_YEAR_BAR)
                fig_year_bar.update_traces(textposition="outside", cliponaxis=False)
                fig_year_bar.update_layout(xaxis_title="Year", yaxis_title="Events",
                                           showlegend=False, bargap=0.2, hovermode="x unified")
                st.plotly_chart(fig_year_bar, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

        # C3) Yearly Distribution by Type (Top-5 + Others)
        _anchor("sec-da-year-bytype")
        subsection_title("Yearly Counts by Type — Stacked / Grouped (Top-5 + Others)")
        dYT = scoped.dropna(subset=["Year"]).copy().astype({"Year":"int"})
        if TYPE_COL not in dYT.columns:
            st.warning(f"Column {TYPE_COL} not found; cannot build type-based distribution.")
        else:
            TOP_N = 5
            totals_scoped = (dYT.groupby(TYPE_COL, as_index=False)["DisNo."].count()
                               .rename(columns={"DisNo.":"TotalCount"})
                               .sort_values("TotalCount", ascending=False))
            top_types = totals_scoped[TYPE_COL].head(TOP_N).tolist()
            full_years = pd.DataFrame({"Year": list(range(years[0], years[1] + 1))})
            grp = (dYT.groupby(["Year", TYPE_COL], as_index=False)["DisNo."].count()
                     .rename(columns={"DisNo.":"Count"}))
            all_types = sorted(dYT[TYPE_COL].dropna().unique().tolist())
            scaffold = full_years.assign(key=1).merge(
                pd.DataFrame({TYPE_COL: all_types, "key":1}), on="key"
            ).drop(columns="key")
            ytc_full = scaffold.merge(grp, on=["Year", TYPE_COL], how="left").fillna({"Count":0}).astype({"Count":"int"})
            ytc_top = ytc_full[ytc_full[TYPE_COL].isin(top_types)].copy()
            ytc_others = (ytc_full[~ytc_full[TYPE_COL].isin(top_types)]
                          .groupby("Year", as_index=False)["Count"].sum()
                          .assign(**{TYPE_COL: "Others"}))
            ytc_plot = pd.concat([ytc_top, ytc_others], ignore_index=True)
            type_order = top_types + (["Others"] if "Others" in ytc_plot[TYPE_COL].unique() else [])
            ytc_plot[TYPE_COL] = pd.Categorical(ytc_plot[TYPE_COL], categories=type_order, ordered=True)
            ytc_plot = ytc_plot.sort_values(["Year", TYPE_COL])
            story_context("Stacked bars show mix by year; grouped contrasts type magnitudes.")
            tab_stack, tab_group = st.tabs(["Stacked Bars", "Grouped Bars"])
            common_orders = {"category_orders": {TYPE_COL: type_order}}
            with tab_stack:
                fig_stack = px.bar(
                    ytc_plot, x="Year", y="Count", color=TYPE_COL, **common_orders,
                    color_discrete_sequence=PALETTE_YEAR_BYTYPE
                )
                for tr in fig_stack.data:
                    if tr.name == "Others":
                        tr.marker.color = OTHERS_COLOR
                fig_stack.update_layout(barmode="stack", xaxis_title="Year", yaxis_title="Events",
                                        legend_title="Type", hovermode="x unified", bargap=0.15)
                st.plotly_chart(fig_stack, use_container_width=True, config=PLOTLY_CFG_NOZOOM)
            with tab_group:
                fig_group = px.bar(
                    ytc_plot, x="Year", y="Count", color=TYPE_COL, **common_orders,
                    color_discrete_sequence=PALETTE_YEAR_BYTYPE
                )
                for tr in fig_group.data:
                    if tr.name == "Others":
                        tr.marker.color = OTHERS_COLOR
                fig_group.update_layout(barmode="group", xaxis_title="Year", yaxis_title="Events",
                                        legend_title="Type", hovermode="x unified", bargap=0.20)
                st.plotly_chart(fig_group, use_container_width=True, config=PLOTLY_CFG_NOZOOM)

        # ======================
        # D — Temporal & Spatial Heat
        # ======================
        st.markdown("---")
        section_title("Temporal & Spatial Heat")

        _anchor("sec-da-calendar")
        subsection_title("Calendar Heatmap (Year × Month)")
        d6 = scoped.copy()
        type6 = type_sel("cal")
        if type6 and type6 != "All" and TYPE_COL in d6.columns:
            d6 = d6[d6[TYPE_COL] == type6]
        story_context("Calendar shading highlights months with elevated event counts.")
        if "Event Date" in d6.columns and d6["Event Date"].notna().any():
            ym6 = d6["Event Date"].dt.to_period("M").astype(str)
        else:
            sy = pd.to_numeric(d6["Start Year"], errors="coerce")
            sm = pd.to_numeric(d6["Start Month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
            ym6 = pd.to_datetime(dict(year=sy, month=sm, day=1), errors="coerce").dt.to_period("M").astype(str)
        cal = d6.assign(YearMonth=ym6)
        cal["Year"]  = pd.to_numeric(cal["YearMonth"].str[:4], errors="coerce")
        cal["Month"] = pd.to_numeric(cal["YearMonth"].str[5:7], errors="coerce")
        years_span  = list(range(years[0], years[1] + 1))
        months_span = list(range(1, 13))
        grid = pd.MultiIndex.from_product([years_span, months_span], names=["Year","Month"]).to_frame(index=False)
        heat = (cal.dropna(subset=["Year","Month"])
                .groupby(["Year","Month"], as_index=False)["DisNo."]
                .count().rename(columns={"DisNo.":"Count"}))
        heat_full = grid.merge(heat, on=["Year","Month"], how="left").fillna({"Count":0})
        pivot = heat_full.pivot(index="Year", columns="Month", values="Count").reindex(
            index=years_span, columns=months_span, fill_value=0
        )
        fig_heat = px.imshow(
            pivot, color_continuous_scale=PALETTE_CALENDAR, aspect="auto", origin="lower", zmin=0
        )
        fig_heat.update_layout(coloraxis_colorbar=dict(title="Events"), xaxis_title="Month", yaxis_title="Year")
        st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CFG)

        _anchor("sec-da-xy")
        subsection_title("Lat/Lon Density Heat (Cartesian Grid)")
        d7 = scoped.copy()
        type7 = type_sel("xy")
        if type7 and type7 != "All" and TYPE_COL in d7.columns:
            d7 = d7[d7[TYPE_COL] == type7]
        story_context("Cartesian density map reveals geographic clustering by latitude/longitude.")
        d7 = d7.dropna(subset=["Latitude","Longitude"])
        if d7.empty:
            st.info("No geocoded events for the selected filters.")
        else:
            fig_xy = px.density_heatmap(
                d7, x="Longitude", y="Latitude", nbinsx=40, nbinsy=40, color_continuous_scale=PALETTE_CONC_MAP
            )
            fig_xy.update_layout(xaxis_title="Longitude", yaxis_title="Latitude",
                                 coloraxis_colorbar=dict(title="Density"))
            st.plotly_chart(fig_xy, use_container_width=True, config=PLOTLY_CFG)

        # Footer
        st.markdown("---")
        st.caption("Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")

    # Close sticky scope wrapper
    st.markdown("</div>", unsafe_allow_html=True)
