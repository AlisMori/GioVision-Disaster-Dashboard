"""
alerts_tab.py
-------------
GDACS alerts page styled to match the app's enterprise theme:
- Themed section/subsection bars (gv-section-title / gv-subsection-title)
- In-page anchors that work with ?page=Alerts&section=...
- Same data logic as before, now with KPI totals above the map

Update (2025-10-31):
- Moved filter box from sidebar to the right of visuals (~20% width)
- Added short explanatory captions (6–12 words) above each visual (non-italic)
- Softer, multi-hue palette for *disaster type* bar/line charts
"""

from __future__ import annotations
import os
import sys
import math
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import streamlit as st

# ✅ Ensure `src` folder (which contains `data_pipeline`) is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from data_pipeline.fetch_gdacs import fetch_gdacs  # type: ignore


# ----------------------------
# Styling / Palette (consistent across visuals)
# ----------------------------
ALERT_COLORS = {
    "Red":    "#EA6455",  # soft red
    "Orange": "#EFB369",  # soft orange
    "Green":  "#59B3A9",  # soft teal
    "Unknown":"#8A8A8A",
}

# Cohesive blue range (kept for other visuals if needed)
TYPE_PALETTE = [
    "#D8E7F3", "#B8D5EC", "#95C2E3", "#73AFDA",
    "#539DD1", "#3A8CC5", "#2677AF", "#165F94"
]

# NEW: Soft, multi-hue colors for *disaster type* bar/line charts
TYPE_SOFT_COLORS = [
    "#A3C4F3",  # pastel blue
    "#FFCAD4",  # pastel pink
    "#CDEAC0",  # pastel green
    "#FDE2A0",  # pastel yellow
    "#D7BCE8",  # pastel purple
    "#B8E0D2",  # pastel teal
    "#F6C1B0",  # pastel coral
    "#C9E4F1",  # pastel sky
]


# ----------------------------
# Small helpers (titles, anchors, map bounds)
# ----------------------------
def _fmt(dt):
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def _anchor(id_: str):
    """Invisible HTML anchor for smooth scrolling targets."""
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)


def section_title(text: str):
    """Theme-aligned section bar (same visuals as the app)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)


def subsection_title(text: str):
    """Theme-aligned subsection bar."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)


def _center_zoom_from_points(lat_series: pd.Series, lon_series: pd.Series):
    """Compute center and zoom for Mapbox from points."""
    lats = pd.to_numeric(lat_series, errors="coerce").dropna()
    lons = pd.to_numeric(lon_series, errors="coerce").dropna()

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
    zoom = min(8.0, zoom)

    if lon_span < 0.01 and lat_span < 0.01:
        zoom = 5.0

    return center, zoom


# ----------------------------
# Main render
# ----------------------------
def render():
    """Render the Alerts tab (styled to match the app)."""

    # =========================
    # OVERVIEW
    # =========================
    _anchor("sec-alerts-overview")
    section_title("Overview")
    st.markdown(
        "This page displays real-time GDACS alerts categorized by severity. "
        "Use the filters to focus on specific countries or disaster types."
    )
    st.markdown(
        "- **Red alerts** indicate severe, large-scale disasters.\n"
        "- **Orange alerts** signal potential escalation and require monitoring.\n"
        "- **Green alerts** represent minor events or those with limited impact."
    )

    # =========================
    # LOAD DATA (live with snapshot fallback)
    # =========================
    df = pd.DataFrame()
    load_note = ""
    try:
        with st.spinner("Fetching live GDACS data..."):
            df = fetch_gdacs()
    except Exception as e:
        load_note = f"(live fetch failed: {e})"

    if df is None or df.empty:
        snapshot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cleaned_gdacs.csv"))
        if os.path.exists(snapshot_path):
            df = pd.read_csv(snapshot_path)
            st.caption("Showing snapshot from data/cleaned_gdacs.csv " + load_note)
        else:
            st.warning("No live data and no snapshot available.")
            return

    # =========================================
    # FILTERS (moved from sidebar to right of visuals)
    # =========================================
    alert_levels = ["All", "Red", "Orange", "Green"]

    # Layout: 80% visual area, 20% slim filter panel (same row as the map)
    left_col, right_col = st.columns([8, 2])  # widened filter area

    with right_col:
        st.markdown("<div class='gv-subsection-title'>Filters</div>", unsafe_allow_html=True)
        alert_filter = st.selectbox("Alert level", alert_levels, index=0)

        if alert_filter == "All":
            df_alert = df.copy()
        else:
            df_alert = df[df["Alert Level"].str.lower() == alert_filter.lower()].copy()

        countries_filtered = sorted(df_alert.get("Country", pd.Series(dtype=str)).dropna().unique().tolist())
        country_choice = st.selectbox("Country", options=["All countries"] + countries_filtered, index=0)

        # Keep selections in session for consistency across visuals
        st.session_state.setdefault("_alerts_filter", {})
        st.session_state["_alerts_filter"].update({
            "alert_filter": alert_filter,
            "country_choice": country_choice,
        })

    # Apply selected filters to build `filtered_df`
    if st.session_state.get("_alerts_filter", {}).get("alert_filter", "All") == "All":
        df_alert = df.copy()
    else:
        a = st.session_state["_alerts_filter"]["alert_filter"].lower()
        df_alert = df[df["Alert Level"].str.lower() == a].copy()

    if st.session_state.get("_alerts_filter", {}).get("country_choice", "All countries") == "All countries":
        filtered_df = df_alert.copy()
    else:
        c = st.session_state["_alerts_filter"]["country_choice"]
        filtered_df = df_alert[df_alert["Country"] == c].copy()

    # =========================
    # MAP
    # =========================
        # =========================
    # MAP (ended in gray; KPI shows total & active breakdowns)
    # =========================
    with left_col:
        st.markdown("---")
        _anchor("sec-alerts-map")
        section_title("Live GDACS Events")
        st.markdown("Map of current disaster alerts and their levels.")

        if ("Latitude" not in filtered_df.columns) or ("Longitude" not in filtered_df.columns):
            st.markdown("Map view unavailable: latitude/longitude columns not found.")
        else:
            map_df = filtered_df.copy()

            # Prefer Started/Ended; otherwise fall back to Start Date/End Date
            start_col = "Started" if "Started" in map_df.columns else "Start Date"
            end_col   = "Ended"   if "Ended"   in map_df.columns else "End Date"
            now_utc = pd.Timestamp.utcnow()

            map_df["_start_dt"] = pd.to_datetime(map_df.get(start_col), errors="coerce", utc=True)
            map_df["_end_dt"]   = pd.to_datetime(map_df.get(end_col), errors="coerce", utc=True)

            # Human-friendly type names (unchanged)
            type_lookup = {
                "EQ": "Earthquake", "FL": "Flood", "TC": "Tropical Cyclone",
                "DR": "Drought", "VO": "Volcano", "WF": "Wildfire", "LS": "Landslide",
            }
            base_type = map_df.get("Disaster Type", pd.Series(index=map_df.index, dtype=object))
            map_df["Type Name"] = (
                base_type.map(type_lookup)
                         .fillna(base_type)
                         .replace("", "Unknown")
            )

            # Optional type filter (unchanged behavior)
            types_present = sorted(map_df["Type Name"].dropna().unique().tolist())
            type_choice = st.selectbox("Select disaster type", options=["All"] + types_present, index=0)
            if type_choice != "All":
                map_df = map_df[map_df["Type Name"] == type_choice]

            # ---------- KPI rows above the map ----------
            # Total (within current filters/view)
            total_n = int(len(map_df))
            total_lvl = (
                map_df.get("Alert Level", pd.Series(dtype=str))
                      .fillna("Unknown").value_counts()
                      .reindex(["Red", "Orange", "Green"], fill_value=0)
            )

            # Active = not ended (Ended is NaT or in the future)
            active_df = map_df[(map_df["_end_dt"].isna()) | (map_df["_end_dt"] >= now_utc)].copy()
            active_n = int(len(active_df))
            active_lvl = (
                active_df.get("Alert Level", pd.Series(dtype=str))
                         .fillna("Unknown").value_counts()
                         .reindex(["Red", "Orange", "Green"], fill_value=0)
            )

            st.markdown("<div style='margin:6px 0;'></div>", unsafe_allow_html=True)
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1: st.metric("Total Alerts", f"{total_n:,}")
            with r1c2: st.metric("Total Red",    f"{int(total_lvl['Red']):,}")
            with r1c3: st.metric("Total Orange", f"{int(total_lvl['Orange']):,}")
            with r1c4: st.metric("Total Green",  f"{int(total_lvl['Green']):,}")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1: st.metric("Active Alerts", f"{active_n:,}")
            with r2c2: st.metric("Active Red",    f"{int(active_lvl['Red']):,}")
            with r2c3: st.metric("Active Orange", f"{int(active_lvl['Orange']):,}")
            with r2c4: st.metric("Active Green",  f"{int(active_lvl['Green']):,}")
            # -------------------------------------------

            if map_df.empty:
                st.markdown("No events match the selected filters.")
            else:
                event_name = map_df.get("Event Name", "Event").fillna("Event")
                country    = map_df.get("Country", "—").fillna("—")
                severity   = map_df.get("Severity", "—").fillna("—")

                map_df["hover"] = (
                    "<b>" + event_name + "</b><br>"
                    + "Type: " + map_df["Type Name"].fillna("Unknown") + "<br>"
                    + "Alert: " + map_df.get("Alert Level", "Unknown").fillna("Unknown") + "<br>"
                    + "Country: " + country + "<br>"
                    + "Start: " + map_df["_start_dt"].map(_fmt) + "<br>"
                    + "End: "   + map_df["_end_dt"].map(_fmt) + "<br>"
                    + "Severity: " + severity
                )

                # Split into active vs ended; ended colored gray
                ended_df  = map_df[map_df["_end_dt"].notna() & (map_df["_end_dt"] < now_utc)].copy()
                live_df   = map_df[(map_df["_end_dt"].isna()) | (map_df["_end_dt"] >= now_utc)].copy()

                def halo_rgba(hex_color: str):
                    base = hex_color.lstrip("#")
                    r = int(base[0:2], 16); g = int(base[2:4], 16); b = int(base[4:6], 16)
                    return f"rgba({r},{g},{b},0.25)"

                fig_map = go.Figure()
                main_size, ring_size, halo_size = 11, 14, 26

                # 1) Ended layer (single gray group)
                if not ended_df.empty:
                    gray = ALERT_COLORS.get("Unknown", "#8A8A8A")
                    fig_map.add_trace(go.Scattermapbox(
                        lat=ended_df["Latitude"], lon=ended_df["Longitude"], mode="markers",
                        marker=dict(size=halo_size, color=[halo_rgba(gray)] * len(ended_df), opacity=1.0),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=ended_df["Latitude"], lon=ended_df["Longitude"], mode="markers",
                        marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=ended_df["Latitude"], lon=ended_df["Longitude"], mode="markers",
                        marker=dict(size=main_size, color=gray, opacity=0.9, symbol="circle"),
                        name="Ended",
                        customdata=ended_df[["hover"]],
                        hovertemplate="%{customdata[0]}<extra></extra>",
                    ))

                # 2) Active layers by Alert Level (keep your colors)
                for alert in ["Red", "Orange", "Green"]:
                    sub = live_df[live_df.get("Alert Level", "").fillna("") == alert]
                    if sub.empty:
                        continue
                    col = ALERT_COLORS.get(alert, ALERT_COLORS["Unknown"])
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=halo_size, color=[halo_rgba(col)] * len(sub), opacity=1.0),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=main_size, color=col, opacity=0.95, symbol="circle"),
                        name=alert,
                        customdata=sub[["hover"]],
                        hovertemplate="%{customdata[0]}<extra></extra>",
                    ))

                # Fit bounds to everything in view (active + ended)
                center, zoom = _center_zoom_from_points(map_df["Latitude"], map_df["Longitude"])

                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=560,
                    hoverlabel=dict(font_size=16),
                    legend_title_text="Status / Level",
                    uirevision=True,
                    mapbox=dict(style="carto-positron", center=center, zoom=zoom),
                )

                st.caption(
                    "Ended events are shown in gray. Active events use Red/Orange/Green level colors. "
                    "Zoom to reveal place labels; halos indicate epicenter area."
                )
                st.plotly_chart(
                    fig_map, use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                )

    # =========================
    # TABLE OF LIVE DISASTERS
    # =========================
    st.markdown("---")
    _anchor("sec-alerts-table")
    section_title("List of Live Disasters")
    st.markdown("Current alerts with timing, severity, and quick links.")

    disaster_legend = {
        "EQ": "Earthquake", "FL": "Flood", "TC": "Tropical Cyclone",
        "DR": "Drought", "VO": "Volcano", "WF": "Wildfire", "LS": "Landslide"
    }
    table_df = filtered_df.copy()
    if "Disaster Type" in table_df.columns:
        table_df["Disaster Type"] = table_df["Disaster Type"].replace(disaster_legend)

    display_df = table_df[[
        "Event Name", "Country", "Disaster Type", "Alert Level",
        "Start Date", "End Date", "Alert Score", "url"
    ]].reset_index(drop=True)
    display_df.index += 1
    display_df.index.name = "#"

    st.dataframe(display_df)

    # =========================
    # DISTRIBUTION
    # =========================
    st.markdown("---")
    _anchor("sec-alerts-distribution")
    section_title("Live Alert Distribution")
    st.markdown("Where alert scores concentrate across countries and types.")

    def _compact_country_label(s: str) -> str:
        if not isinstance(s, str) or not s.strip():
            return "—"
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) <= 2:
            return ", ".join(parts)
        return ", ".join(parts[:2]) + " …"

    viz_df = table_df.copy()
    viz_df["Country Label"] = viz_df["Country"].apply(_compact_country_label)

    tabs = st.tabs(["Bar Chart", "Pie Chart"])

    with tabs[0]:
        st.markdown("Alert scores by country, colored by disaster type.")
        fig_bar = px.bar(
            viz_df, x="Alert Score", y="Country Label",
            color=viz_df.get("Disaster Type", "Type"),
            orientation="h", text="Alert Score",
            color_discrete_sequence=TYPE_SOFT_COLORS,  # ⬅️ soft multi-hue
        )
        fig_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            bargap=0.25,
            legend_title_text="Disaster Type"
        )
        fig_bar.update_traces(
            texttemplate="%{text}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Type: %{legendgroup}<br>Score: %{x}<extra></extra>"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tabs[1]:
        st.markdown("Share of alerts by Red/Orange/Green levels.")
        alert_counts = (
            viz_df["Alert Level"].value_counts()
            .reindex(["Red", "Orange", "Green"], fill_value=0)
            .reset_index()
        )
        alert_counts.columns = ["Alert Level", "Count"]
        fig_pie = px.pie(
            alert_counts, names="Alert Level", values="Count",
            color="Alert Level", color_discrete_map=ALERT_COLORS, hole=0.3,
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0.08 if x == "Red" else 0 for x in alert_counts["Alert Level"]],
            hovertemplate="<b>%{label}</b><br>Number of Alerts: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        fig_pie.update_layout(legend_title_text="Alert Level", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # =========================
    # TIME SERIES (Last 30 Days)
    # =========================
    st.markdown("---")
    _anchor("sec-alerts-timeseries")
    section_title("Active Alerts Over Time (Last 30 Days)")
    st.markdown("Daily counts of active alerts during the last month.")

    ts_df = table_df.copy()
    ts_df["Start Date"] = pd.to_datetime(ts_df.get("Start Date"), errors="coerce", utc=True)
    ts_df["End Date"]   = pd.to_datetime(ts_df.get("End Date"), errors="coerce", utc=True)

    end_window = pd.Timestamp.utcnow().normalize()
    start_window = end_window - pd.Timedelta(days=29)

    ts_df["End Date"] = ts_df["End Date"].fillna(end_window + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    mask = (ts_df["Start Date"] <= end_window) & (ts_df["End Date"] >= start_window)
    ts_df = ts_df[mask].copy()

    if ts_df.empty:
        st.markdown("No alerts intersect the last 30 days for the current filters.")
    else:
        def build_active_timeline(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
            deltas = []
            frame["S"] = frame["Start Date"].clip(lower=start_window, upper=end_window).dt.normalize()
            frame["E"] = frame["End Date"].clip(lower=start_window, upper=end_window).dt.normalize()

            for gval, sub in frame.groupby(group_col, dropna=False):
                if sub.empty:
                    continue
                deltas.append(pd.DataFrame({"Date": sub["S"], "Group": gval, "Delta": 1}))
                deltas.append(pd.DataFrame({"Date": sub["E"] + pd.Timedelta(days=1), "Group": gval, "Delta": -1}))

            if not deltas:
                return pd.DataFrame(columns=["Date", "Active", group_col])

            delta_df = pd.concat(deltas, ignore_index=True)
            full_idx = pd.date_range(start_window, end_window + pd.Timedelta(days=1), freq="D")

            curves = []
            for gval, sub in delta_df.groupby("Group", dropna=False):
                series = sub.groupby("Date")["Delta"].sum().reindex(full_idx, fill_value=0).cumsum()
                series = series.iloc[:-1]  # drop last (end marker)
                curves.append(pd.DataFrame({
                    "Date": series.index.date,
                    "Active": series.values,
                    group_col: gval if pd.notna(gval) else "Unknown"
                }))

            out = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["Date","Active",group_col])
            return out[(out["Date"] >= start_window.date()) & (out["Date"] <= end_window.date())]

        tab1, tab2 = st.tabs(["By Alert Level", "By Disaster Type"])

        with tab1:
            st.markdown("Active alerts per day, grouped by level.")
            lvl_tl = build_active_timeline(ts_df, "Alert Level")
            if lvl_tl.empty:
                st.markdown("No Red/Orange/Green alerts to plot for the last 30 days.")
            else:
                map_df["_start_dt"] = pd.to_datetime(map_df.get("Start Date"), errors="coerce", utc=True)
                map_df["_end_dt"]   = pd.to_datetime(map_df.get("End Date"), errors="coerce", utc=True)
                event_name = map_df.get("Event Name", "Event").fillna("Event")
                country    = map_df.get("Country", "—").fillna("—")
                severity   = map_df.get("Severity", "—").fillna("—")

                map_df["hover"] = (
                    "<b>" + event_name + "</b><br>"
                    + "Type: " + map_df["Type Name"].fillna("Unknown") + "<br>"
                    + "Alert: " + map_df.get("Alert Level", "Unknown").fillna("Unknown") + "<br>"
                    + "Country: " + country + "<br>"
                    + "Start: " + map_df["_start_dt"].map(_fmt) + "<br>"
                    + "End: "   + map_df["_end_dt"].map(_fmt) + "<br>"
                    + "Severity: " + severity
                )

                from plotly import graph_objects as go
                def halo_rgba(alert):
                    base = ALERT_COLORS.get(alert, ALERT_COLORS["Unknown"]).lstrip("#")
                    r = int(base[0:2], 16); g = int(base[2:4], 16); b = int(base[4:6], 16)
                    return f"rgba({r},{g},{b},0.25)"

                fig_map = go.Figure()
                main_size, ring_size, halo_size = 11, 14, 26

                for alert in map_df.get("Alert Level", "Unknown").fillna("Unknown").unique():
                    sub = map_df[map_df["Alert Level"].fillna("Unknown") == alert]

                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=halo_size, color=[halo_rgba(alert)] * len(sub), opacity=1.0),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=main_size, color=ALERT_COLORS.get(alert, ALERT_COLORS["Unknown"]),
                                    opacity=0.95, symbol="circle"),
                        name=str(alert),
                        customdata=sub[["hover"]],
                        hovertemplate="%{customdata[0]}<extra></extra>",
                    ))

                center, zoom = _center_zoom_from_points(map_df["Latitude"], map_df["Longitude"])

                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=560,
                    hoverlabel=dict(font_size=16),
                    legend_title_text="Disaster Level",
                    uirevision=True,
                    mapbox=dict(style="carto-positron", center=center, zoom=zoom),
                )

                st.caption(
                    "Zoom with your mouse/trackpad. Labels appear as you zoom. "
                    "The faint halo indicates the epicenter area; colors follow the disaster level."
                )
                st.plotly_chart(
                    fig_map, use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                )

        with tab2:
            st.markdown("Active alerts per day, grouped by disaster type.")
            type_tl = build_active_timeline(ts_df, "Disaster Type")
            if type_tl.empty:
                st.markdown("No disasters to plot for the last 30 days.")
            else:
                fig_type = px.line(
                    type_tl, x="Date", y="Active", color="Disaster Type",
                    markers=True, color_discrete_sequence=TYPE_SOFT_COLORS,  # ⬅️ soft multi-hue
                )
                fig_bar.update_traces(
                    texttemplate="%{text}",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Type: %{marker.color}<br>Score: %{x}<extra></extra>"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with tabs[1]:
                fig_pie = px.pie(
                    viz_df, names="Disaster Type",
                    color_discrete_sequence=TYPE_PALETTE
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            # =========================
            # TIME SERIES (Last 30 Days)
            # =========================
            st.markdown("---")
            _anchor("sec-alerts-timeseries")
            section_title("Active Alerts Over Time (Last 30 Days)")

            ts_df = table_df.copy()
            ts_df["Start Date"] = pd.to_datetime(ts_df.get("Start Date"), errors="coerce", utc=True)
            ts_df["End Date"] = pd.to_datetime(ts_df.get("End Date"), errors="coerce", utc=True)

            end_window = pd.Timestamp.utcnow().normalize()
            start_window = end_window - pd.Timedelta(days=29)

            ts_df["End Date"] = ts_df["End Date"].fillna(end_window + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            mask = (ts_df["Start Date"] <= end_window) & (ts_df["End Date"] >= start_window)
            ts_df = ts_df[mask].copy()

            if ts_df.empty:
                st.markdown("No alerts intersect the last 30 days for the current filters.")
            else:
                def build_active_timeline(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
                    deltas = []
                    frame["S"] = frame["Start Date"].clip(lower=start_window, upper=end_window).dt.normalize()
                    frame["E"] = frame["End Date"].clip(lower=start_window, upper=end_window).dt.normalize()

                    for gval, sub in frame.groupby(group_col, dropna=False):
                        if sub.empty:
                            continue
                        deltas.append(pd.DataFrame({"Date": sub["S"], "Group": gval, "Delta": 1}))
                        deltas.append(
                            pd.DataFrame({"Date": sub["E"] + pd.Timedelta(days=1), "Group": gval, "Delta": -1}))

                    if not deltas:
                        return pd.DataFrame(columns=["Date", "Active", group_col])

                    delta_df = pd.concat(deltas, ignore_index=True)
                    full_idx = pd.date_range(start_window, end_window + pd.Timedelta(days=1), freq="D")

                    curves = []
                    for gval, sub in delta_df.groupby("Group", dropna=False):
                        series = sub.groupby("Date")["Delta"].sum().reindex(full_idx, fill_value=0).cumsum()
                        series = series.iloc[:-1]
                        curves.append(pd.DataFrame({
                            "Date": series.index.date,
                            "Active": series.values,
                            group_col: gval if pd.notna(gval) else "Unknown"
                        }))

                    out = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
                        columns=["Date", "Active", group_col])
                    return out[(out["Date"] >= start_window.date()) & (out["Date"] <= end_window.date())]

                tab1, tab2 = st.tabs(["By Alert Level", "By Disaster Type"])

                with tab1:
                    lvl_tl = build_active_timeline(ts_df, "Alert Level")
                    if lvl_tl.empty:
                        st.markdown("No Red/Orange/Green alerts to plot for the last 30 days.")
                    else:
                        fig_lvl = px.line(
                            lvl_tl, x="Date", y="Active", color="Alert Level",
                            markers=True, color_discrete_map=ALERT_COLORS,
                        )
                        fig_lvl.update_layout(
                            title="Active Alerts by Alert Level (Daily)",
                            xaxis_title="Date", yaxis_title="Number of Active Alerts",
                            legend_title="Alert Level", hovermode="x unified"
                        )
                        fig_lvl.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>")
                        st.plotly_chart(fig_lvl, use_container_width=True)

                with tab2:
                    type_tl = build_active_timeline(ts_df, "Disaster Type")
                    if type_tl.empty:
                        st.markdown("No disasters to plot for the last 30 days.")
                    else:
                        fig_type = px.line(
                            type_tl, x="Date", y="Active", color="Disaster Type",
                            markers=True, color_discrete_sequence=TYPE_PALETTE,
                        )
                        fig_type.update_layout(
                            title="Active Alerts by Disaster Type (Daily)",
                            xaxis_title="Date", yaxis_title="Number of Active Alerts",
                            legend_title="Disaster Type", hovermode="x unified"
                        )
                        fig_type.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>")
                        st.plotly_chart(fig_type, use_container_width=True)

            # ---- FOOTER ----
            st.markdown("---")
            st.caption("Source: Global Disaster Alert and Coordination System")





