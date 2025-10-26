"""
alerts_tab.py
-------------
Displays current GDACS disaster alerts with a cohesive visual style
and a fallback to a local snapshot if live data is unavailable.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import math
from datetime import datetime

# ✅ Ensure `src` folder (which contains `data_pipeline`) is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from data_pipeline.fetch_gdacs import fetch_gdacs


# ----------------------------
# Styling / Palette (consistent across visuals)
# ----------------------------
ALERT_COLORS = {
    "Red":    "#E56A5D",  # soft red
    "Orange": "#F2B56B",  # soft orange
    "Green":  "#59B3A9",  # soft teal
    "Unknown":"#8A8A8A",
}

TYPE_PALETTE = [
    "#D8E7F3", "#B8D5EC", "#95C2E3", "#73AFDA",
    "#539DD1", "#3A8CC5", "#2677AF", "#165F94"
]  # cohesive blue range


def _fmt(dt):
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def _center_zoom_from_points(lat_series: pd.Series, lon_series: pd.Series):
    """
    Compute an approximate (center, zoom) for Mapbox from point bounds.
    Works on older Plotly versions where layout.mapbox.fitbounds is unsupported.
    """
    lats = pd.to_numeric(lat_series, errors="coerce").dropna()
    lons = pd.to_numeric(lon_series, errors="coerce").dropna()

    if len(lats) == 0 or len(lons) == 0:
        return dict(lat=0, lon=0), 1.3  # safe global default

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    # Center
    center = dict(lat=(lat_min + lat_max) / 2.0, lon=(lon_min + lon_max) / 2.0)

    # Span (avoid zero)
    lat_span = max(1e-6, lat_max - lat_min)
    lon_span = max(1e-6, lon_max - lon_min)

    # Estimate zoom from degree spans (rough but effective)
    k = 1.4
    zoom_from_lon = math.log2(360.0 / (lon_span * k))
    zoom_from_lat = math.log2(180.0 / (lat_span * k))

    zoom = max(1.0, min(zoom_from_lon, zoom_from_lat))
    zoom = min(8.0, zoom)

    if lon_span < 0.01 and lat_span < 0.01:
        zoom = 5.0

    return center, zoom


def render():
    """Render the Alerts tab."""
    st.header("GDACS Disaster Alerts")

    st.markdown(
        "This page displays real-time GDACS alerts categorized by severity."
    )
    st.markdown("---")
    st.subheader("Insights")
    # Plain text (no info box)
    st.markdown(
        "- **Red alerts** indicate severe, large-scale disasters.\n"
        "- **Orange alerts** signal potential escalation and require monitoring.\n"
        "- **Green alerts** represent minor events or those with limited impact.\n"
        "- Use the filters to analyze specific countries or disaster types."
    )

    # ---- LOAD DATA (with snapshot fallback) ----
    df = pd.DataFrame()
    load_note = ""
    try:
        with st.spinner("Fetching live GDACS data..."):
            df = fetch_gdacs()
    except Exception as e:
        load_note = f"(live fetch failed: {e})"

    if df is None or df.empty:
        # Fallback to local cleaned snapshot
        snapshot_path = os.path.join(os.path.dirname(__file__), "../../data/cleaned_gdacs.csv")
        snapshot_path = os.path.abspath(snapshot_path)
        if os.path.exists(snapshot_path):
            df = pd.read_csv(snapshot_path)
            st.caption("Showing snapshot from data/cleaned_gdacs.csv " + load_note)
        else:
            st.warning("No live data and no snapshot available.")
            return

    # =========================================
    # SIDEBAR FILTERS (current events only)
    # =========================================
    st.sidebar.header("Filter Options")

    # 1) Alert level filter (drives country options)
    alert_filter = st.sidebar.selectbox(
        "Alert level",
        ["All", "Red", "Orange", "Green"]
    )

    # Build the alert-filtered frame first so the country list reflects level choice
    if alert_filter == "All":
        df_alert = df.copy()
    else:
        df_alert = df[df["Alert Level"].str.lower() == alert_filter.lower()].copy()

    # 2) Country single-select based on the alert-filtered frame
    countries_filtered = sorted(df_alert["Country"].dropna().unique().tolist())
    country_choice = st.sidebar.selectbox(
        "Country",
        options=["All countries"] + countries_filtered,
        index=0
    )

    # Final filtered frame used by the entire page
    if country_choice == "All countries":
        filtered_df = df_alert.copy()
    else:
        filtered_df = df_alert[df_alert["Country"] == country_choice].copy()

    # =========================
    # FIRST VISUAL: MAP (leave as is stylistically)
    # =========================
    st.markdown("---")
    st.subheader("Live GDACS Events")

    if ("Latitude" not in filtered_df.columns) or ("Longitude" not in filtered_df.columns):
        st.markdown("Map view unavailable: latitude/longitude columns not found.")
    else:
        map_df = filtered_df.copy()

        # Human-friendly disaster type names in a separate column
        type_lookup = {
            "EQ": "Earthquake",
            "FL": "Flood",
            "TC": "Tropical Cyclone",
            "DR": "Drought",
            "VO": "Volcano",
            "WF": "Wildfire",
            "LS": "Landslide",
        }
        map_df["Type Name"] = (
            map_df.get("Disaster Type", "Unknown")
            .map(type_lookup)
            .fillna(map_df.get("Disaster Type", "Unknown"))
            .replace("", "Unknown")
        )

        # ---- Dropdown: Select disaster type (All + types present) ----
        types_present = sorted(map_df["Type Name"].dropna().unique().tolist())
        type_choice = st.selectbox(
            "Select disaster type",
            options=["All"] + types_present,
            index=0,
        )
        if type_choice != "All":
            map_df = map_df[map_df["Type Name"] == type_choice]

        if map_df.empty:
            st.markdown("No live events match the selected filters.")
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

            main_size = 11
            ring_size = 14
            halo_size = 26

            for alert in map_df.get("Alert Level", "Unknown").fillna("Unknown").unique():
                sub = map_df[map_df["Alert Level"].fillna("Unknown") == alert]

                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=sub["Latitude"],
                        lon=sub["Longitude"],
                        mode="markers",
                        marker=dict(size=halo_size, color=[halo_rgba(alert)] * len(sub), opacity=1.0),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=sub["Latitude"],
                        lon=sub["Longitude"],
                        mode="markers",
                        marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=sub["Latitude"],
                        lon=sub["Longitude"],
                        mode="markers",
                        marker=dict(size=main_size, color=ALERT_COLORS.get(alert, ALERT_COLORS["Unknown"]),
                                    opacity=0.95, symbol="circle"),
                        name=str(alert),
                        customdata=sub[["hover"]],
                        hovertemplate="%{customdata[0]}<extra></extra>",
                    )
                )

            center, zoom = _center_zoom_from_points(map_df["Latitude"], map_df["Longitude"])

            fig_map.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=560,
                hoverlabel=dict(font_size=16),
                legend_title_text="Alert Level",
                uirevision=True,
                mapbox=dict(
                    style="carto-positron",
                    center=center,
                    zoom=zoom,
                ),
            )

            st.caption(
                "Zoom with your mouse/trackpad. Labels for countries/regions/towns appear as you zoom. "
                "The faint halo indicates the epicenter area; colors follow alert level."
            )
            st.plotly_chart(
                fig_map,
                use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )

    # ---- DATA DISPLAY ----
    st.subheader("List of Live Disasters")

    # Human-friendly disaster type for the table (non-destructive)
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

    # ---- LIVE ALERT DISTRIBUTION (cohesive colors) ----
    st.markdown("---")
    st.subheader("Live Alert Distribution")

    # Compact country label helper
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
        fig_bar = px.bar(
            viz_df,
            x="Alert Score",
            y="Country Label",
            color=viz_df.get("Disaster Type", "Type"),
            orientation="h",
            text="Alert Score",
            color_discrete_sequence=TYPE_PALETTE
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
        alert_counts = (
            viz_df["Alert Level"]
            .value_counts()
            .reindex(["Red", "Orange", "Green"], fill_value=0)
            .reset_index()
        )
        alert_counts.columns = ["Alert Level", "Count"]
        fig_pie = px.pie(
            alert_counts,
            names="Alert Level",
            values="Count",
            color="Alert Level",
            color_discrete_map=ALERT_COLORS,
            hole=0.3,
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0.08 if x == "Red" else 0 for x in alert_counts["Alert Level"]],
            hovertemplate="<b>%{label}</b><br>Number of Alerts: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        fig_pie.update_layout(legend_title_text="Alert Level", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- TIME SERIES (Last 30 Days) — using ONLY existing filtered data, two tabs ----
    st.markdown("---")
    st.subheader("Active Alerts Over Time (Last 30 Days)")

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
                series = series.iloc[:-1]
                curves.append(pd.DataFrame({
                    "Date": series.index.date,
                    "Active": series.values,
                    group_col: gval if pd.notna(gval) else "Unknown"
                }))

            out = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["Date","Active",group_col])
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
