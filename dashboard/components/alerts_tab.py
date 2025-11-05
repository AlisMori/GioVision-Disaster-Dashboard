# dashboard/components/alerts_tab.py

"""
alerts_tab.py
GDACS alerts page styled to match the app's enterprise theme.
"""

import os
import sys
import math
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import streamlit as st

# ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from data_pipeline.fetch_gdacs import fetch_gdacs  # type: ignore


# ----------------------------
# Styling / palette
# ----------------------------
ALERT_COLORS = {
    "Red":    "#EA6455",
    "Orange": "#EFB369",
    "Green":  "#59B3A9",
    "Unknown":"#8A8A8A",
}

PALETTE_NATURAL = [
    "#3B82F6", "#EF4444", "#F97316", "#22C55E", "#06B6D4",
    "#6366F1", "#EAB308", "#8B5CF6", "#14B8A6", "#F43F5E",
]


# ----------------------------
# Small helpers
# ----------------------------
def _fmt(dt):
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return "-"


def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)


def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)


def story_context(text: str):
    st.markdown(f'<div class="gv-context">{text}</div>', unsafe_allow_html=True)


def _center_zoom_from_points(lat_series: pd.Series, lon_series: pd.Series):
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


def _halo_rgba_from_level(level_or_hex: str):
    if level_or_hex.startswith("#"):
        base = level_or_hex.lstrip("#")
    else:
        base = ALERT_COLORS.get(level_or_hex, ALERT_COLORS["Unknown"]).lstrip("#")
    r = int(base[0:2], 16)
    g = int(base[2:4], 16)
    b = int(base[4:6], 16)
    return f"rgba({r},{g},{b},0.25)"


# ----------------------------
# Main render
# ----------------------------
def render():
    # intro
    section_title("Overview")
    st.markdown(
        "Displays GDACS alerts grouped by severity so operators can quickly see where high impact events are happening."
    )
    st.markdown(
        "- Red alerts: severe, large scale.\n"
        "- Orange alerts: watch closely.\n"
        "- Green alerts: minor or limited impact."
    )

    # load data
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

    # normalize types once
    disaster_legend = {
        "EQ": "Earthquake",
        "FL": "Flood",
        "TC": "Tropical Cyclone",
        "DR": "Drought",
        "VO": "Volcano",
        "WF": "Wildfire",
        "LS": "Landslide",
    }
    if "Disaster Type" in df.columns:
        df["Disaster Type"] = df["Disaster Type"].replace(disaster_legend)

    # date cols
    df["_start_dt"] = pd.to_datetime(df.get("Start Date"), errors="coerce", utc=True)
    df["_end_dt"]   = pd.to_datetime(df.get("End Date"), errors="coerce", utc=True)
    now_utc = pd.Timestamp.utcnow()
    today_start = now_utc.normalize()

    # two scopes
    df_live_base = df[(df["_end_dt"].isna()) | (df["_end_dt"] >= today_start)].copy()
    df_recent_base = df.copy()

    # 🔹 common section ABOVE tabs
    section_title("Current Alerts & Locations")
    story_context("Switch between live GDACS alerts and the latest recent alerts for short-term review.")

    # tabs
    tab_live, tab_recent = st.tabs(["Live alerts", "Recent alerts"])

    # =========================================================
    # TAB 1: LIVE ALERTS
    # =========================================================
    with tab_live:

        col_main, col_filter = st.columns([4, 1], gap="large")

        with col_filter:
            st.markdown('<div class="gv-filter-card">', unsafe_allow_html=True)
            subsection_title("Filters")

            alert_options = ["All"] + sorted(
                df_live_base["Alert Level"].dropna().str.capitalize().unique().tolist()
            )
            alert_filter = st.selectbox("Alert Level", alert_options, key="live_alert_level")
            if alert_filter == "All":
                df_alert = df_live_base.copy()
            else:
                df_alert = df_live_base[
                    df_live_base["Alert Level"].str.lower() == alert_filter.lower()
                ].copy()

            countries = sorted(df_alert["Country"].dropna().unique().tolist())
            country_choice = st.selectbox(
                "Country",
                options=["All countries"] + countries,
                index=0,
                key="live_country"
            )
            if country_choice == "All countries":
                live_df = df_alert.copy()
            else:
                live_df = df_alert[df_alert["Country"] == country_choice].copy()

            st.markdown("</div>", unsafe_allow_html=True)

        with col_main:
            live_df["_start_dt"] = pd.to_datetime(live_df.get("Start Date"), errors="coerce", utc=True)
            live_df["_end_dt"]   = pd.to_datetime(live_df.get("End Date"), errors="coerce", utc=True)
            active_df = live_df

            subsection_title("Alerts map")
            story_context(
                "Map highlights where active alerts are currently located. Color shows alert level, halo shows intensity zone. Useful to scan geography first."
            )
            st.markdown("", unsafe_allow_html=True)

            # KPI
            live_counts = (
                active_df.get("Alert Level", pd.Series(dtype=str))
                         .fillna("Unknown").value_counts()
                         .reindex(["Red", "Orange", "Green", "Unknown"], fill_value=0)
            )
            k1, k2, k3, k4 = st.columns(4)
            with k1: st.metric("Live alerts", f"{len(active_df):,}")
            with k2: st.metric("Red", f"{int(live_counts['Red']):,}")
            with k3: st.metric("Orange", f"{int(live_counts['Orange']):,}")
            with k4: st.metric("Green", f"{int(live_counts['Green']):,}")

            fig = go.Figure()
            main_size, ring_size, halo_size = 11, 14, 26

            if active_df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=[0], lon=[0], mode="markers",
                    marker=dict(size=0),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=560,
                    mapbox=dict(style="carto-positron", center=dict(lat=0, lon=0), zoom=1.1),
                    annotations=[
                        dict(
                            text="No active alerts at the current time",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=1.0,
                            y=0.0,
                            xanchor="right",
                            yanchor="bottom",
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#EA6455",
                            borderwidth=1,
                            font=dict(size=13),
                        )
                    ],
                )
            else:
                for level in ["Red", "Orange", "Green"]:
                    lvl_series = active_df.get("Alert Level", pd.Series([""] * len(active_df), index=active_df.index))
                    mask = (lvl_series.fillna("").astype(str).to_numpy() == level)
                    if not mask.any():
                        continue
                    sub = active_df.loc[mask].copy()
                    col = ALERT_COLORS.get(level, ALERT_COLORS["Unknown"])

                    # extra fields for hover
                    typ = sub.get("Disaster Type", "Unknown").fillna("Unknown")
                    mag = sub.get("Severity", "-").fillna("-")

                    sub["hover"] = (
                        "<b>" + sub.get("Event Name", "Event").fillna("Event") + "</b><br>"
                        + "Type: " + typ + "<br>"
                        + "Severity: " + mag.astype(str) + "<br>"
                        + "Alert: " + sub.get("Alert Level", "Unknown").fillna("Unknown") + "<br>"
                        + "Country: " + sub.get("Country", "-").fillna("-") + "<br>"
                        + "Start: " + sub["_start_dt"].map(_fmt) + "<br>"
                        + "End: "   + sub["_end_dt"].map(_fmt)
                    )

                    # halo
                    fig.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=halo_size, color=[_halo_rgba_from_level(level)] * len(sub), opacity=1.0),
                        hoverinfo="skip", showlegend=False,
                    ))
                    # ring
                    fig.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    # main
                    fig.add_trace(go.Scattermapbox(
                        lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                        marker=dict(size=main_size, color=col, opacity=0.95, symbol="circle"),
                        name=level,
                        text=sub["hover"], hovertemplate="%{text}<extra></extra>",
                    ))

                center, zoom = _center_zoom_from_points(active_df["Latitude"], active_df["Longitude"])
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=560,
                    hoverlabel=dict(font_size=16),
                    legend_title_text="Alert Level",
                    mapbox=dict(style="carto-positron", center=center, zoom=zoom),
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )

            # table
            subsection_title("List of live alerts")
            story_context("While the map shows where alerts are happening, this table helps us track the alerts in real time, supporting faster decision-making and guiding where resources or further investigation may be needed.")
            st.markdown("", unsafe_allow_html=True)
            if active_df.empty:
                st.info("No active alerts at the current time.")
            else:
                display_live = active_df[[
                    "Event Name", "Country", "Disaster Type", "Alert Level",
                    "Start Date", "End Date", "Alert Score", "url"
                ]].reset_index(drop=True)
                display_live.index += 1
                display_live.index.name = "#"
                st.dataframe(display_live, use_container_width=True)

    # =========================================================
    # TAB 2: RECENT ALERTS
    # =========================================================
    with tab_recent:

        col_main, col_filter = st.columns([4, 1], gap="large")

        with col_filter:
            st.markdown('<div class="gv-filter-card">', unsafe_allow_html=True)
            subsection_title("Filters")

            alert_options = ["All"] + sorted(
                df_recent_base["Alert Level"].dropna().str.capitalize().unique().tolist()
            )
            alert_filter = st.selectbox("Alert Level", alert_options, key="section2_alert_level")

            if alert_filter == "All":
                df_alert = df_recent_base.copy()
            else:
                df_alert = df_recent_base[
                    df_recent_base["Alert Level"].str.lower() == alert_filter.lower()
                ].copy()

            countries = sorted(df_alert["Country"].dropna().unique().tolist())
            country_choice = st.selectbox(
                "Country",
                options=["All countries"] + countries,
                index=0,
                key="section2_country"
            )
            if country_choice == "All countries":
                recent_df = df_alert.copy()
            else:
                recent_df = df_alert[df_alert["Country"] == country_choice].copy()

            st.markdown("</div>", unsafe_allow_html=True)

        with col_main:
            recent_df["_start_dt"] = pd.to_datetime(recent_df.get("Start Date"), errors="coerce", utc=True)
            recent_df["_end_dt"]   = pd.to_datetime(recent_df.get("End Date"), errors="coerce", utc=True)
            active_recent = recent_df[(recent_df["_end_dt"].isna()) | (recent_df["_end_dt"] >= today_start)].copy()
            ended_recent  = recent_df[recent_df["_end_dt"].notna() & (recent_df["_end_dt"] < today_start)].copy()

            subsection_title("Alerts map")
            story_context(
                "Map shows all recent alerts for the selected filters. Active events are layered on top. Useful for short term review and debriefs."
            )
            st.markdown("", unsafe_allow_html=True)

            # KPI
            rec_counts = (
                recent_df.get("Alert Level", pd.Series(dtype=str))
                         .fillna("Unknown").str.capitalize().value_counts()
                         .reindex(["Red", "Orange", "Green", "Unknown"], fill_value=0)
            )
            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("Recent alerts", f"{len(recent_df):,}")
            with r2: st.metric("Red", f"{int(rec_counts['Red']):,}")
            with r3: st.metric("Orange", f"{int(rec_counts['Orange']):,}")
            with r4: st.metric("Green", f"{int(rec_counts['Green']):,}")

            fig_r = go.Figure()
            main_size, ring_size, halo_size = 11, 14, 26
            shown_levels = set()

            # ended first
            for level in ["Red", "Orange", "Green", "Unknown"]:
                sub = ended_recent[ended_recent["Alert Level"].fillna("Unknown").str.capitalize() == level]
                if sub.empty:
                    continue
                col = ALERT_COLORS.get(level, ALERT_COLORS["Unknown"])

                # extra fields
                typ = sub.get("Disaster Type", "Unknown").fillna("Unknown")
                mag = sub.get("Severity", "-").fillna("-")

                fig_r.add_trace(go.Scattermapbox(
                    lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                    marker=dict(size=halo_size, color=[_halo_rgba_from_level(level)] * len(sub), opacity=1.0),
                    hoverinfo="skip", showlegend=False,
                ))
                fig_r.add_trace(go.Scattermapbox(
                    lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                    marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                    hoverinfo="skip", showlegend=False,
                ))
                sub["hover"] = (
                    "<b>" + sub.get("Event Name", "Event").fillna("Event") + "</b><br>"
                    + "Type: " + typ + "<br>"
                    + "Severity: " + mag.astype(str) + "<br>"
                    + "Alert: " + sub.get("Alert Level", "Unknown").fillna("Unknown") + "<br>"
                    + "Country: " + sub.get("Country", "-").fillna("-") + "<br>"
                    + "Start: " + sub["_start_dt"].map(_fmt) + "<br>"
                    + "End: "   + sub["_end_dt"].map(_fmt)
                )
                fig_r.add_trace(go.Scattermapbox(
                    lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                    marker=dict(size=main_size, color=col, opacity=0.9, symbol="circle"),
                    name=level,
                    showlegend=(level not in shown_levels),
                    text=sub["hover"], hovertemplate="%{text}<extra></extra>",
                ))
                shown_levels.add(level)

            # active on top
            for level in ["Red", "Orange", "Green"]:
                sub = active_recent[active_recent["Alert Level"].fillna("").str.capitalize() == level]
                if sub.empty:
                    continue
                col = ALERT_COLORS.get(level, ALERT_COLORS["Unknown"])

                typ = sub.get("Disaster Type", "Unknown").fillna("Unknown")
                mag = sub.get("Severity", "-").fillna("-")

                fig_r.add_trace(go.Scattermapbox(
                    lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                    marker=dict(size=halo_size, color=[_halo_rgba_from_level(level)] * len(sub), opacity=1.0),
                    hoverinfo="skip", showlegend=False,
                ))
                fig_r.add_trace(go.Scattermapbox(
                    lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                    marker=dict(size=ring_size, color="white", opacity=0.95, symbol="circle"),
                    hoverinfo="skip", showlegend=False,
                ))
                sub["hover"] = (
                    "<b>" + sub.get("Event Name", "Event").fillna("Event") + "</b><br>"
                    + "Type: " + typ + "<br>"
                    + "Severity: " + mag.astype(str) + "<br>"
                    + "Alert: " + sub.get("Alert Level", "Unknown").fillna("Unknown") + "<br>"
                    + "Country: " + sub.get("Country", "-").fillna("-") + "<br>"
                    + "Start: " + sub["_start_dt"].map(_fmt) + "<br>"
                    + "End: "   + sub["_end_dt"].map(_fmt)
                )
                fig_r.add_trace(go.Scattermapbox(
                    lat=sub["Latitude"], lon=sub["Longitude"], mode="markers",
                    marker=dict(size=main_size, color=col, opacity=0.95, symbol="circle"),
                    name=level,
                    showlegend=(level not in shown_levels),
                    text=sub["hover"], hovertemplate="%{text}<extra></extra>",
                ))
                shown_levels.add(level)

            if recent_df.empty:
                center, zoom = (dict(lat=0, lon=0), 1.1)
            else:
                center, zoom = _center_zoom_from_points(recent_df["Latitude"], recent_df["Longitude"])

            fig_r.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=560,
                hoverlabel=dict(font_size=16),
                legend_title_text="Alert Level",
                mapbox=dict(style="carto-positron", center=center, zoom=zoom),
            )

            st.plotly_chart(
                fig_r,
                use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )

            # table
            subsection_title("List of recent alerts")
            story_context("Recent GDACS records after filters. Use to inspect attributes and click through to source.")
            st.markdown("", unsafe_allow_html=True)
            if recent_df.empty:
                st.info("No recent alerts to list for the selected filters.")
            else:
                display_recent = recent_df[[
                    "Event Name", "Country", "Disaster Type", "Alert Level",
                    "Start Date", "End Date", "Alert Score", "url"
                ]].reset_index(drop=True)
                display_recent.index += 1
                display_recent.index.name = "#"
                st.dataframe(display_recent, use_container_width=True)

    # =================================================
    # SECTION 2: Recent Alert Statistics
    # =================================================
    st.markdown("---")
    section_title("Recent Alert Statistics")
    story_context(
        "Understanding recent alert patterns helps identify emerging hotspots and hazard trends. It turns raw alerts from GDACS into insight. These visuals below show where activity is building and which events demand closer attention."
    )
    col_main, col_filter = st.columns([4, 1], gap="large")

    with col_filter:
        st.markdown('<div class="gv-filter-card">', unsafe_allow_html=True)
        subsection_title("Filters")

        alert_options = ["All"] + sorted(
            df_recent_base["Alert Level"].dropna().str.capitalize().unique().tolist()
        )
        alert_filter = st.selectbox("Alert Level", alert_options, key="recent_alert_level")
        if alert_filter == "All":
            df_alert = df_recent_base.copy()
        else:
            df_alert = df_recent_base[
                df_recent_base["Alert Level"].str.lower() == alert_filter.lower()
                ].copy()

        countries = sorted(df_alert["Country"].dropna().unique().tolist())
        country_choice = st.selectbox(
            "Country",
            options=["All countries"] + countries,
            index=0,
            key="recent_country"
        )
        if country_choice == "All countries":
            recent_df = df_alert.copy()
        else:
            recent_df = df_alert[df_alert["Country"] == country_choice].copy()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_main:
        table_df = df.copy()

        subsection_title("Alert Score Distribution")
        top_type_dist = (
            table_df.get("Disaster Type", pd.Series(dtype=str))
            .fillna("Unknown").value_counts().idxmax()
            if not table_df.empty else "-"
        )
        story_context(
            f"This chart reveals which disaster types are driving the highest alert scores in the current data. It helps spot dominant hazards and understand where risk levels are peaking "
            f", currently led by {top_type_dist}."
        )
        def _compact_country_label(s: str) -> str:
            if not isinstance(s, str) or not s.strip():
                return "-"
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
                color_discrete_sequence=PALETTE_NATURAL,
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
                hovertemplate="<b>%{y}</b><br>Type: %{legendgroup}<br>Score: %{x}<extra></extra>",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tabs[1]:
            fig_pie = px.pie(
                viz_df,
                names="Disaster Type",
                color_discrete_sequence=PALETTE_NATURAL
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        subsection_title("Active Alerts Over Time (Last 30 Days)")
        story_context(
            "Trends over the past month show how global alert activity fluctuates. Peaks highlight periods of heightened risk, while lulls suggest calmer conditions, offering context for ongoing or seasonal disaster patterns. In order words, daily counts make it easier to see if the last days were busier or calmer than usual in this data slice."
        )

        ts_df = table_df.copy()
        ts_df["Start Date"] = pd.to_datetime(ts_df.get("Start Date"), errors="coerce", utc=True)
        ts_df["End Date"]   = pd.to_datetime(ts_df.get("End Date"), errors="coerce", utc=True)
        end_window = pd.Timestamp.utcnow().normalize()
        start_window = end_window - pd.Timedelta(days=29)

        ts_df["End Date"] = ts_df["End Date"].fillna(end_window + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        mask = (ts_df["Start Date"] <= end_window) & (ts_df["End Date"] >= start_window)
        ts_df = ts_df[mask].copy()

        if ts_df.empty:
            st.markdown("No alerts intersect the last 30 days.")
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

                out = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
                    columns=["Date", "Active", group_col]
                )
                return out[(out["Date"] >= start_window.date()) & (out["Date"] <= end_window.date())]

            t1, t2 = st.tabs(["By Alert Level", "By Disaster Type"])

            with t1:
                lvl_tl = build_active_timeline(ts_df, "Alert Level")
                if lvl_tl.empty:
                    st.markdown("No Red, Orange or Green alerts to plot.")
                else:
                    fig_lvl = px.line(
                        lvl_tl, x="Date", y="Active", color="Alert Level",
                        markers=True, color_discrete_map=ALERT_COLORS,
                    )
                    fig_lvl.update_layout(
                        xaxis_title="Date", yaxis_title="Number of Active Alerts",
                        legend_title="Alert Level", hovermode="x unified"
                    )
                    st.plotly_chart(fig_lvl, use_container_width=True)

            with t2:
                type_tl = build_active_timeline(ts_df, "Disaster Type")
                if type_tl.empty:
                    st.markdown("No disasters to plot.")
                else:
                    fig_type = px.line(
                        type_tl, x="Date", y="Active", color="Disaster Type",
                        markers=True, color_discrete_sequence=PALETTE_NATURAL,
                    )
                    fig_type.update_layout(
                        xaxis_title="Date", yaxis_title="Number of Active Alerts",
                        legend_title="Disaster Type", hovermode="x unified"
                    )
                    st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("---")
        st.caption("Source: Global Disaster Alert and Coordination System")
