import math
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import reverse_geocoder as rg
import pycountry
from datetime import datetime

from src.data_pipeline.data.fetch_eonet_data import fetch_eonet_data
from src.utils.merge_datasets import merge_datasets  # kept for consistency if you use it elsewhere


# =========================
# THEME HELPERS
# =========================
def section_title(text: str):
    """Main section bar (registered by app.py capture)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)


def subsection_title(text: str):
    """Smaller subsection bar."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)


def story_context(text: str):
    """One-line context/caption above a visual."""
    st.markdown(f'<div class="gv-context">{text}</div>', unsafe_allow_html=True)


def _fmt(dt):
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return "—"


# =========================
# COLORS
# =========================
ALERT_COLORS = {
    "Red": "#EA6455",
    "Orange": "#EFB369",
    "Green": "#59B3A9",
    "Unknown": "#8A8A8A",
}


# =========================
# MAP HELPERS
# =========================
def _center_zoom_from_points(lat_series, lon_series):
    lats = pd.to_numeric(lat_series, errors="coerce").dropna()
    lons = pd.to_numeric(lon_series, errors="coerce").dropna()
    if len(lats) == 0 or len(lons) == 0:
        return dict(lat=0, lon=0), 1.3

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    center = dict(lat=(lat_min + lat_max) / 2, lon=(lon_min + lon_max) / 2)
    lat_span, lon_span = max(1e-6, lat_max - lat_min), max(1e-6, lon_max - lon_min)

    zoom = max(
        1.0,
        min(
            math.log2(360.0 / (lon_span * 1.4)),
            math.log2(180.0 / (lat_span * 1.4)),
        ),
    )
    zoom = min(8.0, zoom)

    if lon_span < 0.01 and lat_span < 0.01:
        zoom = 5.0

    return center, zoom


# =========================
# DATA LOADER
# =========================
@st.cache_data(ttl=3600, show_spinner="Fetching fresh data from NASA EONET...")
def load_eonet_data(days=365):
    return fetch_eonet_data(days=days, limit=10000)


# =========================
# COUNTRY DERIVATION
# =========================
def derive_country(df):
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = list(zip(df["latitude"], df["longitude"]))
        results = rg.search(coords, mode=1)
        df["country_code"] = [res.get("cc", "Unknown") for res in results]
        df["country"] = [
            pycountry.countries.get(alpha_2=cc).name
            if pycountry.countries.get(alpha_2=cc)
            else "Unknown"
            for cc in df["country_code"]
        ]
    else:
        df["country"] = "Unknown"
    return df


# =========================
# MAIN RENDER
# =========================
def render():
    # sticky column CSS
    st.markdown(
        """
        <style>
        div[data-testid="column"]:has(div.sticky-filters-right) {
            position: -webkit-sticky;
            position: sticky;
            top: 60px;
            align-self: flex-start;
            z-index: 999;
            height: fit-content;
            overflow-y: visible;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # layout: spacer / content / filter
    content_col, filter_col = st.columns([6, 1.5])


    # ----------------------------------
    # FILTER COLUMN (RIGHT)
    # ----------------------------------
    with filter_col:
        st.markdown('<div class="sticky-filters-right">', unsafe_allow_html=True)
        st.markdown('<div class="gv-filter-card">', unsafe_allow_html=True)
        subsection_title("Filters")

        # 1) days slider moved here
        days = st.slider(
            "Number of past days to fetch",
            min_value=30,
            max_value=365,
            value=365,
            key="filter_days",
            help="Adjust the time window for NASA EONET events.",
        )

        # load data AFTER knowing days
        with st.spinner("Fetching EONET data..."):
            df = load_eonet_data(days=days)

        if df.empty:
            st.error("Unable to fetch data from NASA EONET. Please try again later.")
            st.markdown("</div></div>", unsafe_allow_html=True)
            st.stop()

        # make sure we have countries so we can filter on them
        df = derive_country(df)

        # build options
        disaster_types_filter = sorted(
            df["disaster_type_standardized"].dropna().unique().tolist()
        )
        alert_levels_filter = sorted(df["alert_level"].dropna().unique().tolist())
        countries_filter = sorted(df["country"].dropna().unique().tolist())

        selected_disaster_type = st.selectbox(
            "Disaster Type", ["All Types"] + disaster_types_filter, key="filter_disaster"
        )
        selected_alert = st.selectbox(
            "Alert Level", ["All Alerts"] + alert_levels_filter, key="filter_alert"
        )
        selected_country = st.selectbox(
            "Country", ["All Countries"] + countries_filter, key="filter_country"
        )

        st.markdown("</div>", unsafe_allow_html=True)  # close gv-filter-card
        st.markdown("</div>", unsafe_allow_html=True)  # close sticky

    # ----------------------------------
    # APPLY FILTERS
    # ----------------------------------
    df_filtered = df.copy()
    if selected_disaster_type != "All Types":
        df_filtered = df_filtered[
            df_filtered["disaster_type_standardized"] == selected_disaster_type
            ]
    if selected_alert != "All Alerts":
        df_filtered = df_filtered[df_filtered["alert_level"] == selected_alert]
    if selected_country != "All Countries":
        df_filtered = df_filtered[df_filtered["country"] == selected_country]

    # ----------------------------------
    # CONTENT COLUMN (CENTER)
    # ----------------------------------
    with content_col:
        # Ensure left alignment for all content
        st.markdown(
            """
            <style>
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                text-align: left !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Header
        section_title("Overview")
        st.markdown(
            f"""
            This dashboard provides a global overview of natural disasters using NASA EONET data.
            **Showing events from the last {days} days.**
            """
        )

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Events", f"{len(df_filtered):,}")
        col2.metric("Disaster Types", df_filtered["disaster_type_standardized"].nunique())
        col3.metric("Countries", df_filtered["country"].nunique())
        col4.metric("Latest Event", _fmt(df_filtered["event_date"].max()))
        st.markdown("---")

        # 1. Global Disaster Map
        subsection_title("Global Disaster Map")
        st.markdown(
            "This interactive map displays the geographic distribution of all disaster events. "
            "Each marker represents a disaster, color-coded by alert level, and shows detailed information on hover."
        )
        df_map = df_filtered.dropna(subset=["latitude", "longitude"])
        if not df_map.empty:
            fig_map = go.Figure()
            for disaster_type in df_map["disaster_type_standardized"].unique():
                sub = df_map[df_map["disaster_type_standardized"] == disaster_type]
                alert_level = (
                    sub["alert_level"].mode().iloc[0] if len(sub) > 0 else "Unknown"
                )
                color_hex = ALERT_COLORS.get(alert_level, ALERT_COLORS["Unknown"])
                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=sub["latitude"],
                        lon=sub["longitude"],
                        mode="markers",
                        marker=dict(size=11, color=color_hex),
                        name=disaster_type,
                        text=sub.apply(
                            lambda r: f"<b>{r['event_name']}</b><br>Date: {_fmt(r['event_date'])}"
                                      f"<br>Alert Level: {r['alert_level']}<br>Country: {r['country']}",
                            axis=1,
                        ),
                        hoverinfo="text",
                    )
                )
            center, zoom = _center_zoom_from_points(
                df_map["latitude"], df_map["longitude"]
            )
            fig_map.update_layout(
                mapbox=dict(style="carto-positron", center=center, zoom=zoom),
                height=600,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geographic data available for the selected filters.")

        # 2. Disasters Over Time
        st.markdown("---")
        subsection_title("Disasters Over Time")
        st.markdown(
            "This line chart tracks the monthly trend of disaster events, helping identify patterns and seasonal variations."
        )
        df_filtered["year_month"] = pd.to_datetime(df_filtered["event_date"]).dt.to_period(
            "M"
        )
        df_monthly = df_filtered.groupby("year_month").size().reset_index(name="count")
        df_monthly["year_month"] = df_monthly["year_month"].dt.to_timestamp()
        fig_line = px.line(
            df_monthly,
            x="year_month",
            y="count",
            markers=True,
            labels={"year_month": "Month", "count": "Number of Events"},
            title="Disasters Over Time",
        )
        fig_line.update_traces(
            line_color="#59B3A9",
            marker=dict(size=8, color="#EA6455"),
            hovertemplate="%{y} events<extra></extra>",
        )
        fig_line.update_xaxes(dtick="M1", tickformat="%b\n%Y")
        fig_line.update_layout(height=450)
        st.plotly_chart(fig_line, use_container_width=True)

        # 3. Disaster Type Distribution  (FIXED)
        st.markdown("---")
        subsection_title("Disaster Type Distribution")
        st.markdown(
            "This bar chart shows which disaster categories appear most often in the selected time period."
        )
        type_counts = (
            df_filtered["disaster_type_standardized"]
            .value_counts()
            .reset_index()
        )
        # make names predictable
        type_counts.columns = ["Disaster Type", "Count"]

        if type_counts.empty:
            st.info("No disaster type data available for the selected filters.")
        else:
            fig_bar = px.bar(
                type_counts.head(10),
                x="Disaster Type",
                y="Count",
                text="Count",
                color="Disaster Type",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 4. Alert + Monthly distribution
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            subsection_title("Alert Level Distribution")
            st.markdown(
                "This pie chart shows the proportion of events by alert level, helping assess severity mix."
            )
            alert_counts = (
                df_filtered["alert_level"]
                .value_counts()
                .reset_index()
            )
            alert_counts.columns = ["Alert Level", "Count"]
            if alert_counts.empty:
                st.info("No alert-level data for current filters.")
            else:
                fig_pie = px.pie(
                    alert_counts,
                    names="Alert Level",
                    values="Count",
                    color="Alert Level",
                    color_discrete_map=ALERT_COLORS,
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            subsection_title("Monthly Distribution")
            st.markdown(
                "This chart displays the distribution of events across months of the current year."
            )
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy["event_date_dt"] = pd.to_datetime(
                df_filtered_copy["event_date"]
            )
            current_year = datetime.now().year
            current_month = datetime.now().month
            df_filtered_copy = df_filtered_copy[
                df_filtered_copy["event_date_dt"].dt.year == current_year
                ]
            df_filtered_copy = df_filtered_copy[
                df_filtered_copy["event_date_dt"].dt.month <= current_month
                ]
            df_filtered_copy["month_name"] = df_filtered_copy["event_date_dt"].dt.month_name()
            valid_months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ][:current_month]
            month_counts = (
                df_filtered_copy["month_name"]
                .value_counts()
                .reindex(valid_months, fill_value=0)
            )
            fig_month = px.bar(
                x=month_counts.index,
                y=month_counts.values,
            )
            fig_month.update_traces(hovertemplate="%{y} events<extra></extra>")
            fig_month.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Events",
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig_month, use_container_width=True)

        # 5. Bubble Chart
        st.markdown("---")
        subsection_title("Disaster Frequency & Severity Analysis by Country")
        st.markdown(
            """
            This bubble chart illustrates both the frequency and calculated severity of disasters by country
            for a specific disaster type.
            """
        )
        if not df_filtered.empty:
            bubble_disaster_type = st.selectbox(
                "Select Disaster Type for Bubble Chart",
                sorted(df_filtered["disaster_type_standardized"].dropna().unique()),
                key="bubble_chart_selector",
            )
            df_bubble = df_filtered[
                df_filtered["disaster_type_standardized"] == bubble_disaster_type
                ].copy()
        else:
            df_bubble = pd.DataFrame()

        if not df_bubble.empty:
            severity_metrics = {
                "Earthquake": {"unit": "Deaths", "multiplier": 10},
                "Wildfire": {"unit": "Hectares Burnt", "multiplier": 500},
                "Flood": {"unit": "People Affected", "multiplier": 1000},
                "Drought": {"unit": "People Affected", "multiplier": 5000},
                "Severe Storm": {"unit": "Deaths", "multiplier": 5},
                "Tropical Storm": {"unit": "Deaths", "multiplier": 8},
                "Volcano": {"unit": "Deaths", "multiplier": 15},
                "Landslide": {"unit": "Deaths", "multiplier": 7},
                "Snow": {"unit": "People Affected", "multiplier": 800},
                "Temperature Extreme": {"unit": "Deaths", "multiplier": 12},
                "Dust and Haze": {"unit": "People Affected", "multiplier": 2000},
                "Water Color": {"unit": "Area Affected (km²)", "multiplier": 100},
                "Sea and Lake Ice": {"unit": "Area Affected (km²)", "multiplier": 150},
                "Manmade": {"unit": "People Affected", "multiplier": 500},
            }
            metric_info = severity_metrics.get(
                bubble_disaster_type, {"unit": "Events", "multiplier": 1}
            )
            bubble_data = (
                df_bubble.groupby("country")["event_name"]
                .count()
                .reset_index(name="event_count")
            )
            bubble_data["severity_value"] = (
                    bubble_data["event_count"] * metric_info["multiplier"]
            )
            bubble_data["severity_display"] = bubble_data["severity_value"].apply(
                lambda x: f"Severity: {int(x):,} {metric_info['unit']}"
            )
            fig_bubble = go.Figure()
            fig_bubble.add_trace(
                go.Scatter(
                    x=bubble_data["country"],
                    y=bubble_data["event_count"],
                    mode="markers",
                    marker=dict(
                        size=bubble_data["severity_value"],
                        sizemode="area",
                        sizeref=2.0 * max(bubble_data["severity_value"]) / (40.0 ** 2),
                        color="#59B3A9",
                        line=dict(width=1, color="white"),
                    ),
                    text=bubble_data["severity_display"],
                    hovertemplate="<b>%{x}</b><br>Events: %{y}<br>%{text}<extra></extra>",
                )
            )
            fig_bubble.update_layout(
                title=f"{bubble_disaster_type} - Frequency & Severity by Country",
                xaxis_title="Country",
                yaxis_title="Number of Events",
                xaxis_tickangle=-45,
                height=500,
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("No data available for this disaster type.")

        # 6. Recent Events Table
        st.markdown("---")
        subsection_title("Recent Disaster Events")
        st.markdown(
            "This table lists the most recent disaster events with key details including type, date, alert level, and location."
        )
        df_recent = (
            df_filtered.sort_values(by="event_date", ascending=False)
            .head(10)
            .copy()
            .reset_index(drop=True)
        )
        df_recent["#"] = range(1, len(df_recent) + 1)
        df_recent["event_name"] = (
            df_recent["event_name"]
            .str.replace(r"\s*\d+$", "", regex=True)
            .str.strip()
        )
        st.dataframe(
            df_recent[
                [
                    "#",
                    "event_name",
                    "disaster_type_standardized",
                    "event_date",
                    "alert_level",
                    "country",
                ]
            ],
            use_container_width=True,
        )

        # footer
        st.markdown("---")
        st.caption(
            f"Data Source: NASA EONET API | Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        st.download_button(
            "Download Filtered Data (CSV)",
            df_filtered.to_csv(index=False),
            "eonet_disasters.csv",
            "text/csv",
        )