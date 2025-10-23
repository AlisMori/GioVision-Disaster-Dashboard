"""
alerts_tab.py
-------------
Displays current GDACS disaster alerts focused on red-level events,
now with enhanced interactivity and meaningful filters.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from datetime import datetime

# ✅ Ensure `src` folder (which contains `data_pipeline`) is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from data_pipeline.fetch_gdacs import fetch_gdacs


def render():
    """Render the Alerts tab."""
    st.header("🚨 GDACS Disaster Alerts")

    st.markdown("""
        This page displays real-time GDACS alerts categorised by severity
    """)
    # ---- INSIGHT SECTION ----
    st.markdown("---")
    st.subheader("🌍 Insights")
    st.info("""
           - **Red alerts** indicate severe, large-scale disasters.  
           - **Orange alerts** signal potential escalation and require monitoring.  
           - **Green alerts** represent minor events or those with limited impact.  
           - Use the filters to analyse specific countries or disaster types over time.
       """)

    # ---- LOAD DATA ----
    with st.spinner("Fetching live GDACS data..."):
        df = fetch_gdacs()

    if df.empty:
        st.warning("No GDACS processed_data available right now.")
        return

    # ---- DISPLAY DATA OVERVIEW ----
    st.success(f"✅ Successfully loaded {len(df)} alerts from GDACS!")

    # Add a progress bar visualising alert count
    st.progress(min(len(df) / 100, 1.0))

    # ---- FILTER SECTION ----
    st.sidebar.header("🔎 Filter Options")

    # Dropdown for alert levels
    alert_filter = st.sidebar.selectbox(
        "Select alert level:",
        ["All", "Red", "Orange", "Green"]
    )

    # Country filter
    countries = sorted(df["Country"].dropna().unique())
    country_filter = st.sidebar.multiselect("Filter by country:", countries)

    # Date filter
    min_date = pd.to_datetime(df["Start Date"]).min()
    max_date = pd.to_datetime(df["Start Date"]).max()
    start_date, end_date = st.sidebar.date_input(
        "Filter by date range:",
        [min_date, max_date]
    )

    # Apply filters
    filtered_df = df.copy()

    if alert_filter != "All":
        filtered_df = filtered_df[filtered_df["Alert Level"].str.lower() == alert_filter.lower()]
    if country_filter:
        filtered_df = filtered_df[filtered_df["Country"].isin(country_filter)]
    filtered_df["Start Date"] = pd.to_datetime(filtered_df["Start Date"])
    start_dt = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(
        start_date).tzinfo is None else pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(
        end_date)

    filtered_df = filtered_df[
        (filtered_df["Start Date"] >= start_dt) &
        (filtered_df["End Date"] <= end_dt)
    ]

    # ---- DISASTER TYPE LEGEND ----
    disaster_legend = {
        "EQ": "Earthquake",
        "FL": "Flood",
        "TC": "Tropical Cyclone",
        "DR": "Drought",
        "VO": "Volcano",
        "WF": "Wildfire",
        "LS": "Landslide"
    }
    filtered_df["Disaster Type"] = filtered_df["Disaster Type"].replace(disaster_legend)

    # ---- DATA DISPLAY ----
    display_df = filtered_df[[
        "Event Name", "Country", "Disaster Type", "Alert Level",
        "Start Date", "End Date", "Alert Score", "url"
    ]]

    # ---- THE FIRST VISUAL - TABLE TO SHOW ALL TOP ALERTS  ----
    st.subheader("Showing Top Alerts")

    # Reset index so table numbering starts from 1
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1
    display_df.index.name = "#"

    st.dataframe(display_df)

    # ---- VISUALISATION SECTION ----
    st.markdown("---")
    st.subheader("📊 Alert Distribution Overview")

    # Toggle chart type
    chart_type = st.radio("Select chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)

    if chart_type == "Bar Chart":
        fig = px.bar(
            filtered_df,
            x="Alert Score",
            y="Country",
            color="Disaster Type",
            orientation="h",
            text="Alert Score",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    else:
        # ---- Pie Chart ----
        alert_counts = (
            filtered_df["Alert Level"]
            .value_counts()
            .reindex(["Red", "Orange", "Green"], fill_value=0)  # consistent order
            .reset_index()
        )
        alert_counts.columns = ["Alert Level", "Count"]
        fig = px.pie(
            alert_counts,
            names="Alert Level",
            values="Count",
            color="Alert Level",
            color_discrete_map={"Red": "red", "Orange": "orange", "Green": "green"},
            hole=0.3,  # donut chart
        )

        # Update trace for better hover
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0.1 if x == "Red" else 0 for x in alert_counts["Alert Level"]],
            hovertemplate="<b>%{label}</b><br>Number of Alerts: %{value}<br>Percentage: %{percent}"
        )

        fig.update_layout(
            legend_title_text="Alert Level",
            height=400
        )

    st.plotly_chart(fig, use_container_width=True)
