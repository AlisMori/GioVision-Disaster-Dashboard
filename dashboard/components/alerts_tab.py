"""
alerts_tab.py
-------------
Displays current GDACS disaster alerts focused on red-level events.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# ✅ Ensure `src` folder (which contains `data_pipeline`) is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from data_pipeline.fetch_gdacs import fetch_gdacs


def render():
    """Render the Alerts tab."""
    st.header("🚨 GDACS Disaster Alerts – Focus on Red Alerts")

    # ---- LOAD DATA ----
    df = fetch_gdacs()

    if df.empty:
        st.warning("No GDACS processed_data available right now.")
        return

    # ---- FILTER RED ALERTS ----
    red_alerts = df[df["Alert Level"].str.lower() == "red"].copy()
    red_alerts = red_alerts.sort_values("Start Date", ascending=False)
    top10_red = red_alerts.head(10)

    # ---- CONTEXT ----
    st.markdown("""
    ### Why this matters for NGOs
    Red alerts indicate the most severe and urgent disasters worldwide.  
    By focusing on the latest red alerts, NGOs can allocate resources efficiently, 
    prepare emergency response teams, and coordinate with local authorities.
    """)

    # ---- TABLE ----
    st.subheader("Top 10 Recent Red Alerts")
    st.dataframe(
        top10_red[[
            "Event Name", "Country", "Disaster Type", "Start Date",
            "End Date", "Alert Score", "url"
        ]]
    )

    # ---- BAR CHART ----
    st.subheader("Alert Score by Country (Top 10 Red Alerts)")
    fig = px.bar(
        top10_red,
        x="Alert Score",
        y="Country",
        color="Disaster Type",
        orientation="h",
        text="Alert Score",
        color_discrete_sequence=px.colors.sequential.Reds
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # ---- EMPTY CASE ----
    if top10_red.empty:
        st.info("No recent red alerts to display.")

    # ---- GREEN ALERTS SECTION ----
    st.markdown("---")
    st.header("🟢 GDACS Green Alerts Overview")

    green_alerts = df[df["Alert Level"].str.lower() == "green"].copy()
    green_alerts = green_alerts.sort_values("Start Date", ascending=False)
    top10_green = green_alerts.head(10)

    if top10_green.empty:
        st.info("No green alerts available at the moment.")
    else:
        # ---- TABLE ----
        st.subheader("Top 10 Recent Green Alerts")
        st.dataframe(
            top10_green[[
                "Event Name", "Country", "Disaster Type", "Start Date",
                "End Date", "Alert Score", "url"
            ]]
        )

        # ---- BAR CHART ----
        st.subheader("Alert Score by Country (Top 10 Green Alerts)")
        fig2 = px.bar(
            top10_green,
            x="Alert Score",
            y="Country",
            color="Disaster Type",
            orientation="h",
            text="Alert Score",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)


