# alerts_tab.py
def render(data_pipeline=None):
    """Render the Alerts tab."""
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from data_pipeline.fetch_gdacs import fetch_gdacs

    st.title("GDACS Disaster Alerts – Focus on Red Alerts")

    df = fetch_gdacs()

    if df.empty:
        st.warning("No GDACS data available right now.")
        return

    red_alerts = df[df["Alert Level"].str.lower() == "red"].copy()
    red_alerts = red_alerts.sort_values("Start Date", ascending=False)
    top10_red = red_alerts.head(10)

    st.markdown("""
    ### Why this matters for NGOs
    Red alerts indicate the most severe and urgent disasters worldwide.  
    By focusing on the latest red alerts, NGOs can allocate resources efficiently, 
    prepare emergency response teams, and coordinate with local authorities.
    """)

    st.subheader("Top 10 Recent Red Alerts")
    st.dataframe(top10_red[["Event Name", "Country", "Disaster Type", "Start Date", "End Date", "Alert Score", "url"]])

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
    
    if top10_red.empty:
        st.info("No recent red alerts to display.")
