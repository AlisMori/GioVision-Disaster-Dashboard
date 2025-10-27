# dashboard/components/home_tab.py
import streamlit as st

def render():
    st.title("🌍 GeoVision Disaster Dashboard")
    st.subheader("Summary Overview")

    st.markdown("""
    Welcome to the **GeoVision Disaster Dashboard (GDD)** — an interactive platform
    that visualizes natural disaster data from **NASA EONET**, **GDACS**, and **EM-DAT**.
    
    The dashboard is designed for NGOs, policymakers, and citizens to:
    - Monitor real-time alerts and historical patterns.
    - Understand human and economic impacts.
    - Strengthen disaster preparedness and response strategies.
    """)

    # --- KPI Section ---
    st.markdown("### Key Global Indicators (2025)")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌪️ Total Disasters", "124")
    col2.metric("👥 People Affected", "2.3M")
    col3.metric("💀 Fatalities", "12,300")
    col4.metric("💸 Economic Loss", "$1.2B")

    st.markdown("---")
    st.subheader("Quick Navigation")
    st.markdown("""
    🔹 **Environmental Overview** – Explore global distribution and trends.  
    🔹 **Impact of Natural Disasters** – Examine human and economic costs.  
    🔹 **Disaster Analysis** – Study detailed comparisons by country and type.  
    🔹 **Alerts** – View live GDACS updates.  
    🔹 **Hypothesis Testing** – Validate disaster impact trends and insights.
    """)

    st.markdown("---")
    st.caption("📊 Data Sources: NASA EONET | GDACS | EM-DAT")
    st.caption("Developed by Team GeoVision – ICT305, Murdoch University (2025)")
