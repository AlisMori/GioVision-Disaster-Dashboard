"""
hypothesis_tab.py
-----------------
Displays the Hypothesis page with analysis insights or project assumptions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

def render():
    """Render the Hypothesis tab."""
    st.header("🧠 Project Hypothesis")

    st.markdown("""
    ### Overview  
    This section outlines the key hypotheses and assumptions guiding the data analysis 
    and dashboard design. It provides context for interpreting the GDACS data and expected outcomes.
    """)

    st.markdown("""
    **Example Hypotheses:**
    1. A higher number of people from less developed or lower-income countries are affected by natural disasters compared to developed nations. 
    2. Since the 2000s, there has been an increase in the frequency of severe weather-related events such as floods, cyclones, and heatwaves. 
    3. Earthquakes, although less frequent than floods, have a more severe impact on humans compared to floods. 
    """)

    st.info("These hypotheses can be tested later using the three datasets: ENONET, EMDAT, and GDACS.")

    # Optional small placeholder chart
    sample_data = pd.DataFrame({
        "Alert Level": ["Green", "Orange", "Red"],
        "Average Duration (Days)": [3, 7, 14]
    })

    fig = px.bar(
        sample_data,
        x="Alert Level",
        y="Average Duration (Days)",
        color="Alert Level",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Sample Hypothesis Visual: Alert Duration by Severity Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Interpretation  
    If actual GDACS data follows this pattern, it would support the hypothesis that higher-severity
    alerts are linked with longer-lasting disaster events.
    """)

