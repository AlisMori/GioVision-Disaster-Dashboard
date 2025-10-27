"""
Previously called: comparisons_tab.py
New name: Disaster Analysis tab
------------------
Displays comparisons between countries, disaster types, or time periods.
"""

import streamlit as st
import plotly.express as px
import pandas as pd

# ===========================
# THEME HELPERS
# ===========================
def _anchor(id_: str):
    """Invisible HTML anchor for smooth scrolling targets."""
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    """Theme-aligned section bar."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    """Theme-aligned subsection bar."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)


# ===========================
# MAIN RENDER
# ===========================
def render():
    """Render the Disaster Analysis tab (styled)."""

    # ---- Overview ----
    _anchor("sec-disaster-analysis-overview")
    section_title("Disaster Analysis")

    st.markdown(
        "This section provides comparative views of disaster patterns across countries "
        "and categories. It supports analytical insights into severity, distribution, "
        "and historical change."
    )

    # ---- Placeholder (visual demonstration) ----
    st.markdown("---")
    _anchor("sec-disaster-analysis-placeholder")
    subsection_title("Disasters by Country (Placeholder)")

    df = pd.DataFrame({
        "Country": ["USA", "India", "Japan", "Brazil"],
        "Disasters": [120, 90, 60, 45]
    })

    fig = px.bar(
        df,
        x="Country",
        y="Disasters",
        text="Disasters",
        title="Number of Recorded Disasters by Country (Placeholder)",
        color="Country",
        color_discrete_sequence=["#73AFDA", "#3A8CC5", "#2677AF", "#165F94"],
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )
    fig.update_layout(
        yaxis_title="Disaster Count",
        xaxis_title="Country",
        bargap=0.25
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: This is temporary demo data used to demonstrate visual styling.")

