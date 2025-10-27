"""
home_page.py
-------------
Minimal landing page showing project purpose.
"""

import streamlit as st

# ===========================
# THEME HELPERS
# ===========================
def _anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

#TODO write the homepage descriptions

# ===========================
# MAIN RENDER
# ===========================
def render():
    _anchor("sec-home-overview")
    section_title("Overview")
    st.markdown(
        "This dashboard summarizes global natural disasters using live alerts "
        "and historical impact data."
    )

    st.markdown("---")
    subsection_title("Purpose")
    st.markdown(
        "It provides high-level insights and supports comparison across affected countries and disaster types."
    )
