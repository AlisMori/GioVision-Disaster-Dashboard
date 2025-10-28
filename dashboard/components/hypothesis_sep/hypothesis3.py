# hypothesis_sep/hypothesis3.py
import pandas as pd
import streamlit as st
import os

# ===========================
# THEME HELPERS
# ===========================
def _anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)

def section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

# ===========================
# CONFIG
# ===========================
EMDAT_PATHS = [
    "data/processed/emdat_cleaned.csv",
    "data/emdat_cleaned.csv",
    "dashboard/data/emdat_cleaned.csv",
    "../data/processed/emdat_cleaned.csv",
]

def _read_csv_first_match(paths):
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None

def render():
    _anchor("sec-h3-overview")
    section_title("Hypothesis 3")

    st.markdown(
        "> **Earthquakes** have a **higher human impact per event** (e.g., deaths or total affected per event) "
        "than **floods**."
    )

    st.markdown("---")
    

    #TODO support the hypothesis by visuals

    st.markdown("---")
    subsection_title("Data Source")
    st.caption(" ") # add here the data source

    # (No visuals by request)
