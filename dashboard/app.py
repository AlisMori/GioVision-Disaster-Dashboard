# app.py
import os
import sys
from pathlib import Path
import streamlit as st

# --- sys.path so imports work no matter how you run the app ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dashboard.components import (
    environmental_overview_tab,
    impact_tab,
    disaster_analysis_tab,
    alerts_tab,
    hypothesis_tab,
)
from src.utils import style_config

# ----------------------------
# PAGE CONFIG + BASE STYLE
# ----------------------------
st.set_page_config(page_title="GeoVision Disaster Dashboard", page_icon=None, layout="wide")
style_config.apply_streamlit_style()

css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("assets/style.css not found — styles may not render as designed.")

# ----------------------------
# NAV STRUCTURE
# ----------------------------
PAGES = {
    "Home": [],
    "Alerts": [],
    "Environmental Overview": [],
    "Impact of Natural Disasters": [],
    "Disaster Analysis": [],
    "Hypothesis": [],
}
ORDER = list(PAGES.keys())
DEFAULT_PAGE = "Alerts"

# ----------------------------
# THEME PICKER (Gray by default)
# ----------------------------
THEMES = {
    "Gray (default)": {"900":"#1f2937","800":"#374151","700":"#4b5563","600":"#6b7280","050":"#f3f4f6"},
    "Blue":           {"900":"#0f3e6b","800":"#134d88","700":"#185aa3","600":"#1b66b9","050":"#eef5fc"},
    "Red":            {"900":"#6b1321","800":"#8a1a2c","700":"#a32236","600":"#c12941","050":"#fdecef"},
    "Dark":           {"900":"#e5e7eb","800":"#d1d5db","700":"#9ca3af","600":"#6b7280","050":"#111827"},
}
st.sidebar.header("Navigation")
theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0)
t = THEMES[theme_name]
st.markdown(
    f"""
    <style>
    :root {{
      --brand-900:{t['900']};
      --brand-800:{t['800']};
      --brand-700:{t['700']};
      --brand-600:{t['600']};
      --brand-050:{t['050']};
    }}
    .gv {{
      --brand-900:{t['900']};
      --brand-800:{t['800']};
      --brand-700:{t['700']};
      --brand-600:{t['600']};
      --brand-050:{t['050']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# QUERY PARAMS (page only)
# ----------------------------
qp = st.query_params
page = qp.get("page", DEFAULT_PAGE)
if page not in ORDER:
    page = DEFAULT_PAGE
st.query_params["page"] = page

# ----------------------------
# BANNER
# ----------------------------
st.markdown(
    """
<div class="gv">
  <div class="gv-banner">
    <div class="gv-banner__inner">
      <div class="gv-banner__title">Global Natural Disasters Dashboard</div>
      <div class="gv-banner__subtitle">ICT305 · Data Visualisation and Simulation · Murdoch University · 2025</div>
    </div>
  </div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# HORIZONTAL MENU (content-width, grey, black text)
# ----------------------------
def top_menu_html(active_page: str) -> str:
    items = []
    for p in ORDER:
        cls = "gv-m-item gv-m-item--active" if p == active_page else "gv-m-item"
        items.append(
            f'<div class="{cls}"><a class="gv-m-link" href="?page={p}" target="_self" rel="noopener">{p}</a></div>'
        )
    return '<nav class="gv-menu" aria-label="Primary Navigation">' + "".join(items) + "</nav>"

st.markdown(top_menu_html(page), unsafe_allow_html=True)

# ----------------------------
# VERTICAL MENU (grey, black text)
# ----------------------------
def side_menu_html(active_page: str) -> str:
    blocks = ['<div class="gv-side">']
    for p in ORDER:
        wrap_cls = "gv-side-item gv-side-item--active" if p == active_page else "gv-side-item"
        blocks.append(
            f'<div class="{wrap_cls}"><a class="gv-side-link" href="?page={p}" target="_self" rel="noopener">{p}</a></div>'
        )
    blocks.append("</div>")
    return "".join(blocks)

st.sidebar.markdown(side_menu_html(page), unsafe_allow_html=True)

# ----------------------------
# TITLE HELPERS (page/section bars)
# ----------------------------
def gv_page_title(text: str):
    st.markdown(f'<div class="gv-page-title">{text}</div>', unsafe_allow_html=True)

def gv_section_title(text: str):
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

# ----------------------------
# ROUTING
# ----------------------------
def page_home():
    gv_page_title("Home")
    gv_section_title("Overview")
    st.write("**GeoVision** aggregates global disaster information for academic analysis and insight.")

if page == "Home":
    page_home()
elif page == "Alerts":
    gv_page_title("GDACS Alerts")   # <- requested page name
    alerts_tab.render()
elif page == "Environmental Overview":
    gv_page_title("Environmental Overview")
    environmental_overview_tab.render()
elif page == "Impact of Natural Disasters":
    gv_page_title("Impact of Natural Disasters")
    impact_tab.render()
elif page == "Disaster Analysis":
    gv_page_title("Disaster Analysis")
    disaster_analysis_tab.render()
elif page == "Hypothesis":
    gv_page_title("Hypothesis")
    hypothesis_tab.render()

# ----------------------------
# FOOTER + close wrapper
# ----------------------------
st.markdown(
    '<div class="gv-separator"></div><div class="gv-footer">Working version — functionality and visuals are being expanded.</div></div>',
    unsafe_allow_html=True,
)
