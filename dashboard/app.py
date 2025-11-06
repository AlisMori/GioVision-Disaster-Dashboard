# app.py
import os
import sys
from pathlib import Path
import re
import unicodedata
import html
from contextlib import contextmanager
from PIL import Image

import streamlit as st

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dashboard.components import (
    home_tab,
    environmental_overview_tab,
    impact_tab,
    disaster_analysis_tab,
    alerts_tab,
    trends_tab,
)
from src.utils import style_config

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="GeoVision Disaster Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------------------
# PAGE CONFIG + BASE STYLE
# ---------------------------------------------------------------------
style_config.apply_streamlit_style()

# Find absolute path of the folder where this file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust to reach the shared assets folder
css_path = os.path.join(BASE_DIR, "assets", "style.css")

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found at {css_path}")

# Small inline overrides — now only smooth scroll + anchor spacing
st.markdown(
    """
    <style>
      html{ scroll-behavior:smooth; }
      .gv-section-title{ scroll-margin-top:96px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# NAV STRUCTURE
# ---------------------------------------------------------------------

# Open and resize the logo
logo = Image.open("dashboard/assets/company_logo.png")
logo = logo.resize((250, 40))  # width=150px, height=50px (adjust as needed)

# Display in sidebar
st.sidebar.image(logo)

PAGES = {
    "Home": [],
    "Alerts": [],
    "Environmental Overview": [],
    "Impact of Natural Disasters": [],
    "Disaster Analysis": [],
    "Trends": [],
}
ORDER = list(PAGES.keys())
DEFAULT_PAGE = "Home"   # 👈 now Home is the default

# ---------------------------------------------------------------------
# SIDEBAR NAV (no theme picker anymore)
# ---------------------------------------------------------------------
st.sidebar.header("Navigation")

# ---------------------------------------------------------------------
# QUERY PARAMS (page only)
# ---------------------------------------------------------------------
qp = st.query_params
page = qp.get("page", DEFAULT_PAGE)
if page not in ORDER:
    page = DEFAULT_PAGE
st.query_params["page"] = page  # normalize

# ---------------------------------------------------------------------
# BANNER
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# TOP HORIZONTAL MENU (boxed tabs)
# ---------------------------------------------------------------------
def top_menu_html(active_page: str) -> str:
    items = []
    for p in ORDER:
        cls = "gv-m-item gv-m-item--active" if p == active_page else "gv-m-item"
        items.append(
            f'<div class="{cls}"><a class="gv-m-link" href="?page={p}" target="_self" rel="noopener">{p}</a></div>'
        )
    return '<nav class="gv-menu" aria-label="Primary Navigation">' + "".join(items) + "</nav>"

st.markdown(top_menu_html(page), unsafe_allow_html=True)

# Placeholder for the dropdown bar that sits DIRECTLY under the page tabs
_subnav_placeholder = st.empty()

# ---------------------------------------------------------------------
# VERTICAL MENU (sidebar)
# ---------------------------------------------------------------------
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

# Sidebar placeholder for the "Go to section" dropdown
_side_subnav_placeholder = st.sidebar.empty()

# ---------------------------------------------------------------------
# SECTION CAPTURE (no edits to page files)
# ---------------------------------------------------------------------
def _slugify(text: str) -> str:
    txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"[^a-zA-Z0-9]+", "-", txt).strip("-").lower()
    return txt or "section"

if "gv_sections" not in st.session_state:
    st.session_state["gv_sections"] = {}   # {page: [(label, id), ...]}

def _reset_page_sections(current_page: str):
    st.session_state["gv_sections"][current_page] = []

def _register_section(current_page: str, label: str) -> str:
    base = f"sec-{_slugify(label)}"
    existing = {sid for _, sid in st.session_state["gv_sections"].get(current_page, [])}
    anchor = base
    n = 2
    while anchor in existing:
        anchor = f"{base}-{n}"
        n += 1
    st.session_state["gv_sections"].setdefault(current_page, []).append((label, anchor))
    return anchor

@contextmanager
def capture_sections(current_page: str):
    """
    Intercept .gv-section-title outputs to:
     1) prepend an anchor <div id='sec-...'></div>
     2) register the section for both dropdowns
    """
    _reset_page_sections(current_page)
    original_markdown = st.markdown

    def patched_markdown(body, *args, **kwargs):
        try:
            if isinstance(body, str) and 'class="gv-section-title"' in body:
                m = re.search(r'gv-section-title">(.*?)</div>', body, flags=re.DOTALL | re.IGNORECASE)
                if m:
                    raw = html.unescape(m.group(1))
                    label = re.sub(r"<.*?>", "", raw).strip()
                    if label:
                        anchor_id = _register_section(current_page, label)
                        body = f"<div id='{anchor_id}'></div>" + body
        except Exception:
            pass
        return original_markdown(body, *args, **kwargs)

    st.markdown = patched_markdown
    try:
        yield
    finally:
        st.markdown = original_markdown

# ---------------------------------------------------------------------
# PAGE / SECTION TITLE HELPERS
# ---------------------------------------------------------------------
def gv_page_title(text: str):
    st.markdown(f'<div class="gv-page-title">{text}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# DROPDOWNS (horizontal under tabs, and sidebar)
# ---------------------------------------------------------------------
def render_sections_dropdown(current_page: str):
    secs = st.session_state["gv_sections"].get(current_page, [])
    if not secs:
        _subnav_placeholder.empty()
    else:
        items = []
        for label, sid in secs:
            items.append(f'<div class="gv-m-item"><a class="gv-m-link" href="#{sid}">{label}</a></div>')

        html_dropdown = f"""
        <nav class="gv-menu" aria-label="Section Navigation" style="margin-top:-8px;">
          <div class="gv-m-item" style="display:block;">
            <details class="gv-sections-details">
              <summary class="gv-m-link" style="list-style:none; cursor:pointer;">
                Go to section ▾
              </summary>
              <div class="gv-sections-panel">
                {''.join(items)}
              </div>
            </details>
          </div>
        </nav>
        """
        _subnav_placeholder.markdown(html_dropdown, unsafe_allow_html=True)

def render_sidebar_sections_dropdown(current_page: str):
    """Sidebar 'Go to section ▾' — renders as real HTML, no rerun."""
    secs = st.session_state["gv_sections"].get(current_page, [])
    _side_subnav_placeholder.empty()
    if not secs:
        return

    items = "".join(
        f'<div class="gv-side-item" style="margin:6px 10px 0 18px;"><a class="gv-side-link" href="#{sid}">{label}</a></div>'
        for label, sid in secs
    )

    sidebar_html = (
        '<div class="gv-side" style="margin-top:6px;">'
        '<details>'
        '<summary class="gv-side-link" style="list-style:none; cursor:pointer; display:block;">Go to section ▾</summary>'
        f'{items}'
        '</details>'
        '</div>'
    )

    _side_subnav_placeholder.markdown(sidebar_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# ROUTING
# ---------------------------------------------------------------------
def page_home():
    st.markdown('<div class="gv-section-title">Overview</div>', unsafe_allow_html=True)
    st.write("**GeoVision** aggregates global disaster information for academic analysis and insight.")

# 1) Page title
gv_page_title(page)

# 2) Render page with section capture
with capture_sections(page):
    if page == "Home":
        home_tab.render()
    elif page == "Alerts":
        alerts_tab.render()
    elif page == "Environmental Overview":
        environmental_overview_tab.render()
    elif page == "Impact of Natural Disasters":
        impact_tab.render()
    elif page == "Disaster Analysis":
        disaster_analysis_tab.render()
    elif page == "Trends":
        trends_tab.render()

# 3) After capture, print dropdown under top tabs + in sidebar
# 👉 hide "Go to section" on Trends page
if page != "Trends":
    render_sections_dropdown(page)
    render_sidebar_sections_dropdown(page)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown(
    '<div class="gv-separator"></div><div class="gv-footer">Working version — functionality and visuals are being expanded.</div></div>',
    unsafe_allow_html=True,
)
