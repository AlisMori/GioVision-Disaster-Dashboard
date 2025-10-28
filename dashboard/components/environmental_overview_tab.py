#app.py
import os
import sys
from pathlib import Path
import re
import unicodedata
import html
from contextlib import contextmanager

import streamlit as st

# ---------------------------------------------------------------------
# PATHS / IMPORTS
# ---------------------------------------------------------------------
ROOT = Path(_file_).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# (optional) confirm it's added
# print("Python path includes:", ROOT)


import streamlit as st
from dashboard.components import home_tab, environmental_overview_tab, impact_tab, disaster_analysis_tab, alerts_tab, hypothesis_tab
from src.utils import style_config

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="GeoVision Disaster Dashboard",
    page_icon="🌍",
    layout="wide"
)
from src.utils import style_config

# ---------------------------------------------------------------------
# PAGE CONFIG + BASE STYLE
# ---------------------------------------------------------------------
style_config.apply_streamlit_style()

css_path = os.path.join("dashboard", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("assets/style.css not found — styles may not render as designed.")

# Small inline overrides so you don't need to touch assets/style.css:
# 1) smooth scroll, 2) slightly lighter section boxes, 3) anchors land below banner+menus
st.markdown(
    """
    <style>
      html{ scroll-behavior:smooth; }
      .gv-section-title{ background:#f9fafb; border:1px solid #ececec; scroll-margin-top:96px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# NAV STRUCTURE
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# THEME PICKER (Gray by default)
# ---------------------------------------------------------------------
THEMES = {
    "Gray (default)": {"900":"#1f2937","800":"#374151","700":"#4b5563","600":"#6b7280","050":"#f3f4f6"},
    "Blue": {"900":"#0f3e6b","800":"#134d88","700":"#185aa3","600":"#1b66b9","050":"#eef5fc"},
    "Red": {"900":"#6b1321","800":"#8a1a2c","700":"#a32236","600":"#c12941","050":"#fdecef"},
    "Dark": {"900":"#e5e7eb","800":"#d1d5db","700":"#9ca3af","600":"#6b7280","050":"#111827"},
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

# Sidebar placeholder for the "Go to section" dropdown (we'll render with components.html)
_side_subnav_placeholder = st.sidebar.empty()

# ---------------------------------------------------------------------
# SECTION CAPTURE (no edits to page files)
# - Intercepts st.markdown while a page renders.
# - Any .gv-section-title gets an anchor injected and is registered.
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
    """Intercept .gv-section-title outputs to:
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
# (Page modules already render their own .gv-section-title — we leave them.)
# ---------------------------------------------------------------------
def gv_page_title(text: str):
    st.markdown(f'<div class="gv-page-title">{text}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# DROPDOWNS (horizontal under tabs, and sidebar)
# - Both use #anchor links -> no rerun, just scroll.
# - Horizontal reuses .gv-menu / .gv-m-link boxes (nav style).
# - Sidebar uses components.html to avoid HTML escaping and reuses .gv-side-link style.
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
    """Sidebar 'Go to section ▾' — renders as real HTML (no iframe), no rerun."""
    secs = st.session_state["gv_sections"].get(current_page, [])
    _side_subnav_placeholder.empty()
    if not secs:
        return

    # Build items with NO leading spaces so Markdown doesn't turn it into a code block
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
    # Example section (your real pages already output .gv-section-title themselves)
    st.markdown('<div class="gv-section-title">Overview</div>', unsafe_allow_html=True)
    st.write("*GeoVision* aggregates global disaster information for academic analysis and insight.")

# 1) Page title
gv_page_title(page)

# 2) Render the selected page while capturing section blocks it emits
with capture_sections(page):
    if page == "Home":
        page_home()
    elif page == "Alerts":
        alerts_tab.render()
    elif page == "Environmental Overview":
        environmental_overview_tab.render()
    elif page == "Impact of Natural Disasters":
        impact_tab.render()
    elif page == "Disaster Analysis":
        disaster_analysis_tab.render()
    elif page == "Hypothesis":
        hypothesis_tab.render()

# 3) After capture, print dropdown under top tabs + in sidebar
render_sections_dropdown(page)
render_sidebar_sections_dropdown(page)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown(
    '<div class="gv-separator"></div><div class="gv-footer">Working version — functionality and visuals are being expanded.</div></div>',
    unsafe_allow_html=True,
)