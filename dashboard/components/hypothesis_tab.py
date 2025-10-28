# dashboard/components/hypothesis.py
import os
import importlib.util
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

def _load_module_from_path(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def render():
    # --- Tabs styling (kept from your version) ---
    st.markdown(
        """
        <style>
        /* Increase spacing between tabs */
        div.stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            margin-top: 10px;
        }
        /* Enlarge font, add more padding */
        div.stTabs [data-baseweb="tab"] {
            font-size: 1.25rem !important;
            padding: 1rem 1.4rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.3px;
        }
        /* Stronger active underline */
        div.stTabs [data-baseweb="tab"]::after {
            height: 4px !important;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Overview ---
    _anchor("sec-hyp-overview")
    section_title("Overview")
    st.markdown(
        "This page outlines the hypotheses guiding our disaster-impact analysis and "
        "links directly to dedicated sections where each hypothesis is operationalized."
    )
    st.markdown(
        "- Hypothesis 1: People in **Least Developed Countries (LDCs)** are more affected than in developed nations.\n"
        "- Hypothesis 2: Since the 2000s, **severe weather-related events** (floods, cyclones, heatwaves) have increased in frequency.\n"
        "- Hypothesis 3: **Earthquakes** have **higher human impact per event** than floods."
    )

    st.markdown("---")

    # --- Tab loader setup ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        "Hypothesis 1": os.path.join(base_dir, "hypothesis_sep", "hypothesis1.py"),
        "Hypothesis 2": os.path.join(base_dir, "hypothesis_sep", "hypothesis2.py"),
        "Hypothesis 3": os.path.join(base_dir, "hypothesis_sep", "hypothesis3.py"),
    }

    tabs = st.tabs(list(paths.keys()))
    for tab, (label, path) in zip(tabs, paths.items()):
        with tab:
            _anchor(f"sec-{label.lower().replace(' ', '-')}")
            try:
                mod = _load_module_from_path(label.replace(" ", "").lower(), path)
                if hasattr(mod, "render"):
                    mod.render()
                else:
                    st.warning(f"`{os.path.basename(path)}` loaded but no `render()` found.")
            except FileNotFoundError:
                st.info(f"`{path}` not found. Add it to render {label}.")
            except Exception as e:
                st.error(f"Failed to load {label}: {e}")