# dashboard/components/hypothesis.py
import os
import importlib.util
import streamlit as st

def _load_module_from_path(module_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def render():
    
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
        font-size: 1.25rem !important;   /* BIGGER text */
        padding: 1rem 1.4rem !important; /* thicker clickable area */
        font-weight: 700 !important;     /* bold */
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

    st.header("🧠 Project Hypothesis")

    st.markdown("""
    ### Overview  
    This section outlines the key hypotheses and assumptions guiding the data analysis 
    and dashboard design. It provides context for interpreting the GDACS/EMDAT data and expected outcomes.
    """)

    st.markdown("""
    **Hypotheses:**
    1. A higher number of people from less developed or lower-income countries are affected by natural disasters compared to developed nations.  
    2. Since the 2000s, there has been an increase in the frequency of severe weather-related events such as floods, cyclones, and heatwaves.  
    3. Earthquakes, although less frequent than floods, have a more severe impact on humans compared to floods.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        "Hypothesis 1": os.path.join(base_dir, "hypothesis_sep", "hypothesis1.py"),
        "Hypothesis 2": os.path.join(base_dir, "hypothesis_sep", "hypothesis2.py"),
        "Hypothesis 3": os.path.join(base_dir, "hypothesis_sep", "hypothesis3.py"),
    }

    tabs = st.tabs(list(paths.keys()))
    for tab, (label, path) in zip(tabs, paths.items()):
        with tab:
            try:
                mod = _load_module_from_path(label.replace(" ", "").lower(), path)
                if hasattr(mod, "render"):
                    mod.render()
                else:
                    st.warning(f"⚠️ `{os.path.basename(path)}` loaded but no `render()` found.")
            except FileNotFoundError:
                st.info(f"ℹ️ `{path}` not found yet. Add it to render {label}.")
            except Exception as e:
                st.error(f"❌ Failed to load {label}: {e}")