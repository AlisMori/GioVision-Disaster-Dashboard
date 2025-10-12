"""
style_config.py
---------------
Centralizes dashboard color palette, fonts, and theme configuration.
"""

COLOR_SCHEME = {
    "background": "#F8F9FA",
    "primary": "#0056B3",
    "accent": "#FFA500",
    "text": "#212529",
    "info": "#17A2B8",
}

FONTS = {
    "header": "Helvetica, Arial, sans-serif",
    "body": "Open Sans, sans-serif",
}


def apply_streamlit_style():
    """
    Injects custom CSS into Streamlit app to apply global style theme.
    """
    import streamlit as st

    custom_css = f"""
        <style>
            body {{
                background-color: {COLOR_SCHEME['background']};
                color: {COLOR_SCHEME['text']};
                font-family: {FONTS['body']};
            }}

            h1, h2, h3 {{
                color: {COLOR_SCHEME['primary']};
                font-family: {FONTS['header']};
            }}

            .stButton > button {{
                background-color: {COLOR_SCHEME['primary']};
                color: white;
                border-radius: 6px;
                padding: 0.5rem 1rem;
                border: none;
                transition: background-color 0.2s ease;
            }}

            .stButton > button:hover {{
                background-color: {COLOR_SCHEME['accent']};
            }}

            .css-18e3th9 {{
                padding-top: 1rem;
            }}
        </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)
    st.session_state["theme_loaded"] = True
