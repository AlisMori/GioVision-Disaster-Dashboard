# hypothesis_sep/hypothesis3.py
import pandas as pd
import streamlit as st
import plotly.express as px


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
    section_title("Earthquakes VS Floods")


    # Load and prepare data
    data = _read_csv_first_match(EMDAT_PATHS)
    df = data[data["Disaster Type"].isin(["Flood", "Earthquake"])].copy()
    df["Start Year"] = pd.to_numeric(df["Start Year"], errors="coerce").fillna(0).astype(int)
    df = df[df["Start Year"].between(2010, 2025)]

    for col in ["Total Deaths", "No. Injured", "Total Affected"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    color_map = {"Flood": "#5DADE2", "Earthquake": "#E74C3C"}

    # Event Frequency
    st.subheader("Event Frequency Comparison")
    st.write(
        "Floods appear far more frequently than earthquakes in the EM-DAT records from 2010 to 2025. "
        "This confirms the first part of the hypothesis — floods are more common."
    )

    freq_df = df.groupby("Disaster Type").size().reset_index(name="Event Count")
    fig_freq = px.bar(
        freq_df,
        x="Event Count",
        y="Disaster Type",
        orientation="h",
        color="Disaster Type",
        color_discrete_map=color_map,
        text_auto=True,
    )
    fig_freq.update_layout(
        title="Number of Recorded Events (2010 – 2025)",
        yaxis_title="",
        xaxis_title="Event Count",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # Average Impact per Event
    st.subheader("Average Human Impact per Event")
    st.write(
        "Average per-event impact shows that earthquakes cause dramatically higher casualties and damage per incident. "
        "Each bar represents the mean value across all events from 2010 to 2025."
    )

    impact_avg = (
        df.groupby("Disaster Type")[["Total Deaths", "No. Injured", "Total Affected"]]
        .mean()
        .reset_index()
        .melt(id_vars="Disaster Type", var_name="Impact Type", value_name="Average Impact")
    )

    fig_avg = px.bar(
        impact_avg,
        y="Impact Type",
        x="Average Impact",
        color="Disaster Type",
        barmode="group",
        orientation="h",
        color_discrete_map=color_map,
        title="Mean Human Impact per Event (2010 – 2025)",
    )
    fig_avg.update_xaxes(type="log", title="Average (per event, log scale)")
    fig_avg.update_layout(template="plotly_white", height=520)
    st.plotly_chart(fig_avg, use_container_width=True)

    # Quantitative Summary
    st.markdown("---")
    avg_flood_deaths = df[df["Disaster Type"] == "Flood"]["Total Deaths"].mean()
    avg_eq_deaths = df[df["Disaster Type"] == "Earthquake"]["Total Deaths"].mean()
    avg_flood_aff = df[df["Disaster Type"] == "Flood"]["Total Affected"].mean()
    avg_eq_aff = df[df["Disaster Type"] == "Earthquake"]["Total Affected"].mean()
    freq_ratio = (
            freq_df.loc[freq_df["Disaster Type"] == "Flood", "Event Count"].values[0]
            / freq_df.loc[freq_df["Disaster Type"] == "Earthquake", "Event Count"].values[0]
    )

    # Quantitative Summary (visual insight cards)
    st.subheader("Insight Summary")

    # Calculate ratios
    death_ratio = avg_eq_deaths / avg_flood_deaths
    aff_ratio = avg_eq_aff / avg_flood_aff

    # Display results in large-number cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Floods Occur More Often",
            value=f"{freq_ratio:.1f}×",
            delta="Flood frequency vs earthquakes",
            help="Floods are recorded significantly more frequently than earthquakes (2010–2025).",
        )

    with col2:
        st.metric(
            label="Deaths per Event (Earthquake vs Flood)",
            value=f"{death_ratio:.1f}×",
            delta="Higher fatality impact",
            help="Average earthquake causes many more deaths per event than a flood.",
        )

    with col3:
        st.metric(
            label="People Affected per Event (Earthquake vs Flood)",
            value=f"{aff_ratio:.1f}×",
            delta="Overall human impact",
            help="Average number of people affected per earthquake compared to a flood.",
        )

    st.markdown(
        """
        <div style='margin-top:20px; font-size:18px; line-height:1.6; text-align:justify;'>
            <strong>Interpretation:</strong> Between 2010 and 2025, floods occurred about 
            <strong>{:.1f}×</strong> more often than earthquakes. However, each earthquake caused approximately 
            <strong>{:.1f}×</strong> more deaths and <strong>{:.1f}×</strong> times the human impact per event.
            These findings <span style='color:#E74C3C; font-weight:bold;'>strongly support the hypothesis</span> — 
            earthquakes are less frequent but substantially more disastrous on a per-event basis.
        </div>
        """.format(freq_ratio, death_ratio, aff_ratio),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    subsection_title("Data Source")
    st.caption("EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
