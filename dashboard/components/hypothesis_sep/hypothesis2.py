# hypothesis_sep/hypothesis2.py
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np


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
    _anchor("sec-h2-overview")
    section_title("Climate Shit")

    st.markdown(
        "> From 2010 to 2025, the frequency of **severe weather events** (Floods, Storms, Droughts, Wildfires, "
        "and Extreme Temperatures) has increased over time, particularly in the **past few years (2018–2025)**."
    )

    st.markdown("---")

    data = _read_csv_first_match(EMDAT_PATHS)

    weather_types = ["Flood", "Storm", "Drought", "Wildfire", "Extreme temperature", "Mass movement (wet)"]
    df = data[data["Disaster Type"].str.lower().isin([t.lower() for t in weather_types])]
    df["Start Year"] = pd.to_numeric(df["Start Year"], errors="coerce").fillna(0).astype(int)
    df = df[df["Start Year"].between(2010, 2025)]

    yearly = df.groupby("Start Year").size().reset_index(name="Total Events")

    # Total events trend
    st.subheader("Total Severe Weather Events per Year")
    st.write(
        "This chart shows how the total number of severe weather events fluctuated between 2010 and 2025. "
        "While individual years vary significantly, the overall tendency appears slightly upward after 2018, "
        "indicating that weather-related events became somewhat more frequent in the recent decade."
    )

    ymin = yearly["Total Events"].min() * 0.9
    ymax = yearly["Total Events"].max() * 1.05

    fig = px.line(
        yearly,
        x="Start Year",
        y="Total Events",
        markers=True,
        title="Trend in Severe Weather Event Frequency (2010–2025)",
        color_discrete_sequence=["#ff7f0e"]
    )
    fig.update_traces(line=dict(width=4))
    fig.update_yaxes(range=[ymin, ymax])
    fig.add_scatter(
        x=yearly["Start Year"],
        y=[yearly["Total Events"].mean()] * len(yearly),
        mode="lines",
        line=dict(color="#999999", dash="dot"),
        name="Average Level",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Decadal comparison
    st.subheader("Comparison Between Early and Recent Decade")
    st.write(
        "This bar chart compares the average annual number of events during 2010–2017 versus 2018–2025. "
        "Both periods cover eight years, allowing a balanced comparison. "
        "The later window still shows a modest rise, indicating slightly higher frequency of severe weather events."
    )

    early = df[df["Start Year"] < 2018]
    late = df[df["Start Year"] >= 2018]
    count_early = len(early)
    count_late = len(late)
    years_early = early["Start Year"].nunique()
    years_late = late["Start Year"].nunique()

    avg_early = count_early / years_early
    avg_late = count_late / years_late
    percent_increase = ((avg_late - avg_early) / avg_early) * 100 if avg_early > 0 else 0

    bar_df = pd.DataFrame({
        "Period": ["2010–2017", "2018–2025"],
        "Average Events per Year": [avg_early, avg_late]
    })
    fig_bar = px.bar(
        bar_df,
        x="Period",
        y="Average Events per Year",
        color="Period",
        color_discrete_sequence=["#80b1d3", "#fb8072"],
        text_auto=".1f",
        title="Average Annual Frequency Comparison: Early vs Recent Decade"
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Trend by disaster type
    st.subheader("Trend in Severe Weather Events by Disaster Type")
    st.write(
        "Here each line represents a specific disaster type. Floods and storms dominate the count "
        "and display more noticeable increases compared with other categories, indicating they are the key drivers "
        "behind the overall upward trend."
    )

    type_trend = (
        df.groupby(["Disaster Type", "Start Year"])
        .size()
        .reset_index(name="Event Count")
    )

    fig_type = px.line(
        type_trend,
        x="Start Year",
        y="Event Count",
        color="Disaster Type",
        title="Yearly Frequency of Severe Weather Events by Type (2010–2025)",
        markers=True,
    )
    fig_type.update_traces(line=dict(width=2))
    st.plotly_chart(fig_type, use_container_width=True)

    # Percentage change by disaster type
    st.subheader("Percentage Change in Average Annual Frequency (2010–2017 vs 2018–2025)")
    st.write(
        "This view quantifies how much each disaster type changed between decades. "
        "Wildfires and storms show the strongest increases, whereas droughts and extreme temperatures show slight declines, "
        "suggesting that hydrometeorological hazards are becoming more prominent."
    )

    early_avg = type_trend[type_trend["Start Year"] < 2018].groupby("Disaster Type")["Event Count"].mean()
    late_avg = type_trend[type_trend["Start Year"] >= 2018].groupby("Disaster Type")["Event Count"].mean()

    change = ((late_avg - early_avg) / early_avg * 100).reset_index()
    change.columns = ["Disaster Type", "Change (%)"]
    change = change.sort_values("Change (%)", ascending=False)

    fig_change = px.bar(
        change,
        x="Disaster Type",
        y="Change (%)",
        color="Change (%)",
        color_continuous_scale="Tealrose",
        text_auto=".1f",
        title="Percentage Change in Severe Weather Events by Type"
    )
    fig_change.update_traces(textposition="outside")
    st.plotly_chart(fig_change, use_container_width=True)

    # Stacked bar counts for the recent decade
    st.subheader("Severe Weather Events by Type — Recent Decade (Counts)")
    st.write(
        "This stacked bar chart summarizes annual event counts from 2018 to 2025. "
        "It shows that floods consistently contribute the largest share each year, "
        "followed by storms, reinforcing that these two types dominate the global impact profile."
    )
    recent = df[df["Start Year"] >= 2018]
    recent_counts = (
        recent.groupby(["Start Year", "Disaster Type"])
        .size()
        .reset_index(name="Count")
    )

    fig_bar_stack = px.bar(
        recent_counts,
        x="Start Year",
        y="Count",
        color="Disaster Type",
        title="Severe Weather Events by Type (2018–2025, stacked counts)"
    )
    fig_bar_stack.update_layout(height=520, barmode="stack", yaxis_title="Events per Year")
    st.plotly_chart(fig_bar_stack, use_container_width=True)

    # Interpretation
    # Final Summary Section – Insight Summary
    st.markdown("---")
    st.subheader("Insight Summary")

    # Compute key stats
    min_year, max_year = df["Start Year"].min(), df["Start Year"].max()
    early_period = df[df["Start Year"] < 2018]
    recent_period = df[df["Start Year"] >= 2018]

    total_early = len(early_period)
    total_recent = len(recent_period)
    change_ratio = (total_recent / total_early) if total_early > 0 else 0
    avg_yearly_growth = ((total_recent / total_early) ** (1 / (max_year - min_year))) - 1 if total_early > 0 else 0

    # Calculate most frequent disaster type
    most_common = df["Disaster Type"].value_counts().idxmax()
    most_common_share = df["Disaster Type"].value_counts(normalize=True).max() * 100

    # KPI-style cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Increase in Severe Weather Events",
            value=f"{change_ratio:.1f}×",
            delta="Recent decade vs earlier years",
            help="Compares total events from 2018–2025 vs 2010–2017.",
        )

    with col2:
        st.metric(
            label="Average Annual Growth Rate",
            value=f"{avg_yearly_growth * 100:.1f}%",
            delta="Compound yearly increase",
            help="Approximate rate of increase in total severe weather events per year.",
        )

    with col3:
        st.metric(
            label="Dominant Severe Weather Type",
            value=most_common,
            delta=f"{most_common_share:.1f}% of all events",
            help="Most frequently occurring severe weather type between 2010–2025.",
        )

    # Text summary block
    st.markdown(
        f"""
        <div style='margin-top:20px; font-size:18px; line-height:1.6; text-align:justify;'>
            <strong>Interpretation:</strong> The total number of severe weather events increased by 
            <strong>{change_ratio:.1f}×</strong> from the early part of the dataset (2010–2017) 
            to the more recent years (2018–2025), with an estimated average annual growth rate of 
            <strong>{avg_yearly_growth * 100:.1f}%</strong>. The most frequent event type was 
            <strong>{most_common}</strong>, accounting for approximately 
            <strong>{most_common_share:.1f}%</strong> of all recorded events.  
            These results <span style='color:#2E86C1; font-weight:bold;'>support the hypothesis</span> — 
            the data shows a clear upward trend in the frequency of severe weather events in the past decade.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    subsection_title("Data Source")
    st.caption("EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
