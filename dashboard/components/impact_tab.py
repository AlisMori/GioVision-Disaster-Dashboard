"""
Impact of Natural Disasters Tab
--------------------------------
Provides advanced visual analytics using the EM-DAT cleaned dataset.
Includes: interactive filters, choropleth map, trend analysis, country-level views,
and correlation analysis.

Aligned to the unified Streamlit app structure and style used in app.py.
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st


# SECTION HELPERS (consistent with app.py)
def section_title(text: str):
    """Main section bar (registered by app.py capture)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)


def subsection_title(text: str):
    """Smaller subsection bar."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)


# DATA LOADING
@st.cache_data
def load_emdat_data() -> pd.DataFrame:
    """Load cleaned EM-DAT dataset from common locations."""
    paths = [
        "data/processed/emdat_cleaned.csv",
        "data/emdat_cleaned.csv",
        "dashboard/data/emdat_cleaned.csv",
        "../data/processed/emdat_cleaned.csv",
    ]
    df = next((pd.read_csv(p) for p in paths if os.path.exists(p)), None)
    if df is None:
        st.error("❌ Could not locate 'emdat_cleaned.csv' in data folders.")
        st.stop()

    df.columns = [c.strip() for c in df.columns]
    required = ["Country", "Region", "Disaster Type", "Start Year", "Total Deaths", "No. Injured", "Total Affected"]
    for c in required:
        if c not in df.columns:
            st.error(f"❌ Missing column: {c}")
            st.stop()

    df = df[required]
    for col in ["Total Deaths", "No. Injured", "Total Affected"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Start Year"] = df["Start Year"].astype(int)
    return df


# MAIN RENDER FUNCTION
def render():
    """Render the full Impact of Natural Disasters analytics tab."""
    section_title("Overview")
    st.markdown("""
    This section provides a comprehensive analysis of how natural disasters impact human populations worldwide using the EM-DAT dataset.  
    You can:

    - Explore global patterns of deaths, injuries, and affected populations over time  
    - Filter by region, year range, and disaster type  
    - Compare countries and categories using interactive visualizations  
    - Examine trends and correlations between different impact metrics  

    It offers both global insights and detailed country-level perspectives to better understand disaster effects and resilience patterns.
    """)

    df = load_emdat_data()

    # Layout
    col_main, col_filter = st.columns([4, 1], gap="large")

    # ---------------- Filters ----------------
    with col_filter:
        st.markdown('<div class="gv-filter-card">', unsafe_allow_html=True)
        subsection_title("Filters")

        years = sorted(df["Start Year"].unique())
        selected_years = st.slider("Select Year Range", int(min(years)), int(max(years)),
                                   (int(min(years)), int(max(years))))
        selected_region = st.selectbox("Select Region", ["All"] + sorted(df["Region"].dropna().unique().tolist()))
        selected_metric = st.selectbox("Select Impact Metric", ["Total Affected", "Total Deaths", "No. Injured"])

    # Filtered data
    filtered = df[(df["Start Year"] >= selected_years[0]) & (df["Start Year"] <= selected_years[1])]
    if selected_region != "All":
        filtered = filtered[filtered["Region"] == selected_region]
    metric = selected_metric

    # Helpers for consistent, filter-aware captions
    def _region_label(r: str) -> str:
        return "all regions" if (not r or r == "All") else r

    def _years_label(yrange: tuple[int, int]) -> str:
        a, b = int(yrange[0]), int(yrange[1])
        return f"{a}–{b}" if a != b else f"{a}"

    def _cap_context(metric: str, yrange: tuple[int, int], region: str) -> str:
        return f"{metric} · {_years_label(yrange)} · {_region_label(region)}"

    # ---------------- Visuals ----------------
    with col_main:
        st.markdown("---")
        section_title(f"Global {metric} by Country")
        st.caption(
            f"Choropleth of **{_cap_context(selected_metric, selected_years, selected_region)}**. "
            "Values are aggregated across all disaster types. "
            "Use filters to change the period, region subset, and metric."
        )

        # Choropleth map
        map_df = filtered.groupby("Country", as_index=False)[metric].sum()
        fig_map = px.choropleth(
            map_df,
            locations="Country",
            locationmode="country names",
            color=metric,
            hover_name="Country",
            color_continuous_scale="plasma",
            title=f"{metric} by Country ({selected_years[0]}–{selected_years[1]})"
        )
        fig_map.update_geos(visible=False, showcountries=True, countrycolor="#888", showcoastlines=True)
        fig_map.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_map, use_container_width=True)

        # Top 10 countries
        st.markdown("---")
        subsection_title(f"Top 10 Countries by {metric}")
        st.caption(
            f"Ranks countries by **{_cap_context(selected_metric, selected_years, selected_region)}**. "
            "Bars show totals over the selected period."
        )
        top10 = map_df.sort_values(metric, ascending=False).head(10)
        fig_bar = px.bar(top10, x=metric, y="Country", orientation="h", color=metric, color_continuous_scale="Reds")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Distribution by disaster type
        st.markdown("---")
        section_title(f"Distribution of {metric} by Disaster Type")
        st.caption(
            f"Proportional breakdown of **{_cap_context(selected_metric, selected_years, selected_region)}** "
            "across disaster types. Use this to see which hazards contribute most within the filters."
        )
        dist_df = filtered.groupby("Disaster Type", as_index=False)[metric].sum()
        fig_pie = px.pie(dist_df, names="Disaster Type", values=metric, hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Yearly trend
        st.markdown("---")
        section_title(f"Trend of {metric} Over Time")
        st.caption(
            f"Annual totals of **{_cap_context(selected_metric, selected_years, selected_region)}**. "
            "Helps spot spikes and long-term movement over the chosen period."
        )
        trend_df = filtered.groupby("Start Year", as_index=False)[metric].sum()
        fig_line = px.line(trend_df, x="Start Year", y=metric, markers=True)
        fig_line.update_traces(line_color="#FF8800")
        st.plotly_chart(fig_line, use_container_width=True)

        # Comparative analysis
        st.markdown("---")
        section_title("Comparative Analysis")
        mode = st.selectbox("Compare by:", ["Country", "Disaster Type", "Time Period"])
        st.markdown("---")

        if mode == "Country":
            subsection_title(f"Top 20 Countries by {metric}")
            st.caption(
                f"Distribution of **{_cap_context(selected_metric, selected_years, selected_region)}** "
                "across countries. Use to compare national impacts under the same filters."
            )
            comp = filtered.groupby("Country", as_index=False)[metric].sum().sort_values(metric, ascending=False)
            top20 = comp.head(20)
            others = comp.iloc[20:][metric].sum()
            top20 = pd.concat([top20, pd.DataFrame({"Country": ["Other"], metric: [others]})], ignore_index=True)
            fig = px.bar(top20, x="Country", y=metric, color=metric, color_continuous_scale="plasma")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        elif mode == "Disaster Type":
            subsection_title(f"{metric} by Disaster Type")
            st.caption(
                f"Distribution of **{_cap_context(selected_metric, selected_years, selected_region)}** "
                "across disaster types. Highlights which hazards dominate under the filters."
            )
            ddf = filtered.groupby("Disaster Type", as_index=False)[metric].sum()
            fig = px.bar(ddf, x="Disaster Type", y=metric, color="Disaster Type")
            fig.update_layout(xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        else:
            subsection_title(f"{metric} Over Time")
            st.caption(
                f"{selected_metric} aggregated by year for **{_region_label(selected_region)}** "
                f"over **{_years_label(selected_years)}**. Useful for comparing periods directly."
            )

            tdf = filtered.groupby("Start Year", as_index=False)[metric].sum()
            fig = px.area(tdf, x="Start Year", y=metric, color_discrete_sequence=["#ff6600"])
            fig.update_traces(mode="lines+markers", fill="tozeroy")
            st.plotly_chart(fig, use_container_width=True)

        # Country-level analysis
        st.markdown("---")
        section_title("Country-Level Analysis")
        country = st.selectbox("Select Country", sorted(filtered["Country"].unique()))
        cdf = filtered[filtered["Country"] == country]
        st.caption(
            f"Detail for **{country}** showing **{_cap_context('Total Affected', selected_years, selected_region)}** "
            "and top contributing disaster types. Metrics and trend respect the global filters."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Deaths", f"{int(cdf['Total Deaths'].sum()):,}")
            st.metric("Injured", f"{int(cdf['No. Injured'].sum()):,}")
            st.metric("Total Affected", f"{int(cdf['Total Affected'].sum()):,}")

        with col2:
            subsection_title("Trend of Total Affected")
            tcountry = cdf.groupby("Start Year", as_index=False)["Total Affected"].sum()
            fig = px.line(tcountry, x="Start Year", y="Total Affected", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        subsection_title("Top Disaster Types in Country")
        types = cdf.groupby("Disaster Type", as_index=False)["Total Affected"].sum().sort_values("Total Affected",
                                                                                                 ascending=False)
        fig = px.bar(types.head(10), x="Total Affected", y="Disaster Type", orientation="h", color="Total Affected",
                     color_continuous_scale="Oranges")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.caption("Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
