"""
impact_tab.py
-------------
Visual analytics for "Impact of Natural Disasters" using the EM-DAT cleaned dataset.
This version matches the actual CSV structure provided.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
import os


# --------------------------------------
# Load and Prepare Data
# --------------------------------------
@st.cache_data
def load_emdat_data():
    possible_paths = [
        "data/processed/emdat_cleaned.csv",
        "data/emdat_cleaned.csv",
        "dashboard/data/emdat_cleaned.csv",
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        st.error("❌ Could not locate 'emdat_cleaned.csv' in data folders.")
        st.stop()

    # Normalize column names
    df.columns = [col.strip() for col in df.columns]

    # Keep only relevant columns
    required_columns = [
        "Country",
        "Region",
        "Disaster Type",
        "Start Year",
        "Total Deaths",
        "No. Injured",
        "Total Affected"
    ]

    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Missing column in dataset: {col}")
            st.stop()

    df = df[required_columns]

    # Clean numeric data
    numeric_cols = ["Total Deaths", "No. Injured", "Total Affected"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Start Year"] = df["Start Year"].astype(int)

    return df


# --------------------------------------
# Visualization Logic
# --------------------------------------
def render():
    st.header("🌍 Impact of Natural Disasters")
    st.caption("Analysis of human impact from the EM-DAT dataset (cleaned).")

    df = load_emdat_data()

    # Filters
    years = sorted(df["Start Year"].dropna().unique())
    min_year, max_year = int(min(years)), int(max(years))
    selected_years = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    selected_region = st.selectbox(
        "Select Region",
        ["All"] + sorted(df["Region"].dropna().unique().tolist())
    )

    selected_metric = st.selectbox(
        "Select Impact Metric",
        ["Total Affected", "Total Deaths", "No. Injured"]
    )

    metric_col = selected_metric
    filtered_df = df[(df["Start Year"] >= selected_years[0]) & (df["Start Year"] <= selected_years[1])]

    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    st.markdown("---")

    # 1️⃣ Global Overview - Choropleth Map
    st.subheader(f"Global {selected_metric} by Country")
    map_data = filtered_df.groupby("Country", as_index=False)[metric_col].sum()

    fig_map = px.choropleth(
        map_data,
        locations="Country",
        locationmode="country names",
        color=metric_col,
        color_continuous_scale="YlOrRd",
        title=f"{selected_metric} by Country ({selected_years[0]}–{selected_years[1]})",
        hover_name="Country"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # 2️⃣ Top 10 Countries
    st.subheader(f"Top 10 Countries by {selected_metric}")
    top10 = map_data.sort_values(metric_col, ascending=False).head(10)
    fig_bar = px.bar(
        top10,
        x=metric_col,
        y="Country",
        orientation="h",
        text=metric_col,
        color=metric_col,
        color_continuous_scale="Reds"
    )
    fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3️⃣ Distribution by Disaster Type
    st.subheader(f"Distribution of {selected_metric} by Disaster Type")
    disaster_df = filtered_df.groupby("Disaster Type", as_index=False)[metric_col].sum()
    fig_pie = px.pie(
        disaster_df,
        names="Disaster Type",
        values=metric_col,
        hole=0.4,
        title=f"Share of {selected_metric} by Disaster Type"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # 4️⃣ Yearly Trend
    st.subheader(f"Trend of {selected_metric} Over Time")
    trend_df = filtered_df.groupby("Start Year", as_index=False)[metric_col].sum()
    fig_line = px.line(
        trend_df,
        x="Start Year",
        y=metric_col,
        markers=True,
        title=f"{selected_metric} Over Time ({selected_years[0]}–{selected_years[1]})"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    st.info("📊 Data Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")

    # 5️⃣ Comparative Analysis
    st.subheader("🔍 Comparative Analysis")

    comparison_mode = st.selectbox(
        "Compare by:",
        ["Country", "Disaster Type", "Time Period"]
    )

    if comparison_mode == "Country":
        comp_df = filtered_df.groupby("Country", as_index=False)[metric_col].sum()
        fig_comp = px.box(filtered_df, x="Country", y=metric_col,
                          title=f"Distribution of {selected_metric} by Country",
                          color="Country")
    elif comparison_mode == "Disaster Type":
        comp_df = filtered_df.groupby("Disaster Type", as_index=False)[metric_col].sum()
        fig_comp = px.box(filtered_df, x="Disaster Type", y=metric_col,
                          title=f"Distribution of {selected_metric} by Disaster Type",
                          color="Disaster Type")
    else:  # Time Period
        comp_df = filtered_df.groupby("Start Year", as_index=False)[metric_col].sum()
        fig_comp = px.bar(comp_df, x="Start Year", y=metric_col,
                          title=f"{selected_metric} per Year", color=metric_col)

    st.plotly_chart(fig_comp, use_container_width=True)

    # 6️⃣ Country-Level Analysis
    st.subheader("🏳️ Country-Level Analysis")

    selected_country = st.selectbox(
        "Select a Country for Detailed Analysis",
        sorted(filtered_df["Country"].unique())
    )

    country_df = filtered_df[filtered_df["Country"] == selected_country]

    col1, col2 = st.columns(2)

    with col1:
        total_deaths = country_df["Total Deaths"].sum()
        total_injured = country_df["No. Injured"].sum()
        total_affected = country_df["Total Affected"].sum()

        st.metric("Total Deaths", f"{int(total_deaths):,}")
        st.metric("Injured", f"{int(total_injured):,}")
        st.metric("Total Affected", f"{int(total_affected):,}")

    with col2:
        st.write("#### Trend of Total Affected Over Time")
        trend_country = country_df.groupby("Start Year", as_index=False)["Total Affected"].sum()
        fig_country_line = px.line(
            trend_country,
            x="Start Year", y="Total Affected",
            markers=True,
            color_discrete_sequence=["#FF6600"]
        )
        st.plotly_chart(fig_country_line, use_container_width=True)

    # Top disaster types in selected country
    st.write("#### Top Disaster Types in Selected Country")
    type_df = country_df.groupby("Disaster Type", as_index=False)["Total Affected"].sum().sort_values("Total Affected",
                                                                                                      ascending=False)
    fig_country_bar = px.bar(
        type_df.head(10),
        x="Total Affected",
        y="Disaster Type",
        orientation="h",
        color="Total Affected",
        color_continuous_scale="Oranges"
    )
    fig_country_bar.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_country_bar, use_container_width=True)
