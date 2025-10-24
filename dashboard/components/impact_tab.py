"""
impact_tab.py
-------------
Advanced visual analytics for "Impact of Natural Disasters"
using the EM-DAT cleaned dataset.
Includes: interactive filters, modern visuals, country-level and correlation analyses.
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

    # ---- Add sticky CSS ----
    st.markdown("""
    <style>
    /* Sticky filter box */
    [data-testid="column"]:nth-of-type(2) > div {
        position: sticky;
        top: 90px;          /* adjust based on your header height */
        align-self: flex-start !important;
    }

    /* Optional: make it visually stand out */
    .sticky-filter {
        background-color: rgba(30, 30, 30, 0.95);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #444;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    df = load_emdat_data()

    # ======= Layout: main visuals + fixed filter panel =======
    col_main, col_filter = st.columns([4, 1], gap="large")

    # --------------------------------------
    # Fixed Filter Panel (Right)
    # --------------------------------------
    with col_filter:
        st.markdown("### 🔧 Filters")
        st.markdown('<div class="sticky-filter">', unsafe_allow_html=True)

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

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------
    # Filter Data
    # --------------------------------------
    metric_col = selected_metric
    filtered_df = df[(df["Start Year"] >= selected_years[0]) & (df["Start Year"] <= selected_years[1])]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    # --------------------------------------
    # Main Visuals Area
    # --------------------------------------
    with col_main:
        st.markdown("---")

        # ===== 1️⃣ Global Overview (Improved Choropleth Map) =====
        st.subheader(f"🌎 Global {selected_metric} by Country")

        map_data = filtered_df.groupby("Country", as_index=False)[metric_col].sum()
        fig_map = px.choropleth(
            map_data,
            locations="Country",
            locationmode="country names",
            color=metric_col,
            hover_name="Country",
            hover_data={metric_col: ":,0f"},
            color_continuous_scale="plasma",
            title=f"{selected_metric} by Country ({selected_years[0]}–{selected_years[1]})"
        )

        fig_map.update_geos(
            visible=False,
            showcountries=True,
            countrycolor="#404040",
            showcoastlines=True,
            coastlinecolor="#404040",
            projection_type="natural earth"
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_colorbar=dict(title=selected_metric, tickformat=".0s"),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # ===== 2️⃣ Top 10 Countries =====
        st.subheader(f"🏆 Top 10 Countries by {selected_metric}")
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
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ===== 3️⃣ Distribution by Disaster Type =====
        st.subheader(f"📊 Distribution of {selected_metric} by Disaster Type")
        disaster_df = filtered_df.groupby("Disaster Type", as_index=False)[metric_col].sum()
        fig_pie = px.pie(
            disaster_df,
            names="Disaster Type",
            values=metric_col,
            hole=0.4,
            title=f"Share of {selected_metric} by Disaster Type"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # ===== 4️⃣ Yearly Trend =====
        st.subheader(f"📈 Trend of {selected_metric} Over Time")
        trend_df = filtered_df.groupby("Start Year", as_index=False)[metric_col].sum()
        fig_line = px.line(
            trend_df,
            x="Start Year",
            y=metric_col,
            markers=True,
            title=f"{selected_metric} Over Time ({selected_years[0]}–{selected_years[1]})"
        )
        fig_line.update_traces(line_color="#FF8800")
        st.plotly_chart(fig_line, use_container_width=True)

        # ===== 5️⃣ Comparative Analysis =====
        st.subheader("🔍 Comparative Analysis")

        comparison_mode = st.selectbox(
            "Compare by:",
            ["Country", "Disaster Type", "Time Period"]
        )

        st.markdown("---")

        if comparison_mode == "Country":
            st.write(f"#### Top 20 Countries by {selected_metric}")

            comp_df = (
                filtered_df.groupby("Country", as_index=False)[metric_col]
                .sum()
                .sort_values(metric_col, ascending=False)
            )

            # Limit to top 20 for readability
            top_n = 20
            display_df = comp_df.head(top_n)
            others_sum = comp_df.iloc[top_n:][metric_col].sum()
            display_df = pd.concat([
                display_df,
                pd.DataFrame({"Country": ["Other"], metric_col: [others_sum]})
            ], ignore_index=True)

            fig_comp = px.bar(
                display_df,
                x="Country",
                y=metric_col,
                color=metric_col,
                color_continuous_scale="plasma",
                text_auto=".2s",
                title=f"Top {top_n} Countries by {selected_metric}",
            )
            fig_comp.update_layout(
                xaxis_tickangle=-45,
                height=600,
                margin=dict(l=20, r=20, t=60, b=80),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(title=selected_metric),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        elif comparison_mode == "Disaster Type":
            st.write(f"#### {selected_metric} by Disaster Type")
            comp_df = filtered_df.groupby("Disaster Type", as_index=False)[metric_col].sum()
            fig_comp = px.bar(
                comp_df,
                x="Disaster Type",
                y=metric_col,
                color="Disaster Type",
                text_auto=".2s",
                title=f"{selected_metric} Distribution by Disaster Type",
            )
            fig_comp.update_layout(
                xaxis_tickangle=-30,
                height=500,
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=80),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        else:  # Time Period
            st.write(f"#### {selected_metric} Over Time")
            comp_df = filtered_df.groupby("Start Year", as_index=False)[metric_col].sum()
            fig_comp = px.area(
                comp_df,
                x="Start Year",
                y=metric_col,
                color_discrete_sequence=["#ff6600"],
                title=f"{selected_metric} Over Time",
            )
            fig_comp.update_traces(mode="lines+markers", fill="tozeroy", line=dict(width=3))
            st.plotly_chart(fig_comp, use_container_width=True)

        # ===== 6️⃣ Country-Level Analysis =====
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
                trend_country, x="Start Year", y="Total Affected",
                markers=True, color_discrete_sequence=["#FF6600"]
            )
            st.plotly_chart(fig_country_line, use_container_width=True)

        st.write("#### Top Disaster Types in Selected Country")
        type_df = country_df.groupby("Disaster Type", as_index=False)["Total Affected"].sum().sort_values(
            "Total Affected", ascending=False)
        fig_country_bar = px.bar(
            type_df.head(10),
            x="Total Affected", y="Disaster Type",
            orientation="h",
            color="Total Affected",
            color_continuous_scale="Oranges"
        )
        fig_country_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_country_bar, use_container_width=True)

        # ===== 7️⃣ Correlation Analysis =====
        st.subheader("📈 Correlation Between Impact Metrics")
        corr_df = filtered_df.groupby("Disaster Type", as_index=False)[
            ["Total Deaths", "No. Injured", "Total Affected"]
        ].sum()

        corr_matrix = corr_df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix (Impact Metrics)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        metric_x = st.selectbox("Select X-axis metric", ["Total Deaths", "No. Injured", "Total Affected"])
        metric_y = st.selectbox("Select Y-axis metric", ["Total Deaths", "No. Injured", "Total Affected"], index=2)
        fig_scatter = px.scatter(
            filtered_df,
            x=metric_x, y=metric_y,
            color="Disaster Type", size="Total Affected",
            hover_name="Country",
            title=f"Correlation between {metric_x} and {metric_y} by Disaster Type"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.info("📊 Data Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
