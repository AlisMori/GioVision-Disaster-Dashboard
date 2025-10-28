"""
Environmental Overview Tab
--------------------------
Displays key statistics and high-level overview of global disaster data.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import the merge function
from src.utils.merge_datasets import merge_datasets

def render():
    st.title("🌎 Environmental Overview")
    st.markdown("""
    This section provides a **global overview of natural disasters** between **2010–2025**, 
    using combined data from **EM-DAT** and **NASA EONET**.  
    The aim is to visualize key patterns, such as frequency over time, major disaster types, 
    and their geographic distribution worldwide.
    """)

    # -----------------------------
    # Load dataset using merge_datasets()
    # -----------------------------
    try:
        data_path = merge_datasets()  # automatically merges if needed
    except FileNotFoundError as e:
        st.error(f"❌ Dataset not found: {e}")
        return

    df = pd.read_csv(data_path, low_memory=False)

    # Normalize column names to lowercase (easier to work with)
    df.columns = [col.lower().strip() for col in df.columns]

    # -----------------------------
    # Debug: Check column names
    # -----------------------------
    st.write("Columns in dataset:", df.columns.tolist())

    # -----------------------------
    # Create 'date' column
    # -----------------------------
    if 'date' not in df.columns:
        possible_cols = ['event date', 'start date']
        for col in possible_cols:
            if col in df.columns:
                df['date'] = pd.to_datetime(df[col], errors='coerce')
                st.write(f"Using '{col}' as the date column")  # Debug info
                break

    # -----------------------------
    # Filter for years 2010–2025
    # -----------------------------
    if 'date' in df.columns:
        df = df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2025)]
    else:
        st.warning("⚠️ Could not find a date column to filter years.")
        return

    # -----------------------------
    # Drop rows with missing coordinates
    # -----------------------------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])
    else:
        st.error("❌ Latitude/Longitude columns not found in dataset.")
        return

    st.markdown("---")

    # -----------------------------
    # 1️⃣ Trend of Disasters Over Time
    # -----------------------------
    st.subheader("📈 Global Disaster Trends (2010–2025)")
    yearly = df.groupby(df['date'].dt.year).size().reset_index(name='count')
    fig1 = px.line(
        yearly,
        x='date',
        y='count',
        markers=True,
        title='Trend of Global Disasters (2010–2025)',
        labels={'date': 'Year', 'count': 'Number of Disasters'}
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 2️⃣ Top Disaster Types
    # -----------------------------
    st.subheader("🌪️ Most Frequent Disaster Types")
    disaster_col = 'disaster type' if 'disaster type' in df.columns else 'disaster_type'
    type_counts = df[disaster_col].value_counts().reset_index()
    type_counts.columns = ['Disaster Type', 'Count']

    fig2 = px.bar(
        type_counts.head(10),
        x='Disaster Type',
        y='Count',
        title='Top 10 Disaster Types (2010–2025)',
        color='Disaster Type'
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 3️⃣ Global Distribution Map
    # -----------------------------
    st.subheader("🗺️ Global Distribution of Disasters")
    country_col = 'country' if 'country' in df.columns else None

    fig3 = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color=disaster_col,
        hover_name=country_col,
        title='Worldwide Disaster Locations (2010–2025)',
        projection='natural earth'
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.caption("📊 Data Sources: EM-DAT | NASA EONET")
    st.caption("Developed by Team GeoVision")
