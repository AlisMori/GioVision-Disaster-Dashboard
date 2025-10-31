# hypothesis_sep/hypothesis1.py
import os
import re
from typing import List, Dict, Optional, Tuple

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

# ---- Hardcoded LDC list for 2024 (44 countries) ----
# Regions aligned to UN usage (Africa, Asia, Oceania, Americas)
LDC_2024: List[Tuple[str, str]] = [
    # Africa (32)
    ("Angola", "Africa"), ("Benin", "Africa"), ("Burkina Faso", "Africa"), ("Burundi", "Africa"),
    ("Central African Republic", "Africa"), ("Chad", "Africa"), ("Comoros", "Africa"),
    ("Democratic Republic of the Congo", "Africa"), ("Djibouti", "Africa"), ("Eritrea", "Africa"),
    ("Ethiopia", "Africa"), ("Gambia", "Africa"), ("Guinea", "Africa"), ("Guinea-Bissau", "Africa"),
    ("Lesotho", "Africa"), ("Liberia", "Africa"), ("Madagascar", "Africa"), ("Malawi", "Africa"),
    ("Mali", "Africa"), ("Mauritania", "Africa"), ("Mozambique", "Africa"), ("Niger", "Africa"),
    ("Rwanda", "Africa"), ("Senegal", "Africa"), ("Sierra Leone", "Africa"), ("Somalia", "Africa"),
    ("South Sudan", "Africa"), ("Sudan", "Africa"), ("Togo", "Africa"), ("Uganda", "Africa"),
    ("United Republic of Tanzania", "Africa"), ("Zambia", "Africa"),
    # Asia (8)
    ("Afghanistan", "Asia"), ("Bangladesh", "Asia"), ("Cambodia", "Asia"),
    ("Lao People's Democratic Republic", "Asia"), ("Myanmar", "Asia"), ("Nepal", "Asia"),
    ("Timor-Leste", "Asia"), ("Yemen", "Asia"),
    # Oceania (3)
    ("Kiribati", "Oceania"), ("Solomon Islands", "Oceania"), ("Tuvalu", "Oceania"),
    # Americas (1)
    ("Haiti", "Americas"),
]

REGIONS = ["All", "Africa", "Asia", "Oceania", "Americas"]
TARGET_YEAR = 2024

# ===========================
# HELPERS
# ===========================
def _read_csv_first_match(paths) -> Optional[pd.DataFrame]:
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None

def normalize_country_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r"[0-9¹²³⁴⁵⁶⁷⁸⁹]+$", "", str(name).strip())
    name = name.replace("’", "'").replace("–", "-").replace("—", "-")
    name = " ".join(name.split())
    return name

def load_emdat(paths=EMDAT_PATHS):
    return _read_csv_first_match(paths)

def build_ldc_dataframe_2024() -> pd.DataFrame:
    df = pd.DataFrame(LDC_2024, columns=["Name", "Region"])
    df["Country_norm"] = df["Name"].apply(normalize_country_name)
    return df

def _aggregate_topn(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    agg = (
        df.groupby("Country_norm", as_index=False)["Total Affected"]
          .sum()
          .sort_values("Total Affected", ascending=False)
          .head(n)
          .reset_index(drop=True)
    )
    return agg

def _map_to_macro_region(label: str) -> Optional[str]:
    """
    Map assorted region/continent labels from EMDAT into one of our macro-regions.
    Returns one of {"Africa","Asia","Oceania","Americas"} or None if not mappable (e.g., Europe).
    """
    if not isinstance(label, str):
        return None
    s = label.strip().lower()
    if "africa" in s:
        return "Africa"
    if "asia" in s:
        return "Asia"
    if "oceania" in s or "pacific" in s:
        return "Oceania"
    if "america" in s:
        return "Americas"
    return None

def _country_to_macro_region(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """
    Build Country_norm -> MacroRegion mapping using EMDAT metadata if available.
    Looks for 'Region' or 'Continent' columns and normalizes them.
    """
    candidate_cols = [c for c in df.columns if c.lower() in {"region", "continent"}]
    if not candidate_cols:
        return None
    col = candidate_cols[0]
    tmp = (
        df[["Country_norm", col]]
        .dropna()
        .copy()
    )
    tmp[col] = tmp[col].apply(_map_to_macro_region)
    tmp = tmp.dropna(subset=[col])
    if tmp.empty:
        return None
    mapping = (
        tmp.groupby("Country_norm")[col]
           .agg(lambda s: s.value_counts().index[0])
           .to_dict()
    )
    return mapping

# ===========================
# MAIN RENDERER (2024, region-driven)
# ===========================
def render():
    _anchor("sec-h1-overview")
    section_title("Impact Gap")

    st.markdown(
        "> A higher number of people from **Least Developed Countries (LDCs)** are affected by natural disasters "
        "compared to developed nations."
    )
    st.markdown(
        "We use a **fixed 2024 LDC list of 44 countries** grouped by UN regions. Below we show the **Top-10 countries** "
        "by people affected in 2024 (from EM-DAT). Selecting a region filters the chart strictly to that region."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ===== Load data =====
    df = load_emdat()
    if df is None:
        st.warning("Could not load EMDAT file. Please check data location.")
        return

    # ===== Ensure Total Affected =====
    if "Total Affected" not in df.columns:
        df["No. Injured"]  = df.get("No. Injured", 0).fillna(0)
        df["No. Affected"] = df.get("No. Affected", 0).fillna(0)
        df["No. Homeless"] = df.get("No. Homeless", 0).fillna(0)
        df["Total Affected"] = df["No. Injured"] + df["No. Affected"] + df["No. Homeless"]

    # ===== Derive Year and normalize country names =====
    if "Event Date" in df.columns:
        df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")
        df["Year"] = df["Event Date"].dt.year
    elif "Start Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Start Year"], errors="coerce")
    else:
        st.warning("EM-DAT needs 'Event Date' or 'Start Year' to filter by year.")
        return

    if "Country" not in df.columns:
        st.warning("EM-DAT missing 'Country' column.")
        return
    df["Country_norm"] = df["Country"].astype(str).apply(normalize_country_name)

    # ===== Filter to TARGET_YEAR (2024) =====
    df_2024 = df[df["Year"] == TARGET_YEAR].copy()
    if df_2024.empty:
        st.warning("No records found in EM-DAT for the year 2024.")
        return

    # ===== LDCs (2024) =====
    ldc_df = build_ldc_dataframe_2024()
    ldc_set_all = set(ldc_df["Country_norm"])

    # ===== Optional macro-region mapping from EMDAT =====
    macro_map = _country_to_macro_region(df_2024)  # None if EMDAT lacks region info

    # ===== Region selector =====
    region = st.selectbox("Select Region", REGIONS, index=0)

    # ===== Build chart pool strictly for the selected region =====
    if region == "All":
        chart_pool = df_2024.copy()
        chart_title_region = "World"
    else:
        if macro_map:
            allowed = {c for c, r in macro_map.items() if r == region}
            chart_pool = df_2024[df_2024["Country_norm"].isin(allowed)].copy()
        else:
            # fallback: if EMDAT has no region info, restrict to the region's LDC set only
            region_ldc_set = set(ldc_df.loc[ldc_df["Region"] == region, "Country_norm"])
            chart_pool = df_2024[df_2024["Country_norm"].isin(region_ldc_set)].copy()
        chart_title_region = region

    # ===== Aggregate Top-N for the region (no cross-region backfill) =====
    agg = _aggregate_topn(chart_pool, n=10)

    # Mark LDC/non-LDC consistently (colors never flip)
    agg["Is LDC"] = agg["Country_norm"].apply(lambda c: "LDC" if c in ldc_set_all else "Non-LDC")

    # ===== Metrics (for the chart pool) =====
    total_affected_pool = int(chart_pool["Total Affected"].sum()) if not chart_pool.empty else 0
    top10_affected_total = int(agg["Total Affected"].sum()) if not agg.empty else 0
    top10_ldc_share = float(agg.loc[agg["Is LDC"] == "LDC", "Total Affected"].sum()) if not agg.empty else 0.0

    st.columns(1)  # spacing
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(f"Total Affected (pool, {TARGET_YEAR})", f"{total_affected_pool:,}")
    with m2:
        st.metric(f"Total Affected (Top 10, {TARGET_YEAR})", f"{top10_affected_total:,}")
    with m3:
        pct = (top10_ldc_share / top10_affected_total * 100.0) if top10_affected_total else 0.0
        st.metric("LDC Share in Top 10", f"{pct:,.1f}%")

    # ===== Chart (horizontal) =====
    color_map = {
        "Non-LDC": "rgba(180,205,230,0.9)",  # light blue
        "LDC": "rgba(30,92,150,1)",          # blue
    }
    fig = px.bar(
        agg,
        y="Country_norm",
        x="Total Affected",
        orientation="h",
        color="Is LDC",
        color_discrete_map=color_map,
        category_orders={"Is LDC": ["Non-LDC", "LDC"]},
        title=f"Top 10 Countries by People Affected - {TARGET_YEAR} - {chart_title_region}",
        hover_data={"Is LDC": True, "Total Affected": ":,"}
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title=f"People Affected in {TARGET_YEAR}",
        yaxis_title="Country"
    )

    # ===== Layout =====
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # ===== Right-side: LDC list (per-line, scrollable, region-aware) =====
    with col2:
        subsection_title(f"LDC Countries (2024) — {region}")
        ldc_df = build_ldc_dataframe_2024()  # rebuild to ensure clean sort independent of mapping
        if region == "All":
            list_df = ldc_df.sort_values(["Region", "Name"])
        else:
            list_df = ldc_df[ldc_df["Region"] == region].sort_values("Name")

        lines = [f"• {row.Name}" for row in list_df.itertuples(index=False)]
        if not lines:
            lines = ["—"]

        html = """
        <div style="max-height: 420px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 8px; border-radius: 8px; font-size: 0.95rem; line-height: 1.4;">
            {items}
        </div>
        """.format(items="<br/>".join(lines))
        st.markdown(html, unsafe_allow_html=True)

        st.caption("Source: United Nations — List of Least Developed Countries (2024).")

    # ===== References =====
    st.markdown("---")
    subsection_title("References")
    st.markdown(
        "- United Nations — List of Least Developed Countries (as of December 2024)  \n"
        "  https://www.un.org/development/desa/dpad/least-developed-country-category.html"
    )

    st.markdown("---")
    st.caption("Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
