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
    """Main section bar (registered by app.py capture)."""
    st.markdown(f'<div class="gv-section-title">{text}</div>', unsafe_allow_html=True)

def subsection_title(text: str):
    """Smaller subsection bar."""
    st.markdown(f'<div class="gv-subsection-title">{text}</div>', unsafe_allow_html=True)

def story_context(text: str):
    """One-line context/caption above a visual."""
    st.markdown(f'<div class="gv-context">{text}</div>', unsafe_allow_html=True)
    
# ===========================
# CONFIG
# ===========================
EMDAT_PATHS = [
    "data/processed/emdat_cleaned.csv",
    "data/emdat_cleaned.csv",
    "dashboard/data/emdat_cleaned.csv",
    "../data/processed/emdat_cleaned.csv",
]

# UN 2024 LDC list (44) grouped by broad region
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
    candidate_cols = [c for c in df.columns if c.lower() in {"region", "continent"}]
    if not candidate_cols:
        return None
    col = candidate_cols[0]
    tmp = df[["Country_norm", col]].dropna().copy()
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
# MAIN RENDERER
# ===========================
def render():
    _anchor("sec-h1-overview")
    section_title("Impact Gap")

    st.markdown(
        "Disaster impacts do not fall evenly across countries. In many regions, the countries that appear most often in "
        "the **worst-affected list** are those on the UN’s 2024 list of Least Developed Countries (LDCs). This view shows "
        "that pattern for one year and lets you switch regions to see if the trend holds."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    df = load_emdat()
    if df is None:
        st.warning("Could not load EM-DAT file. Please check data location.")
        return

    if "Total Affected" not in df.columns:
        df["No. Injured"]  = df.get("No. Injured", 0).fillna(0)
        df["No. Affected"] = df.get("No. Affected", 0).fillna(0)
        df["No. Homeless"] = df.get("No. Homeless", 0).fillna(0)
        df["Total Affected"] = df["No. Injured"] + df["No. Affected"] + df["No. Homeless"]

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

    df_2024 = df[df["Year"] == TARGET_YEAR].copy()
    if df_2024.empty:
        st.warning("No records found in EM-DAT for the year 2024.")
        return

    ldc_df = build_ldc_dataframe_2024()
    ldc_set_all = set(ldc_df["Country_norm"])
    macro_map = _country_to_macro_region(df_2024)

    region = st.selectbox("Select Region", REGIONS, index=0)

    if region == "All":
        chart_pool = df_2024.copy()
        chart_title_region = "World"
    else:
        if macro_map:
            allowed = {c for c, r in macro_map.items() if r == region}
            chart_pool = df_2024[df_2024["Country_norm"].isin(allowed)].copy()
        else:
            region_ldc_set = set(ldc_df.loc[ldc_df["Region"] == region, "Country_norm"])
            chart_pool = df_2024[df_2024["Country_norm"].isin(region_ldc_set)].copy()
        chart_title_region = region

    agg = _aggregate_topn(chart_pool, n=10)
    agg["Is LDC"] = agg["Country_norm"].apply(lambda c: "LDC" if c in ldc_set_all else "Non-LDC")

    total_affected_pool = int(chart_pool["Total Affected"].sum()) if not chart_pool.empty else 0
    top10_affected_total = int(agg["Total Affected"].sum()) if not agg.empty else 0
    top10_ldc_share = float(agg.loc[agg["Is LDC"] == "LDC", "Total Affected"].sum()) if not agg.empty else 0.0
    ldc_in_top10 = int((agg["Is LDC"] == "LDC").sum()) if not agg.empty else 0

    non_ldc_share = max(top10_affected_total - top10_ldc_share, 0)
    dist_df = pd.DataFrame({
        "Group": ["LDC", "Non-LDC"],
        "Total Affected": [top10_ldc_share, non_ldc_share],
    })

    color_map = {
        "Non-LDC": "rgba(180,205,230,0.9)",
        "LDC": "rgba(30,92,150,1)",
    }
    fig = px.bar(
        agg,
        y="Country_norm",
        x="Total Affected",
        orientation="h",
        color="Is LDC",
        color_discrete_map=color_map,
        category_orders={"Is LDC": ["Non-LDC", "LDC"]},
        hover_data={"Is LDC": True, "Total Affected": ":,"}
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title=f"People Affected in {TARGET_YEAR}",
        yaxis_title="Country",
        margin=dict(t=10, r=20, b=40, l=80),
        template="plotly_white",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        subsection_title(f"Top 10 Countries by People Affected: {TARGET_YEAR} - {chart_title_region}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "- Single-year view (2024) for clean comparison.\n"
            "- Ranked by **people affected**.\n"
            "- Colour shows whether the country is on the UN 2024 LDC list."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        subsection_title("Distribution in Top 10: LDC vs Non-LDC")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "The bar chart shows the distribution of affected countries that are in list of LDC vs non-LDC from the above analysis"
        )
        dist_fig = px.bar(
            dist_df,
            x="Group",
            y="Total Affected",
            color="Group",
            color_discrete_map={"LDC": "#1E5C96", "Non-LDC": "#B4CDE6"},
            text_auto=True,
            title=None,
        )
        dist_fig.update_layout(
            xaxis_title="",
            yaxis_title="People Affected (Top 10 only)",
            showlegend=False,
            template="plotly_white",
            height=320,
            margin=dict(t=20, r=20, b=40, l=40),
        )
        st.plotly_chart(dist_fig, use_container_width=True)

    # ===== Right-side: list of LDCs =====
    with col2:
        subsection_title(f"LDC Countries (2024) - {region}")
        st.markdown("<br>", unsafe_allow_html=True)
        ldc_df_full = build_ldc_dataframe_2024()
        if region == "All":
            list_df = ldc_df_full.sort_values(["Region", "Name"])
        else:
            list_df = ldc_df_full[ldc_df_full["Region"] == region].sort_values("Name")

        lines = [f"• {row.Name}" for row in list_df.itertuples(index=False)]
        if not lines:
            lines = ["—"]

        html = """
        <div class="ldc-list-box" style="max-height: 420px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 8px; border-radius: 8px; font-size: 0.95rem; line-height: 1.4;">
            {items}
        </div>
        """.format(items="<br/>".join(lines))
        st.markdown(html, unsafe_allow_html=True)

        # 🌙 dark mode fix for list visibility
        st.markdown(
            """
            <style>
            @media (prefers-color-scheme: dark){
              .ldc-list-box {
                background: rgba(15,23,42,0.9) !important;
                border: 1px solid rgba(148,163,184,0.35) !important;
                color: #e2e8f0 !important;
              }
              .ldc-list-box a { color: #e2e8f0 !important; }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Source: United Nations - List of Least Developed Countries (2024).")

    # ===== Insight Summary (KPI style like st.metric) =====
    subsection_title("Insight Summary")
    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)

    pct_val = (top10_ldc_share / top10_affected_total * 100.0) if top10_affected_total else 0.0

    with k1:
        st.metric(
            label="Total Affected in Selected Pool",
            value=f"{total_affected_pool:,}",
            delta=f"Reported burden in {chart_title_region}",
            help=f"People affected by disasters in {TARGET_YEAR} for the current selection.",
        )

    with k2:
        st.metric(
            label="LDC Share in Top 10",
            value=f"{pct_val:,.1f}%",
            delta="Portion borne by LDCs",
            help="Share of impact taken by countries on the UN LDC list within the 10 worst-hit.",
        )

    with k3:
        st.metric(
            label="LDCs in Top 10",
            value=f"{ldc_in_top10}/10",
            delta="How often LDCs appear",
            help="Count of LDCs inside the 10 worst-hit countries for this region/year.",
        )

    st.markdown(
        f"""
        <p style="margin-top:16px; font-size:0.9rem; line-height:1.6;">
            <strong>Interpretation:</strong> In <strong>{chart_title_region}</strong>, disasters in {TARGET_YEAR}
            affected about <strong>{total_affected_pool:,}</strong> people. When we look only at the 10 worst-hit
            countries, LDCs appear <strong>{ldc_in_top10}</strong> times and take on about
            <strong>{pct_val:,.1f}%</strong> of the reported impact. The small bar chart above confirms this visually,
            the LDC slice of the top-10 impact is not marginal. This supports the trend that lower-income, lower-capacity
            countries still carry a disproportionate share of human impact from disasters.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ===== References =====
    st.markdown("---")
    subsection_title("References")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "- United Nations - List of Least Developed Countries (as of December 2024)  \n"
        "  https://www.un.org/development/desa/dpad/least-developed-country-category.html"
    )
    st.markdown("---")
    st.caption("Source: EM-DAT – Centre for Research on the Epidemiology of Disasters (CRED).")
