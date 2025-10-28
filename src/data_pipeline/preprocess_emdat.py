import os
import pandas as pd
import numpy as np
from datetime import datetime


def preprocess_emdat(filepath: str = "data/raw/emdat_raw.xlsx",
                     year_start: int = 2010,
                     year_end: int = 2025) -> pd.DataFrame:
    """
    Preprocess and clean EM-DAT dataset for disaster analysis.

    Changes vs your previous version:
    - DO NOT drop rows with missing Total Deaths.
    - Drop fully-empty rows.
    - Normalize common blank tokens to NaN, then fill NaN with "" ONLY for text columns.
      (Numeric NaNs are preserved for correct calculations.)
    """

    print(f"Loading EM-DAT data from {filepath}...")
    df = pd.read_excel(filepath)
    print(f"Initial records: {len(df)}")

    # --- Helper: normalize common "blank" tokens to NaN in *string-like* cells
    def _normalize_blank_tokens(frame: pd.DataFrame) -> pd.DataFrame:
        blank_like = {"", " ", "-", "--", "—", "n/a", "na", "N/A", "NA", "none", "None"}
        obj_cols = frame.select_dtypes(include=["object", "string"]).columns
        for c in obj_cols:
            frame[c] = (
                frame[c]
                .astype("string")
                .str.strip()
                .replace({val: pd.NA for val in blank_like})
            )
        return frame

    # 1) CREATE PROPER DATE COLUMN
    print("\n1. Creating date column from Start Year/Month/Day...")
    # Safely build a datetime from separate parts (month/day can be NaN)
    parts = df[['Start Year', 'Start Month', 'Start Day']].rename(
        columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'}
    ).copy()
    # Use 1 for missing month/day so we at least get a parsable date
    parts['month'] = parts['month'].fillna(1)
    parts['day'] = parts['day'].fillna(1)
    df['Event Date'] = pd.to_datetime(parts, errors='coerce')

    # 2) FILTER BY YEAR RANGE
    print(f"2. Filtering data to {year_start}-{year_end}...")
    df = df[(df['Start Year'] >= year_start) & (df['Start Year'] <= year_end)].copy()
    print(f"   Records after year filter: {len(df)}")

    # 3) REMOVE DUPLICATES
    print("3. Removing duplicate records...")
    id_col = 'DisNo.' if 'DisNo.' in df.columns else None
    duplicates_before = df.duplicated(subset=[id_col]).sum() if id_col else 0
    if id_col and duplicates_before > 0:
        print(f"   Removed {duplicates_before} duplicates by {id_col}")
        df = df.drop_duplicates(subset=[id_col], keep='first')
    else:
        print("   No duplicates found (or DisNo. column missing)")

    # 4) STANDARDIZE COUNTRY NAMES
    print("4. Standardizing country names...")
    if 'Country' in df.columns:
        df['Country'] = df['Country'].astype('string').str.strip().str.title()

    # 5) STANDARDIZE DISASTER TYPES
    print("5. Standardizing disaster types...")
    disaster_mapping = {
        'Flood': ['flood', 'flash flood', 'wet mass movement'],
        'Storm': ['storm', 'tropical cyclone', 'hurricane', 'typhoon', 'tornado', 'hail'],
        'Earthquake': ['earthquake', 'seismic'],
        'Drought': ['drought'],
        'Wildfire': ['wildfire', 'forest fire', 'land fire'],
        'Extreme Temperature': ['extreme temperature', 'heat wave', 'cold wave'],
        'Volcano': ['volcanic activity', 'volcano'],
        'Landslide': ['landslide', 'dry mass movement'],
        'Other': []
    }

    def standardize_disaster_type(disaster_type):
        if pd.isna(disaster_type):
            return 'Unknown'
        s = str(disaster_type).lower()
        for standard_type, keywords in disaster_mapping.items():
            if any(k in s for k in keywords):
                return standard_type
        return 'Other'

    if 'Disaster Type' in df.columns:
        df['Disaster Type Standardized'] = df['Disaster Type'].apply(standardize_disaster_type)
        print(f"   Standardized types: {sorted(df['Disaster Type Standardized'].dropna().unique().tolist())}")
    else:
        print("   'Disaster Type' column not found; skipping standardization.")

    # 6) CLEAN IMPACT COLUMNS (keep numeric NaNs!)
    print("6. Cleaning impact columns (Deaths, Injured, Affected, Homeless)...")
    impact_cols = [c for c in ['Total Deaths', 'No. Injured', 'No. Affected',
                               'No. Homeless', 'Total Affected'] if c in df.columns]
    for col in impact_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Don't fill with 0; keep NaN so later stats aren’t biased.
        df[col] = df[col].clip(lower=0)

    # 6b) ***REMOVED*** dropping rows with missing Total Deaths
    print("6b. Skipping removal of rows with missing Total Deaths (as requested).")

    # 7) HANDLE MISSING GEOGRAPHIC DATA (don’t force 0; keep NaN)
    print("7. Handling missing geographic data (keeping NaN for lat/lon)...")
    if 'Country' in df.columns:
        df = df.dropna(subset=['Country'])
    for coord in ['Latitude', 'Longitude']:
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors='coerce')

    # 7b) Normalize blank-like tokens in strings → NaN
    df = _normalize_blank_tokens(df)

    # 7c) DROP FULLY-EMPTY ROWS (all NaN or blanks across the whole row)
    print("7c. Dropping fully-empty rows...")
    before = len(df)
    df = df.dropna(how='all')
    print(f"   Removed {before - len(df)} fully-empty rows.")

    # 8) SELECT AND REORDER KEY COLUMNS
    print("8. Selecting key columns for analysis...")
    key_columns = [
        'DisNo.', 'Event Name', 'Country', 'Region', 'Location',
        'Start Year', 'Start Month', 'Start Day', 'Event Date',
        'Disaster Type', 'Disaster Type Standardized',
        'Latitude', 'Longitude',
        'Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Affected',
        "Total Damage ('000 US$)"
    ]
    available_columns = [c for c in key_columns if c in df.columns]
    df = df[available_columns]

    # 9) FINAL TOUCH: For *text/object* columns, fill remaining NaN with ""
    print("9. Filling remaining NaN in text columns with empty strings...")
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    df[obj_cols] = df[obj_cols].fillna("")

    # (Numeric columns remain as NaN — that’s intended.)

    # 10) SUMMARY
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    if 'Start Year' in df.columns and df['Start Year'].notna().any():
        print(f"Year range: {int(df['Start Year'].min())}-{int(df['Start Year'].max())}")
    if 'Country' in df.columns:
        print(f"Unique countries: {df['Country'].nunique()}")
    if 'Disaster Type Standardized' in df.columns:
        print(f"Disaster types: {sorted([x for x in df['Disaster Type Standardized'].unique() if isinstance(x, str)])}")
    print(f"\nMissing values (numeric NaN kept on purpose):\n{df.isnull().sum()}")
    print("=" * 60)

    return df


def save_emdat_cleaned(df: pd.DataFrame, output_path: str = "data/processed/emdat_cleaned.csv") -> None:
    """Save cleaned EM-DAT data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned EM-DAT saved to: {output_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project_root, "data", "raw", "emdat_raw.xlsx")
    cleaned_df = preprocess_emdat(filepath=filepath)
    save_emdat_cleaned(cleaned_df, os.path.join(project_root, "data", "processed", "emdat_cleaned.csv"))
    print("\nFirst few rows:")
    print(cleaned_df.head())
    print(f"\nDataframe shape: {cleaned_df.shape}")
