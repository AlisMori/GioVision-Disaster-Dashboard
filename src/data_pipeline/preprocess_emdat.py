import pandas as pd
import numpy as np
from datetime import datetime


def preprocess_emdat(filepath: str = "data/raw/emdat_raw.xlsx",
                     year_start: int = 2010,
                     year_end: int = 2025) -> pd.DataFrame:
    """
    Preprocess and clean EM-DAT dataset for disaster analysis.

    This function handles:
    - Loading raw XLSX data
    - Creating proper date columns
    - Filtering to specified year range
    - Standardizing disaster types
    - Cleaning impact metrics (deaths, injuries, affected)
    - Removing duplicates
    - Handling missing geographic data

    The cleaned output is reusable for all hypotheses (H1, H2, H3).

    Args:
        filepath: Path to raw EM-DAT XLSX file
        year_start: Start year for filtering (default 2015)
        year_end: End year for filtering (default 2025)

    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis
    """

    print(f"Loading EM-DAT data from {filepath}...")
    df = pd.read_excel(filepath)
    print(f"Initial records: {len(df)}")

    # 1. CREATE PROPER DATE COLUMN
    print("\n1. Creating date column from Start Year/Month/Day...")
    df['Event Date'] = pd.to_datetime(
        df[['Start Year', 'Start Month', 'Start Day']].rename(
            columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'}
        ).fillna({'month': 1, 'day': 1}),
        errors='coerce'
    )

    # 2. FILTER BY YEAR RANGE
    print(f"2. Filtering data to {year_start}-{year_end}...")
    df = df[(df['Start Year'] >= year_start) & (df['Start Year'] <= year_end)].copy()
    print(f"   Records after year filter: {len(df)}")

    # 3. REMOVE DUPLICATES
    print("3. Removing duplicate records...")
    duplicates_before = df.duplicated(subset=['DisNo.']).sum()
    if duplicates_before > 0:
        print(f"   Removed {duplicates_before} duplicates")
        df = df.drop_duplicates(subset=['DisNo.'], keep='first')
    else:
        print("   No duplicates found")

    # 4. STANDARDIZE COUNTRY NAMES
    print("4. Standardizing country names...")
    df['Country'] = df['Country'].str.strip().str.title()

    # 5. STANDARDIZE DISASTER TYPES
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
        disaster_type_lower = str(disaster_type).lower()
        for standard_type, keywords in disaster_mapping.items():
            for keyword in keywords:
                if keyword in disaster_type_lower:
                    return standard_type
        return 'Other'

    df['Disaster Type Standardized'] = df['Disaster Type'].apply(standardize_disaster_type)
    print(f"   Standardized types: {df['Disaster Type Standardized'].unique().tolist()}")

    # 6. CLEAN IMPACT COLUMNS (Deaths, Injured, Affected, Homeless)
    print("6. Cleaning impact columns (Deaths, Injured, Affected, Homeless)...")
    impact_cols = ['Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Affected']

    for col in impact_cols:
        # Convert to numeric, coerce errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # For non-death columns, fill NaN with 0 (reasonable assumption for missing impact data)
        # But keep Total Deaths as NaN to preserve data integrity
        if col != 'Total Deaths':
            df[col] = df[col].fillna(0)
        # Ensure no negative values
        df[col] = df[col].clip(lower=0)

    # 6b. REMOVE ROWS WITH MISSING DEATH DATA
    print("6b. Removing records with missing Total Deaths (for analysis accuracy)...")
    deaths_missing_before = df['Total Deaths'].isna().sum()
    df = df.dropna(subset=['Total Deaths'])
    deaths_missing_after = df['Total Deaths'].isna().sum()
    print(f"   Removed {deaths_missing_before} records with missing death data")
    print(f"   Records remaining: {len(df)}")

    # 7. HANDLE MISSING GEOGRAPHIC DATA
    print("7. Handling missing geographic data...")
    df = df.dropna(subset=['Country'])
    df['Latitude'] = df['Latitude'].fillna(0)
    df['Longitude'] = df['Longitude'].fillna(0)
    print(f"   Records after geographic filtering: {len(df)}")

    # 8. SELECT AND REORDER KEY COLUMNS
    print("8. Selecting key columns for analysis...")
    key_columns = [
        'DisNo.', 'Event Name', 'Country', 'Region', 'Location',
        'Start Year', 'Start Month', 'Start Day', 'Event Date',
        'Disaster Type', 'Disaster Type Standardized',
        'Latitude', 'Longitude',
        'Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Affected',
        'Total Damage (\'000 US$)'
    ]

    # Keep only columns that exist in the dataframe
    available_columns = [col for col in key_columns if col in df.columns]
    df = df[available_columns]

    # 9. FINAL SUMMARY
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Year range: {df['Start Year'].min()}-{df['Start Year'].max()}")
    print(f"Unique countries: {df['Country'].nunique()}")
    print(f"Disaster types: {sorted(df['Disaster Type Standardized'].unique().tolist())}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print("=" * 60)

    return df

def save_emdat_cleaned(df: pd.DataFrame, output_path: str = "data/processed/emdat_cleaned.csv") -> None:
    """Save cleaned EM-DAT data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned EM-DAT saved to: {output_path}")

if __name__ == "__main__":
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project_root, "data", "raw", "emdat_raw.xlsx")

    cleaned_df = preprocess_emdat(filepath=filepath)
    save_emdat_cleaned(cleaned_df, os.path.join(project_root, "data", "processed", "emdat_cleaned.csv"))

    print("\nFirst few rows:")
    print(cleaned_df.head())
    print(f"\nDataframe shape: {cleaned_df.shape}")