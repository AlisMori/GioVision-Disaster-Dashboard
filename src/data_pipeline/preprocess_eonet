import pandas as pd
import numpy as np
import os
from datetime import datetime


def preprocess_eonet(filepath: str = "data/raw/eonet_raw.csv",
                     year_start: int = 2021,
                     year_end: int = 2025) -> pd.DataFrame:
    """
    Preprocess and clean EONET dataset for disaster analysis.

    EONET provides near real-time event data (wildfires, storms, volcanoes, floods, ice).
    Note: EONET data is recent (2021+) and lacks impact metrics (deaths, affected).
    Best used for H2 (frequency trends) and real-time mapping.

    This function handles:
    - Loading raw CSV data
    - Standardizing column names
    - Validating and cleaning coordinates
    - Standardizing event types
    - Filtering by year range
    - Removing duplicates
    - Handling missing values

    Args:
        filepath: Path to raw EONET CSV file
        year_start: Start year for filtering (default 2021, limited by API)
        year_end: End year for filtering (default 2025)

    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis and merging with EM-DAT
    """

    print(f"Loading EONET data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()

    print(f"Initial records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # 1. STANDARDIZE COLUMN NAMES
    print("\n1. Standardizing column names...")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


    rename_map = {
        'event_id': 'Event ID',
        'event_title': 'Event Title',
        'event_type': 'Disaster Type',
        'event_date': 'Event Date',
        'year': 'Start Year',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'source': 'Source URL'
    }

    df = df.rename(columns=rename_map)

    # 2. CONVERT DATES TO DATETIME
    print("2. Converting dates to datetime...")
    df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
    df['Start Year'] = df['Event Date'].dt.year

    # 3. FILTER BY YEAR RANGE
    print(f"3. Filtering data to {year_start}-{year_end}...")
    df = df[(df['Start Year'] >= year_start) & (df['Start Year'] <= year_end)].copy()
    print(f"   Records after year filter: {len(df)}")

    # 4. VALIDATE AND CLEAN COORDINATES
    print("4. Validating geographic coordinates...")

    # Check for valid latitude/longitude ranges
    # Latitude: -90 to 90, Longitude: -180 to 180
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Remove records with invalid coordinates
    invalid_coords_before = len(df)
    df = df[
        (df['Latitude'].notna()) & (df['Longitude'].notna()) &
        (df['Latitude'] >= -90) & (df['Latitude'] <= 90) &
        (df['Longitude'] >= -180) & (df['Longitude'] <= 180)
        ].copy()
    invalid_coords_after = len(df)

    if invalid_coords_before > invalid_coords_after:
        print(f"   Removed {invalid_coords_before - invalid_coords_after} records with invalid coordinates")
    print(f"   Records with valid coordinates: {len(df)}")

    # 5. STANDARDIZE DISASTER TYPES
    print("5. Standardizing disaster types...")

    # Map EONET event types to standard types
    eonet_type_mapping = {
        'Wildfires': 'Wildfire',
        'Severe Storms': 'Storm',
        'Floods': 'Flood',
        'Volcanoes': 'Volcano',
        'Sea and Lake Ice': 'Ice Event',
        'Dust Storms': 'Storm',
        'Earthquakes': 'Earthquake',
        'Volcanic Activity': 'Volcano',
        'Tropical Cyclones': 'Storm'
    }

    def standardize_type(event_type):
        if pd.isna(event_type):
            return 'Unknown'
        for key, standard_type in eonet_type_mapping.items():
            if key.lower() in str(event_type).lower():
                return standard_type
        return str(event_type)

    df['Disaster Type Standardized'] = df['Disaster Type'].apply(standardize_type)
    print(f"   Standardized types: {sorted(df['Disaster Type Standardized'].unique().tolist())}")

    # 6. REMOVE DUPLICATES
    print("6. Removing duplicate records...")
    duplicates_before = df.duplicated(subset=['Event ID']).sum()
    if duplicates_before > 0:
        print(f"   Removed {duplicates_before} duplicate records")
        df = df.drop_duplicates(subset=['Event ID'], keep='first')
    else:
        print("   No duplicates found")

    # 7. CREATE MONTH COLUMN (useful for seasonality analysis)
    print("7. Creating month column for seasonality analysis...")
    df['Month'] = df['Event Date'].dt.month

    # 8. SELECT KEY COLUMNS FOR ANALYSIS
    print("8. Selecting key columns for analysis...")
    key_columns = [
        'Event ID',
        'Event Title',
        'Disaster Type',
        'Disaster Type Standardized',
        'Event Date',
        'Start Year',
        'Month',
        'Latitude',
        'Longitude',
        'Source URL'
    ]


    available_columns = [col for col in key_columns if col in df.columns]
    df = df[available_columns]

    # 9. FINAL SUMMARY
    print("\n" + "=" * 60)
    print("EONET DATA CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Year range: {df['Start Year'].min()}-{df['Start Year'].max()}")
    print(f"Disaster types: {sorted(df['Disaster Type Standardized'].unique().tolist())}")
    print(f"Geographic coverage:")
    print(f"  Latitude range: {df['Latitude'].min():.2f} to {df['Latitude'].max():.2f}")
    print(f"  Longitude range: {df['Longitude'].min():.2f} to {df['Longitude'].max():.2f}")
    print(f"\nEvents per type:")
    print(df['Disaster Type Standardized'].value_counts())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print("=" * 60)
    print("\nNote: EONET provides event occurrence data (2021+).")
    print("For impact metrics (deaths, affected), use EM-DAT data.")
    print("EONET is best used for Hypothesis 2 (frequency trends) and mapping.")
    print("=" * 60)

    return df


def save_eonet_cleaned(df: pd.DataFrame, output_path: str = "data/processed/eonet_cleaned.csv") -> None:
    """Save cleaned EONET data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned EONET saved to: {output_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project_root, "data", "raw", "eonet_raw.csv")

    cleaned_df = preprocess_eonet(filepath=filepath)

    if len(cleaned_df) > 0:
        output_path = os.path.join(project_root, "data", "processed", "eonet_cleaned.csv")
        save_eonet_cleaned(cleaned_df, output_path)
        print("\nFirst few rows:")
        print(cleaned_df.head())
        print(f"\nDataframe shape: {cleaned_df.shape}")
    else:
        print("No data to process")