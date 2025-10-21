import pandas as pd
import os


def load_datasets(project_root):
    """Load EONET and EM-DAT datasets."""
    print("=" * 70)
    print("LOADING DATASETS FOR MERGE")
    print("=" * 70)

    # EONET data
    eonet_path = "data/eonet_severity.csv"
    print(f"\n1. Loading EONET from {eonet_path}...")
    try:
        eonet_df = pd.read_csv(eonet_path)
        print(f"   ✓ EONET loaded: {len(eonet_df)} records")
        print(f"   Disaster types in EONET: {sorted(eonet_df['Event Type'].unique().tolist())}")
    except Exception as e:
        print(f"   ✗ Error loading EONET: {e}")
        eonet_df = pd.DataFrame()

    # EM-DAT data (needs absolute path)
    emdat_path = os.path.join(project_root, "data", "processed", "emdat_cleaned.csv")
    print(f"\n2. Loading EM-DAT from {emdat_path}...")
    try:
        emdat_df = pd.read_csv(emdat_path)
        print(f"   ✓ EM-DAT loaded: {len(emdat_df)} records")
        print(f"   Disaster types in EM-DAT: {sorted(emdat_df['Disaster Type Standardized'].unique().tolist())}")
    except Exception as e:
        print(f"   ✗ Error loading EM-DAT: {e}")
        emdat_df = pd.DataFrame()

    return eonet_df, emdat_df


def identify_missing_disasters(eonet_df, emdat_df):
    """Figure out which disasters are in EM-DAT but not in EONET."""
    print("\n" + "=" * 70)
    print("IDENTIFYING MISSING DISASTER TYPES")
    print("=" * 70)

    eonet_types = set(eonet_df['Event Type'].unique()) if len(eonet_df) > 0 else set()
    emdat_types = set(emdat_df['Disaster Type Standardized'].unique()) if len(emdat_df) > 0 else set()

    # Standardize EONET type names to match EM-DAT
    eonet_standardized = set()
    for etype in eonet_types:
        if 'flood' in etype.lower():
            eonet_standardized.add('Flood')
        elif 'storm' in etype.lower() or 'cyclone' in etype.lower():
            eonet_standardized.add('Storm')
        elif 'fire' in etype.lower():
            eonet_standardized.add('Wildfire')
        elif 'volcan' in etype.lower():
            eonet_standardized.add('Volcano')
        elif 'ice' in etype.lower():
            eonet_standardized.add('Ice Event')

    missing_in_eonet = emdat_types - eonet_standardized

    print(f"\nDisaster types in EONET (standardized): {sorted(eonet_standardized)}")
    print(f"Disaster types in EM-DAT: {sorted(emdat_types)}")
    print(f"\n✓ Disasters MISSING in EONET (to take from EM-DAT): {sorted(missing_in_eonet)}")

    return missing_in_eonet


def prepare_eonet_records(eonet_df):
    """Get EONET records ready for merging."""
    print("\n3. Preparing EONET records...")

    if len(eonet_df) == 0:
        return pd.DataFrame()

    # Build unified structure
    eonet_unified = pd.DataFrame({
        'ID': eonet_df['Event ID'].astype(str),
        'Disaster Name': eonet_df['Event Title'],
        'Disaster Type': eonet_df['Event Type'],
        'Country': '',
        'Region': '',
        'Description': eonet_df['Event Title'],
        'Latitude': eonet_df['Latitude'],
        'Longitude': eonet_df['Longitude'],
        'Severity': eonet_df['Severity'].fillna(''),
        'Source': 'EONET'
    })

    print(f"   ✓ Prepared {len(eonet_unified)} EONET records")
    return eonet_unified


def prepare_emdat_records(emdat_df, missing_types):
    """Get EM-DAT records ready, but only for disasters NOT in EONET."""
    print("\n4. Preparing EM-DAT records (only missing disaster types)...")

    if len(emdat_df) == 0:
        return pd.DataFrame()

    # Keep only disaster types not covered by EONET
    emdat_filtered = emdat_df[emdat_df['Disaster Type Standardized'].isin(missing_types)].copy()

    print(f"   Filtering EM-DAT to types: {sorted(missing_types)}")
    print(f"   Records before filter: {len(emdat_df)}")
    print(f"   Records after filter: {len(emdat_filtered)}")

    # Build severity info from deaths/affected counts
    def create_severity(row):
        if row['Total Deaths'] > 0:
            return f"{int(row['Total Deaths'])} deaths"
        elif row['Total Affected'] > 0:
            return f"{int(row['Total Affected'])} affected"
        return ""

    emdat_filtered['Severity_Combined'] = emdat_filtered.apply(create_severity, axis=1)

    # Build unified structure
    emdat_unified = pd.DataFrame({
        'ID': 'EMDAT_' + emdat_filtered['DisNo.'].astype(str),
        'Disaster Name': emdat_filtered['Event Name'].fillna(''),
        'Disaster Type': emdat_filtered['Disaster Type Standardized'],
        'Country': emdat_filtered['Country'],
        'Region': emdat_filtered['Region'],
        'Description': emdat_filtered['Location'].fillna(''),
        'Latitude': emdat_filtered['Latitude'],
        'Longitude': emdat_filtered['Longitude'],
        'Severity': emdat_filtered['Severity_Combined'],
        'Source': 'EM-DAT'
    })

    print(f"   ✓ Prepared {len(emdat_unified)} EM-DAT records (missing disaster types only)")
    return emdat_unified


def merge_datasets(eonet_unified, emdat_unified):
    """Combine EONET and EM-DAT into one dataset."""
    print("\n5. Merging datasets...")

    merged = pd.concat([eonet_unified, emdat_unified], ignore_index=True)

    print(f"   ✓ Merged dataset created")
    print(f"   - EONET records: {len(eonet_unified)}")
    print(f"   - EM-DAT records: {len(emdat_unified)}")
    print(f"   - Total records: {len(merged)}")

    return merged


def save_merged_dataset(merged_df, project_root, output_filename="merged_disasters.csv"):
    """Save the merged dataset."""
    print("\n6. Saving merged dataset...")

    output_path = os.path.join(project_root, "data", "processed", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"   ✓ Merged dataset saved to: {output_path}")


def main():
    print("\n" + "=" * 70)
    print("MERGING EONET + EM-DAT DATASETS")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    eonet_df, emdat_df = load_datasets(project_root)

    if len(eonet_df) == 0 or len(emdat_df) == 0:
        print("\n✗ Cannot merge: One or both datasets failed to load")
        return

    missing_types = identify_missing_disasters(eonet_df, emdat_df)

    eonet_unified = prepare_eonet_records(eonet_df)
    emdat_unified = prepare_emdat_records(emdat_df, missing_types)

    merged = merge_datasets(eonet_unified, emdat_unified)

    save_merged_dataset(merged, project_root, output_filename="merged_emdat_eonet.csv")

    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"\nFinal dataset:")
    print(f"  Total records: {len(merged)}")
    print(f"  Disaster types: {sorted(merged['Disaster Type'].unique().tolist())}")
    print(f"  Records with coordinates: {merged[['Latitude', 'Longitude']].notna().all(axis=1).sum()}")
    print(f"  Records with severity: {(merged['Severity'].str.len() > 0).sum()}")
    print("\n Ready for environmental analysis and mapping")
    print("=" * 70)

    print("\nFirst few rows:")
    print(merged.head(10))


if __name__ == "__main__":
    main()