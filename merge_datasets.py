# merge_datasets.py
import pandas as pd
import os

# Paths to your CSVs
emdat_path = os.path.join("data", "emdat_cleaned.csv")
eonet_path = os.path.join("data", "eonet_cleaned.csv")
output_path = os.path.join("data", "processed", "merged_emdat_eonet.csv")

# Create 'processed' folder if it doesn't exist
os.makedirs(os.path.join("data", "processed"), exist_ok=True)

# Load CSVs
emdat = pd.read_csv(emdat_path)
eonet = pd.read_csv(eonet_path)

# Normalize column names
emdat.columns = [col.lower().strip() for col in emdat.columns]
eonet.columns = [col.lower().strip() for col in eonet.columns]

# Keep only necessary columns from EONET to match EM-DAT format
eonet_subset = eonet[['event id', 'event title', 'disaster type', 'latitude', 'longitude', 'source url']]
eonet_subset = eonet_subset.rename(columns={
    'event id': 'id',
    'event title': 'disaster name',
    'source url': 'source'
})
# Add missing EM-DAT columns as empty
for col in ['country', 'region', 'description', 'severity']:
    if col not in eonet_subset.columns:
        eonet_subset[col] = ""

# Reorder columns to match EM-DAT
eonet_subset = eonet_subset[['id', 'disaster name', 'disaster type', 'country', 'region', 'description', 'latitude', 'longitude', 'severity', 'source']]

# Merge
merged = pd.concat([emdat, eonet_subset], ignore_index=True)

# Save
merged.to_csv(output_path, index=False)
print(f"Merged CSV saved at {output_path}")
