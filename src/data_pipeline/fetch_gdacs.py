"""
fetch_gdacs.py
--------------
Fetches and cleans live data from the Global Disaster Alert and Coordination System (GDACS) XML feed.
Returns a clean DataFrame for dashboard use.
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json
import os
import re

# ----------------------------
# Utility Functions
# ----------------------------

def clean_text(text):
    """Remove HTML tags and extra spaces from text."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    return text.strip()

def extract_coords_from_xml(geom_xml):
    """Extract latitude and longitude from GDACS geometry_xml field."""
    if not geom_xml or pd.isna(geom_xml):
        return None, None
    try:
        # GDACS sometimes returns XML with <coordinates>...</coordinates> inside
        geom_str = re.search(r"\{.*\}", geom_xml)
        if geom_str:
            data = json.loads(geom_str.group())
            coords = data.get("coordinates", [])
            if coords and isinstance(coords[0], (int, float)):
                return coords[1], coords[0]  # lat, lon
            elif len(coords) > 0 and isinstance(coords[0], list):
                return coords[0][1], coords[0][0]
    except Exception:
        pass
    return None, None

# ----------------------------
# Main Function
# ----------------------------

def fetch_gdacs():
    print("Fetching GDACS data...")

    url = "https://www.gdacs.org/xml/rss.xml"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GDACS data: {e}")
        return pd.DataFrame()

    # Parse XML
    root = ET.fromstring(response.content)
    ns = {'gdacs': 'http://www.gdacs.org'}

    alerts = []
    for item in root.findall(".//item"):
        data = {
            "title": item.findtext("title"),
            "description": item.findtext("description"),
            "eventtype": item.findtext("gdacs:eventtype", namespaces=ns),
            "eventname": item.findtext("gdacs:eventname", namespaces=ns),
            "country": item.findtext("gdacs:country", namespaces=ns),
            "iso3": item.findtext("gdacs:iso3", namespaces=ns),
            "alertlevel": item.findtext("gdacs:alertlevel", namespaces=ns),
            "alertscore": item.findtext("gdacs:alertscore", namespaces=ns),
            "fromdate": item.findtext("gdacs:fromdate", namespaces=ns),
            "todate": item.findtext("gdacs:todate", namespaces=ns),
            "url": item.findtext("link"),
            "geometry_xml": item.findtext("gdacs:geometry", namespaces=ns)
        }
        alerts.append(data)

    df = pd.DataFrame(alerts)

    # Handle missing geometry
    if "geometry_xml" not in df.columns:
        df["geometry_xml"] = None

    # Basic cleaning
    df.dropna(subset=["eventtype", "country"], inplace=True)
    df["fromdate"] = pd.to_datetime(df["fromdate"], errors="coerce")
    df["todate"] = pd.to_datetime(df["todate"], errors="coerce")
    df["alertscore"] = pd.to_numeric(df["alertscore"], errors="coerce")

    # Rename columns
    df.rename(columns={
        "eventtype": "Disaster Type",
        "eventname": "Event Name",
        "country": "Country",
        "alertlevel": "Alert Level",
        "fromdate": "Start Date",
        "todate": "End Date",
        "alertscore": "Alert Score"
    }, inplace=True)

    # Clean text fields
    df["title"] = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)
    df["Event Name"] = df["Event Name"].apply(clean_text)

    # Extract lat/lon from geometry_xml
    df["Latitude"], df["Longitude"] = zip(*df["geometry_xml"].apply(extract_coords_from_xml))

    # Add Days Active
    df["Days Active"] = (df["End Date"] - df["Start Date"]).dt.days

    # Keep relevant columns
    df = df[[
        "Disaster Type", "Event Name", "Country", "iso3",
        "Alert Level", "Alert Score", "Start Date", "End Date",
        "Days Active", "title", "description", "Latitude", "Longitude", "url"
    ]]


    # Save cleaned data
    os.makedirs("processed_data", exist_ok=True)
    processed_csv_path = "data/gdacs_cleaned.csv"
    df.to_csv(processed_csv_path, index=False)

    print(f"Fetched {len(df)} alerts and saved to {processed_csv_path}")
    return df

# Uncomment to test independently
if __name__ == "__main__":
   df = fetch_gdacs()
   print(df.head())
