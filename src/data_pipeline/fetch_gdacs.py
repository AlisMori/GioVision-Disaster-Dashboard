"""
fetch_gdacs.py
--------------
Fetches and cleans live data from the Global Disaster Alert and Coordination System (GDACS) RSS feed.
Outputs a CSV at processed_data/gdacs_cleaned.csv with key fields for a dashboard.

Columns include:
- Disaster Type, Event Name, Country, iso3
- Alert Level, Alert Score, Start Date, End Date, Days Active
- title, description, Latitude, Longitude, Severity (text), url
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
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GDACS data: {e}")
        return pd.DataFrame()

    # Parse XML
    root = ET.fromstring(response.content)
    ns = {
        'gdacs': 'http://www.gdacs.org',
        'geo': 'http://www.w3.org/2003/01/geo/wgs84_pos#',
        'georss': 'http://www.georss.org/georss'
    }

    alerts = []
    for item in root.findall(".//item"):
        lat, lon = extract_lat_lon_from_item(item, ns)
        severity_text = extract_severity_text(item, ns)

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
            "Latitude": lat,
            "Longitude": lon,
            "Severity": severity_text,   # single text label column
        }
        alerts.append(data)

    df = pd.DataFrame(alerts)

    # Basic typing/cleanup
    if not df.empty:
        # Drop rows without core identifiers (optional, keeps feed cleaner)
        df.dropna(subset=["eventtype", "country"], inplace=True)

        # Types
        df["fromdate"] = pd.to_datetime(df["fromdate"], errors="coerce", utc=True)
        df["todate"] = pd.to_datetime(df["todate"], errors="coerce", utc=True)
        df["alertscore"] = pd.to_numeric(df["alertscore"], errors="coerce")

        # Rename columns for dashboard-friendly names
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
        for col in ["title", "description", "Event Name", "Severity"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        # Days Active
        df["Days Active"] = (df["End Date"] - df["Start Date"]).dt.days

        # Keep relevant columns (in order)
        keep = [
            "Disaster Type", "Event Name", "Country", "iso3",
            "Alert Level", "Alert Score", "Start Date", "End Date",
            "Days Active", "title", "description",
            "Latitude", "Longitude", "Severity", "url"
        ]
        df = df[[c for c in keep if c in df.columns]]


    # Save cleaned data
    os.makedirs("processed_data", exist_ok=True)
    out_path = "processed_data/gdacs_cleaned.csv"
    df.to_csv(out_path, index=False)

    print(f"Fetched {len(df)} alerts and saved to {out_path}")
    return df

# Run directly here
#if __name__ == "__main__":
    df = fetch_gdacs()
    print(df.head())
