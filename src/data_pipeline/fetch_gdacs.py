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


def extract_lat_lon_from_item(item, ns):
    """Extract latitude and longitude from GDACS <geo:Point> or <gdacs:geometry>."""
    # Try <geo:lat> and <geo:long> tags first
    lat = item.findtext("geo:lat", namespaces=ns)
    lon = item.findtext("geo:long", namespaces=ns)

    # Try <gdacs:geometry> fallback
    if (lat is None or lon is None) and item.find("gdacs:geometry", namespaces=ns) is not None:
        geom_xml = item.findtext("gdacs:geometry", namespaces=ns)
        if geom_xml:
            try:
                geom_root = ET.fromstring(geom_xml)
                lat = geom_root.findtext(".//lat") or lat
                lon = geom_root.findtext(".//lon") or lon
            except ET.ParseError:
                pass

    return lat, lon


def extract_severity_text(item, ns):
    """Extract severity text from GDACS feed (gdacs:severity or gdacs:severitytext)."""
    severity = item.findtext("gdacs:severity", namespaces=ns)
    severity_text = item.findtext("gdacs:severitytext", namespaces=ns)
    return severity_text or severity or ""


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
            "Severity": severity_text,
        }
        alerts.append(data)

    df = pd.DataFrame(alerts)

    # Basic typing/cleanup
    if not df.empty:
        df.dropna(subset=["eventtype", "country"], inplace=True)

        df["fromdate"] = pd.to_datetime(df["fromdate"], errors="coerce", utc=True)
        df["todate"] = pd.to_datetime(df["todate"], errors="coerce", utc=True)
        df["alertscore"] = pd.to_numeric(df["alertscore"], errors="coerce")

        df.rename(columns={
            "eventtype": "Disaster Type",
            "eventname": "Event Name",
            "country": "Country",
            "alertlevel": "Alert Level",
            "fromdate": "Start Date",
            "todate": "End Date",
            "alertscore": "Alert Score"
        }, inplace=True)

        for col in ["title", "description", "Event Name", "Severity"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        df["Days Active"] = (df["End Date"] - df["Start Date"]).dt.days

        keep = [
            "Disaster Type", "Event Name", "Country", "iso3",
            "Alert Level", "Alert Score", "Start Date", "End Date",
            "Days Active", "title", "description",
            "Latitude", "Longitude", "Severity", "url"
        ]
        df = df[[c for c in keep if c in df.columns]]

    os.makedirs("processed_data", exist_ok=True)
    out_path = "processed_data/gdacs_cleaned.csv"
    df.to_csv(out_path, index=False)

    print(f"Fetched {len(df)} alerts and saved to {out_path}")
    return df


if __name__ == "__main__":
    df = fetch_gdacs()
    print(df.head())
