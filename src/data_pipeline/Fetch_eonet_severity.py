"""
fetch_eonet_severity.py
-----------------------
Fetches natural event data from NASA EONET API v3 and extracts unified severity
information across all event types.

Outputs a CSV at data/eonet_severity.csv with columns:
- Event ID, Event Title, Event Type, Event Date, Year, Latitude, Longitude, Source
- SeverityValue, SeverityUnit, Severity, SeveritySource
"""

import re
import requests
import pandas as pd
from datetime import datetime
import os

QUALITATIVE_LEVELS = {
    "extreme": 3,
    "severe": 2,
    "moderate": 1
}

def _infer_severity_from_title(event_type: str, title: str):
    """Infer severity from event title text by pattern matching."""
    if not isinstance(title, str) or not title.strip():
        return (None, None, "", "")

    t = title.lower()
    et = (event_type or "").lower()

    # Earthquakes: "M6.3" or "Magnitude 7"
    if "earthquake" in et:
        m = re.search(r'\bm(?:agnitude)?\s*([0-9]+(?:\.[0-9])?)', t)
        if m:
            val = float(m.group(1))
            return (val, "Mw", f"M{val}", "title:Mw")

    # Storms / Cyclones / Hurricanes
    if any(k in et for k in ["storm", "cyclone", "hurricane", "typhoon"]):
        m = re.search(r'\bcat(?:egory)?[\s:-]*([1-5])\b', t)
        if m:
            val = int(m.group(1))
            return (val, "SSHWS", f"Category {val}", "title:SSHWS")

    # Volcanoes: VEI scale
    if "volcan" in et:
        m = re.search(r'\bvei[\s:-]*([0-8])\b', t)
        if m:
            val = int(m.group(1))
            return (val, "VEI", f"VEI {val}", "title:VEI")

    # Wildfires: burned area (acres or hectares)
    if "fire" in et:
        m = re.search(r'([0-9][0-9,\.]*)\s*(acres?|ha|hectares?)\b', t)
        if m:
            num_txt = m.group(1).replace(",", "")
            try:
                val = float(num_txt)
            except ValueError:
                val = None
            unit = m.group(2).lower()
            unit = "acre" if unit.startswith("acre") else ("ha" if unit.startswith("ha") else "hectare")
            combo = f"{num_txt} {unit}{'' if num_txt == '1' else 's'}"
            return (val, unit, combo, "title:area")

    # Floods / Droughts: qualitative
    if "flood" in et or "drought" in et:
        for label in ["extreme", "severe", "moderate"]:
            if re.search(rf'\b{label}\b', t):
                return (QUALITATIVE_LEVELS[label], "qual", label.capitalize(), "title:qual")

    # Generic fallback for any severity words
    for label in ["extreme", "severe", "moderate"]:
        if re.search(rf'\b{label}\b', t):
            return (QUALITATIVE_LEVELS[label], "qual", label.capitalize(), "title:qual")

    return (None, None, "", "")

def _combine_severity(event_type: str, geo_point: dict, title: str):
    """Combine geometry-based magnitude with title-based inference."""
    if isinstance(geo_point, dict):
        mv = geo_point.get("magnitudeValue")
        mu = geo_point.get("magnitudeUnit")
        if mv is not None:
            try:
                val = float(mv)
            except (TypeError, ValueError):
                val = mv
            unit = mu if isinstance(mu, str) else None
            combo = f"{val} {unit}".strip() if unit else str(val)
            return (val, unit, combo, "geometry")

    # fallback: infer from title
    return _infer_severity_from_title(event_type, title)

def fetch_eonet_severity(year_start: int = 2010, year_end: int = 2025, limit: int = 10000) -> pd.DataFrame:
    """
    Fetch NASA EONET events and export to data/eonet_severity.csv.
    Returns a processed DataFrame.
    """
    print("=" * 60)
    print("FETCHING EONET DATA WITH SEVERITY FROM NASA API")
    print("=" * 60)

    base_url = "https://eonet.gsfc.nasa.gov/api/v3/events"
    params = {"limit": limit, "sort": "date", "status": "all"}

    print(f"→ Fetching from: {base_url}")
    print(f"→ Parameters: limit={limit}, status=all")
    print(f"→ Date range: {year_start}-{year_end}")

    # Fetch from API
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events_raw = data.get("events", [])
        print(f"✓ Retrieved {len(events_raw)} events from API")
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching EONET data: {e}")
        return pd.DataFrame()

    print("Processing events...")
    rows = []

    for event in events_raw:
        try:
            eid = event.get("id")
            title = event.get("title", "") or ""
            categories = event.get("categories", [])
            etype = categories[0].get("title", "Unknown") if categories else "Unknown"

            geometry = event.get("geometry", [])
            if not geometry:
                continue

            geo_point = geometry[-1]
            dstr = geo_point.get("date")
            if not dstr:
                continue

            try:
                edate = datetime.fromisoformat(dstr.replace("Z", "+00:00")).date()
            except Exception:
                continue

            if not (year_start <= edate.year <= year_end):
                continue

            coords = geo_point.get("coordinates", [None, None])
            lon, lat = (coords[0], coords[1]) if isinstance(coords, list) and len(coords) >= 2 else (None, None)
            if lat is None or lon is None:
                continue

            sources = event.get("sources", [])
            src_url = sources[0].get("url", "") if sources else ""

            # Severity extraction
            sev_val, sev_unit, sev_combo, sev_from = _combine_severity(etype, geo_point, title)

            rows.append({
                "Event ID": eid,
                "Event Title": title,
                "Event Type": etype,
                "Event Date": edate,
                "Year": edate.year,
                "Latitude": lat,
                "Longitude": lon,
                "Source": src_url,
                "SeverityValue": sev_val,
                "SeverityUnit": sev_unit,
                "Severity": sev_combo,
                "SeveritySource": sev_from
            })
        except Exception:
            continue

    if not rows:
        print("✗ No valid events found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"✓ Processed {len(df)} valid events")

    # Save to data/eonet_severity.csv
    os.makedirs("data", exist_ok=True)
    out_path = "data/eonet_severity.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Data saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EONET SEVERITY DATA SUMMARY")
    print("=" * 60)
    print(f"Total events: {len(df)}")
    print(f"Year range: {df['Year'].min()}-{df['Year'].max()}")
    print(f"Event types: {sorted(df['Event Type'].unique().tolist())}")
    print(f"Events with Severity: {(df['Severity'].astype(str).str.len() > 0).sum()}")
    print("=" * 60)

    return df

if __name__ == "__main__":
    df = fetch_eonet_severity()
    print("\nFirst few rows:")
    print(df.head())