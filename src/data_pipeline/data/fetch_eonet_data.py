"""
fetch_eonet.py
--------------
Unified EONET data fetcher with severity extraction.
Fetches fresh data from NASA EONET API - NO CSV FILES CREATED.

Use this in your Streamlit app to fetch fresh data every time.
"""

import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional

# Qualitative severity mapping
QUALITATIVE_LEVELS = {
    "extreme": 3,
    "severe": 2,
    "moderate": 1
}

def _infer_severity_from_title(event_type: str, title: str) -> Tuple[Optional[float], Optional[str], str, str]:
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

def _combine_severity(event_type: str, geo_point: dict, title: str) -> Tuple[Optional[float], Optional[str], str, str]:
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

def _map_disaster_type(event_type: str) -> str:
    """Map EONET event types to standardized disaster types."""
    mapping = {
        'Severe Storms': 'Storm',
        'Tropical Cyclones': 'Storm',
        'Wildfires': 'Wildfire',
        'Floods': 'Flood',
        'Volcanoes': 'Volcano',
        'Earthquakes': 'Earthquake',
        'Drought': 'Drought',
        'Sea and Lake Ice': 'Ice',
        'Snow': 'Snow',
        'Landslides': 'Landslide',
        'Dust and Haze': 'Other',
        'Water Color': 'Other',
        'Manmade': 'Other',
        'Temperature Extremes': 'Other'
    }
    return mapping.get(event_type, 'Other')

def _map_severity_to_alert(severity_val: Optional[float], severity_unit: Optional[str], event_type: str) -> str:
    """
    Map severity value to alert level (Green/Orange/Red).
    This is a simplified mapping - adjust thresholds as needed.
    """
    if severity_val is None:
        return "Unknown"
    
    event_type_lower = event_type.lower()
    
    # Earthquakes (Magnitude scale)
    if "earthquake" in event_type_lower:
        if severity_val >= 7.0:
            return "Red"
        elif severity_val >= 6.0:
            return "Orange"
        else:
            return "Green"
    
    # Storms (Category scale)
    if any(k in event_type_lower for k in ["storm", "cyclone", "hurricane", "typhoon"]):
        if severity_val >= 4:
            return "Red"
        elif severity_val >= 2:
            return "Orange"
        else:
            return "Green"
    
    # Volcanoes (VEI scale)
    if "volcan" in event_type_lower:
        if severity_val >= 4:
            return "Red"
        elif severity_val >= 2:
            return "Orange"
        else:
            return "Green"
    
    # Qualitative scale
    if severity_unit == "qual":
        if severity_val >= 3:
            return "Red"
        elif severity_val >= 2:
            return "Orange"
        else:
            return "Green"
    
    # Default: Green for low values, Orange for medium, Red for high
    if severity_val > 100:
        return "Orange"
    
    return "Green"

def fetch_eonet_data(days: int = 10, limit: int = 10000) -> pd.DataFrame:
    """
    Fetch natural events from NASA EONET API with severity extraction.
    NO CSV FILES ARE CREATED - returns DataFrame directly.
    
    Args:
        days: Number of days to look back from today (default: 10)
        limit: Maximum number of events to fetch from API
    
    Returns:
        DataFrame with columns matching your visualization needs
    """
    print("=" * 60)
    print("FETCHING FRESH EONET DATA FROM NASA API")
    print("=" * 60)

    # Calculate date range for last N days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    base_url = "https://eonet.gsfc.nasa.gov/api/v3/events"
    params = {
        "limit": limit, 
        "status": "all",
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d")
    }

    print(f"→ Fetching from: {base_url}")
    print(f"→ Parameters: limit={limit}, status=all")
    print(f"→ Date range: Last {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

    # Fetch from API
    try:
        print("\n🌐 Connecting to NASA EONET API...")
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events_raw = data.get("events", [])
        print(f"✓ Retrieved {len(events_raw)} events from API")
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching EONET data: {e}")
        return pd.DataFrame()

    print("⚙️  Processing events...")
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

            coords = geo_point.get("coordinates", [None, None])
            lon, lat = (coords[0], coords[1]) if isinstance(coords, list) and len(coords) >= 2 else (None, None)
            if lat is None or lon is None:
                continue

            sources = event.get("sources", [])
            src_url = sources[0].get("url", "") if sources else ""

            # Severity extraction
            sev_val, sev_unit, sev_combo, sev_from = _combine_severity(etype, geo_point, title)
            
            # Map severity to alert level
            alert_level = _map_severity_to_alert(sev_val, sev_unit, etype)
            
            # Standardize disaster type
            disaster_type_std = _map_disaster_type(etype)
            
            # Extract month and year
            month = edate.month
            year = edate.year

            rows.append({
                "id": eid,
                "event_name": title,
                "disaster_type": etype,
                "disaster_type_standardized": disaster_type_std,
                "event_date": edate,
                "start_year": year,
                "month": month,
                "latitude": lat,
                "longitude": lon,
                "source_url": src_url,
                "source": "EONET",
                "severity_value": sev_val,
                "severity_unit": sev_unit,
                "severity": sev_combo,
                "severity_source": sev_from,
                "alert_level": alert_level,
                "country": None,  # EONET doesn't provide country directly
            })
        except Exception:
            continue

    if not rows:
        print("✗ No valid events found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"✓ Processed {len(df)} valid events")

    # Print summary
    print("\n" + "=" * 60)
    print("EONET DATA SUMMARY")
    print("=" * 60)
    print(f"📊 Total events: {len(df)}")
    print(f"📅 Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"🔥 Event types: {', '.join(sorted(df['disaster_type_standardized'].unique().tolist()))}")
    print(f"⚠️  Alert Levels: {df['alert_level'].value_counts().to_dict()}")
    print(f"📈 Events with Severity: {(df['severity'].astype(str).str.len() > 0).sum()}")
    print("=" * 60)
    print("\n✅ NO CSV FILE SAVED - Data returned directly to Streamlit!\n")

    return df

if __name__ == "__main__":
    # Test the function
    print("🧪 Testing EONET fetcher...")
    df = fetch_eonet_data(days=10)
    if not df.empty:
        print("\n📋 First 5 rows:")
        print(df.head())
        print(f"\n📊 Total events in last 10 days: {len(df)}")
    else:
        print("\n❌ No data retrieved")