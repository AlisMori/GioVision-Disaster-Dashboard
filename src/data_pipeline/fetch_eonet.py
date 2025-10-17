import requests
import pandas as pd
import json
from datetime import datetime, timedelta


def fetch_eonet(year_start: int = 2010, year_end: int = 2025, limit: int = 10000) -> pd.DataFrame:
    """
    Fetch natural events from NASA EONET API.

    EONET covers: wildfires, storms, volcanoes, sea ice, lake ice, dust storms, etc.

    Args:
        year_start: Start year for filtering (default 2010)
        year_end: End year for filtering (default 2025)
        limit: Maximum events to fetch (default 10000)

    Returns:
        pd.DataFrame: Events with columns: event_id, title, event_type, date,
                      latitude, longitude, source
    """

    print("=" * 60)
    print("FETCHING EONET DATA FROM NASA API")
    print("=" * 60)

    # NASA EONET API endpoint
    base_url = "https://eonet.gsfc.nasa.gov/api/v3/events"

    # Parameters for the API request
    params = {
        'limit': limit,  # Get up to 10,000 events
        'sort': 'date',  # Sort by date
        'status': 'all'  # Include all events (closed and open)
    }

    print(f"\nFetching EONET data from: {base_url}")
    print(f"Parameters: limit={limit}, status=all")
    print(f"Date range: {year_start}-{year_end}")

    try:
        # Make the API request
        print("\nConnecting to NASA EONET API...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()  # Raise error if request failed
        print("✓ Successfully connected to API")

        # Parse the JSON response
        data = response.json()
        events_raw = data.get('events', [])
        print(f"✓ Retrieved {len(events_raw)} events from API")

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching from API: {e}")
        print("Returning empty DataFrame")
        return pd.DataFrame()

    # Process the raw event data
    print("\nProcessing events...")
    processed_events = []

    for event in events_raw:
        try:
            # Extract basic info
            event_id = event.get('id')
            title = event.get('title', '')

            # Get event type/category
            categories = event.get('categories', [])
            event_type = categories[0].get('title', 'Unknown') if categories else 'Unknown'

            # Get event date
            geometry = event.get('geometry', [])
            if not geometry:
                continue

            # Most recent geometry point (latest update)
            geo_point = geometry[-1]
            event_date_str = geo_point.get('date')

            if not event_date_str:
                continue

            # Parse date
            try:
                event_date = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).date()
                event_year = event_date.year
            except:
                continue

            # Filter by year range
            if event_year < year_start or event_year > year_end:
                continue

            # Get coordinates
            coordinates = geo_point.get('coordinates', [None, None])
            longitude = coordinates[0]
            latitude = coordinates[1]

            if latitude is None or longitude is None:
                continue

            # Get source
            sources = event.get('sources', [])
            source = sources[0].get('url', '') if sources else ''

            processed_events.append({
                'Event ID': event_id,
                'Event Title': title,
                'Event Type': event_type,
                'Event Date': event_date,
                'Year': event_year,
                'Latitude': latitude,
                'Longitude': longitude,
                'Source': source
            })

        except Exception as e:
            # Skip events that cause errors
            continue

    # Create DataFrame
    if processed_events:
        df = pd.DataFrame(processed_events)
        print(f"✓ Processed {len(df)} events with valid data")
    else:
        print("✗ No valid events found")
        df = pd.DataFrame()

    # Summary statistics
    if len(df) > 0:
        print("\n" + "=" * 60)
        print("EONET DATA SUMMARY")
        print("=" * 60)
        print(f"Total events: {len(df)}")
        print(f"Year range: {df['Year'].min()}-{df['Year'].max()}")
        print(f"Event types: {df['Event Type'].unique().tolist()}")
        print(f"Geographic coverage:")
        print(f"  Latitude range: {df['Latitude'].min():.2f} to {df['Latitude'].max():.2f}")
        print(f"  Longitude range: {df['Longitude'].min():.2f} to {df['Longitude'].max():.2f}")
        print(f"\nEvents per type:")
        print(df['Event Type'].value_counts())
        print("=" * 60)

    return df


def save_eonet_data(df: pd.DataFrame, filepath: str = "data/raw/eonet_raw.csv") -> None:
    """
    Save EONET DataFrame to CSV file.

    Args:
        df: DataFrame to save
        filepath: Where to save the CSV
    """
    if len(df) == 0:
        print("No data to save")
        return

    import os
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"\n✓ Data saved to {filepath}")


if __name__ == "__main__":
    import os

    # Get project root and build filepath
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project_root, "data", "raw", "eonet_raw.csv")

    # Fetch data
    eonet_df = fetch_eonet(year_start=2010, year_end=2025)

    # Save to CSV
    if len(eonet_df) > 0:
        save_eonet_data(eonet_df, filepath=filepath)
        print("\nFirst few rows:")
        print(eonet_df.head())
    else:
        print("Failed to fetch EONET data")