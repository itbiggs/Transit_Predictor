import sqlite3
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Chicago Loop center coordinates
LOOP_CENTER = (41.8781, -87.6298)

def calculate_distance_from_loop(lat, lon):
    """Calculate distance in km from downtown Chicago Loop"""
    if pd.isna(lat) or pd.isna(lon):
        return None
    return geodesic(LOOP_CENTER, (lat, lon)).kilometers

def main():
    conn = sqlite3.connect("smart_transit.db")

    # Load labeled data with weather
    print("Loading labeled data...")
    df = pd.read_sql_query("SELECT * FROM labeled_with_weather", conn)
    print(f"Loaded {len(df)} records")

    # Load GTFS reference data
    print("Loading GTFS reference data...")
    stops = pd.read_sql_query("SELECT stop_id, stop_lat, stop_lon FROM stops", conn)
    trips_table = pd.read_sql_query("SELECT trip_id, route_id FROM trips", conn)

    # Load routes to get route_type
    try:
        routes = pd.read_csv("data/google_transit/routes.txt")
        routes = routes[['route_id', 'route_type']]
    except:
        print("Warning: Could not load routes.txt, route_type will be missing")
        routes = pd.DataFrame(columns=['route_id', 'route_type'])

    # Load transfers to identify transfer hubs
    try:
        transfers = pd.read_csv("data/google_transit/transfers.txt")
        transfer_stops = set(transfers['from_stop_id'].unique()) | set(transfers['to_stop_id'].unique())
    except:
        print("Warning: Could not load transfers.txt, is_transfer_hub will be all False")
        transfer_stops = set()

    # Merge stops to get lat/lon
    print("Adding stop coordinates...")
    df = df.merge(stops, on='stop_id', how='left')

    # Calculate distance from Loop
    print("Calculating distance from Loop...")
    df['distance_from_loop'] = df.apply(
        lambda row: calculate_distance_from_loop(row['stop_lat'], row['stop_lon']),
        axis=1
    )

    # Add is_transfer_hub
    print("Identifying transfer hubs...")
    df['is_transfer_hub'] = df['stop_id'].isin(transfer_stops).astype(int)

    # Merge trips to get route_id
    print("Adding route information...")
    df = df.merge(trips_table, on='trip_id', how='left')

    # Merge routes to get route_type
    if not routes.empty:
        df = df.merge(routes, on='route_id', how='left')
    else:
        df['route_type'] = np.nan

    # Calculate upstream delay and segment length
    print("Calculating upstream delays and segment lengths...")
    df = df.sort_values(['trip_id', 'stop_sequence'])

    upstream_delays = []
    segment_lengths = []

    prev_trip = None
    cumulative_delay = 0
    prev_lat, prev_lon = None, None

    for _, row in df.iterrows():
        trip_id = row['trip_id']

        # Reset for new trip
        if trip_id != prev_trip:
            cumulative_delay = 0
            prev_lat, prev_lon = None, None
            prev_trip = trip_id

        # Upstream delay is cumulative delay so far on this trip
        upstream_delays.append(cumulative_delay)

        # Add current delay to cumulative
        if pd.notna(row.get('delay_minutes')):
            cumulative_delay += row['delay_minutes']

        # Calculate segment length from previous stop
        if prev_lat is not None and pd.notna(row['stop_lat']):
            segment_length = geodesic((prev_lat, prev_lon), (row['stop_lat'], row['stop_lon'])).kilometers
        else:
            segment_length = 0
        segment_lengths.append(segment_length)

        prev_lat = row['stop_lat']
        prev_lon = row['stop_lon']

    df['upstream_delay'] = upstream_delays
    df['segment_length'] = segment_lengths

    # Save enhanced dataset
    print("Saving enhanced dataset...")
    df.to_sql("labeled_with_spatial", conn, if_exists="replace", index=False)

    conn.close()

    print("\n=== Spatial Features Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Distance from Loop - mean: {df['distance_from_loop'].mean():.2f} km")
    print(f"Transfer hubs: {df['is_transfer_hub'].sum()} stops")
    print(f"Segment length - mean: {df['segment_length'].mean():.3f} km")
    print(f"Upstream delay - mean: {df['upstream_delay'].mean():.2f} minutes")
    print(f"\nSaved to labeled_with_spatial table.")

if __name__ == "__main__":
    main()
