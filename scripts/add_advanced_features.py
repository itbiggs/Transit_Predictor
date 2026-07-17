import sqlite3
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Chicago Loop center
LOOP_CENTER = (41.8781, -87.6298)

def calculate_distance_from_loop(lat, lon):
    """Calculate distance in km from downtown Chicago Loop"""
    if pd.isna(lat) or pd.isna(lon):
        return None
    return geodesic(LOOP_CENTER, (lat, lon)).kilometers

def main():
    print("Loading data...")
    conn = sqlite3.connect("smart_transit.db")

    df = pd.read_sql_query("SELECT * FROM labeled_with_weather", conn)
    stops = pd.read_sql_query("SELECT stop_id, stop_lat, stop_lon, stop_name FROM stops", conn)
    trips = pd.read_sql_query("SELECT trip_id, route_id, direction_id FROM trips", conn)
    routes = pd.read_sql_query("SELECT route_id, route_short_name, route_type FROM routes", conn)

    print(f"Loaded {len(df):,} records")

    # Merge reference data
    df = df.merge(stops, on='stop_id', how='left')
    df = df.merge(trips, on='trip_id', how='left')
    df = df.merge(routes, on='route_id', how='left')

    print("\n=== Engineering Advanced Spatial & ML Features ===")

    # 1. Distance from Loop
    print("1. Calculating distance from Loop...")
    df['distance_from_loop'] = df.apply(
        lambda row: calculate_distance_from_loop(row['stop_lat'], row['stop_lon']),
        axis=1
    )

    # 2. Transfer hub detection (from transfers.txt)
    try:
        transfers = pd.read_csv("data/google_transit/transfers.txt")
        transfer_stops = set(transfers['from_stop_id'].unique()) | set(transfers['to_stop_id'].unique())
        df['is_transfer_hub'] = df['stop_id'].isin(transfer_stops).astype(int)
        print(f"2. Transfer hubs identified: {df['is_transfer_hub'].sum():,} stops")
    except:
        df['is_transfer_hub'] = 0
        print("2. Transfer hubs: data not available")

    # 3. Upstream delay propagation & segment length
    print("3. Computing upstream delay propagation...")
    df = df.sort_values(['trip_id', 'stop_sequence'])

    upstream_delays = []
    segment_lengths = []
    prev_trip = None
    cumulative_delay = 0
    prev_lat, prev_lon = None, None

    for _, row in df.iterrows():
        trip_id = row['trip_id']

        if trip_id != prev_trip:
            cumulative_delay = 0
            prev_lat, prev_lon = None, None
            prev_trip = trip_id

        upstream_delays.append(cumulative_delay)

        if pd.notna(row.get('delay_minutes')):
            cumulative_delay += row['delay_minutes']

        if prev_lat is not None and pd.notna(row['stop_lat']):
            segment_lengths.append(geodesic((prev_lat, prev_lon), (row['stop_lat'], row['stop_lon'])).kilometers)
        else:
            segment_lengths.append(0)

        prev_lat = row['stop_lat']
        prev_lon = row['stop_lon']

    df['upstream_delay'] = upstream_delays
    df['segment_length'] = segment_lengths

    # 4. Route-level features
    print("4. Adding route-level features...")
    df['route_name'] = df['route_short_name']
    df['direction'] = df['direction_id']  # 0 or 1 (inbound/outbound)

    # 5. Time-based interaction features
    print("5. Creating time-based interaction features...")
    df['rush_hour'] = df['hour'].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 18) else 0)
    df['late_night'] = df['hour'].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)
    df['weekend_evening'] = ((df['is_weekend'] == 1) & (df['hour'] >= 18)).astype(int)

    # 6. Weather interaction features
    print("6. Creating weather interaction features...")
    df['cold_rush_hour'] = ((df['temp'] < 32) & (df['rush_hour'] == 1)).astype(int)
    df['rain_rush_hour'] = ((df['precip'] > 0) & (df['rush_hour'] == 1)).astype(int)
    df['extreme_weather'] = ((df['temp'] < 20) | (df['temp'] > 90) | (df['precip'] > 5)).astype(int)

    # 7. Stop position features
    print("7. Adding stop position features...")
    # Normalize stop_sequence within each trip
    df['stop_sequence_norm'] = df.groupby('trip_id')['stop_sequence'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1)
    )
    df['is_first_stop'] = (df.groupby('trip_id')['stop_sequence'].transform('min') == df['stop_sequence']).astype(int)
    df['is_last_stop'] = (df.groupby('trip_id')['stop_sequence'].transform('max') == df['stop_sequence']).astype(int)

    # 8. Historical delay patterns (simulated - in production would use real historical data)
    print("8. Simulating historical delay patterns...")
    # Stop-level average delay (by stop_id and hour)
    df['stop_hour_avg_delay'] = df.groupby(['stop_id', 'hour'])['delayed'].transform('mean')
    # Route-level average delay
    df['route_avg_delay'] = df.groupby('route_id')['delayed'].transform('mean')

    # Save enhanced dataset
    print("\nSaving enhanced dataset...")
    df.to_sql("labeled_with_spatial", conn, if_exists="replace", index=False)
    conn.close()

    print("\n=== Advanced Feature Engineering Complete ===")
    print(f"Total records: {len(df):,}")
    print(f"\nSpatial Features:")
    print(f"  - Distance from Loop: {df['distance_from_loop'].mean():.2f} km avg")
    print(f"  - Segment length: {df['segment_length'].mean():.3f} km avg")
    print(f"  - Transfer hubs: {df['is_transfer_hub'].sum():,} stops")
    print(f"  - Upstream delay: {df['upstream_delay'].mean():.2f} min avg")
    print(f"\nTemporal Features:")
    print(f"  - Rush hour trips: {df['rush_hour'].sum():,} ({df['rush_hour'].mean()*100:.1f}%)")
    print(f"  - Weekend evening: {df['weekend_evening'].sum():,}")
    print(f"  - Late night: {df['late_night'].sum():,}")
    print(f"\nInteraction Features:")
    print(f"  - Cold + rush hour: {df['cold_rush_hour'].sum():,}")
    print(f"  - Rain + rush hour: {df['rain_rush_hour'].sum():,}")
    print(f"  - Extreme weather: {df['extreme_weather'].sum():,}")
    print(f"\nRoute Features:")
    print(f"  - Unique routes: {df['route_id'].nunique()}")
    print(f"  - Unique stops: {df['stop_id'].nunique():,}")

if __name__ == "__main__":
    main()
