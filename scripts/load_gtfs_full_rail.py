import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("smart_transit.db")
GTFS_DIR = Path("data/google_transit")

def main():
    """Load ALL CTA rail/train data (exclude buses)"""
    conn = sqlite3.connect(DB_PATH)

    print("Loading GTFS data...")

    # Load routes to filter for rail only
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    print(f"Total routes: {len(routes)}")

    # CTA route_type: 0=Tram, 1=Subway/Metro, 2=Rail, 3=Bus
    # We want only rail/train (0, 1, 2), exclude buses (3)
    rail_routes = routes[routes['route_type'].isin([0, 1, 2])]
    route_names = [str(name) for name in rail_routes['route_short_name'].dropna().unique()]
    print(f"Rail routes: {len(rail_routes)} ({', '.join(route_names)})")

    # Load trips for rail routes only
    trips = pd.read_csv(GTFS_DIR / "trips.txt")
    rail_trips = trips[trips['route_id'].isin(rail_routes['route_id'])]
    print(f"Rail trips: {len(rail_trips):,}")

    # Load stop_times for rail trips only
    print("Loading stop_times (this may take a moment for 11MB+ file)...")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    print(f"Total stop_times: {len(stop_times):,}")

    # Filter to rail trips only
    rail_stop_times = stop_times[stop_times['trip_id'].isin(rail_trips['trip_id'])]
    print(f"Rail stop_times: {len(rail_stop_times):,}")

    # Load all stops
    stops = pd.read_csv(GTFS_DIR / "stops.txt")
    print(f"Total stops: {len(stops):,}")

    # Save to database
    print("\nSaving to database...")
    stops.to_sql("stops", conn, if_exists="replace", index=False)
    rail_trips.to_sql("trips", conn, if_exists="replace", index=False)
    rail_stop_times.to_sql("stop_times", conn, if_exists="replace", index=False)
    routes.to_sql("routes", conn, if_exists="replace", index=False)

    conn.close()

    # Print unique rail stops
    unique_rail_stops = rail_stop_times['stop_id'].nunique()
    print(f"\n=== Summary ===")
    print(f"Unique rail stops: {unique_rail_stops}")
    print(f"Rail routes loaded: {', '.join(route_names)}")
    print(f"Total rail stop_times: {len(rail_stop_times):,}")
    print("\nGTFS rail data loading complete.")

if __name__ == "__main__":
    main()
