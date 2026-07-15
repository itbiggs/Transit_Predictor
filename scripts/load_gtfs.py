import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("smart_transit.db")
GTFS_DIR = Path("data/google_transit")

def load_csv_to_sql(filename, table_name, conn):
    path = GTFS_DIR / filename
    if path.exists():
        print(f"Loading {filename} into table '{table_name}'...")
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    else:
        print(f"File {filename} not found.")

def main():
    conn = sqlite3.connect(DB_PATH)

    load_csv_to_sql("stops.txt", "stops", conn)
    load_csv_to_sql("trips.txt", "trips", conn)
    load_csv_to_sql("stop_times.txt", "stop_times", conn)

    conn.close()
    print("GTFS loading complete.")

if __name__ == "__main__":
    main()