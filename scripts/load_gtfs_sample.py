import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("smart_transit.db")
GTFS_DIR = Path("data/google_transit")
SAMPLE_SIZE = 50000  # Sample for faster testing

def load_csv_to_sql(filename, table_name, conn, sample=None):
    path = GTFS_DIR / filename
    if path.exists():
        print(f"Loading {filename} into table '{table_name}'...")
        if sample and filename == "stop_times.txt":
            # Sample stop_times for faster processing
            print(f"  Sampling {sample} rows...")
            df = pd.read_csv(path, nrows=sample)
        else:
            df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"  Loaded {len(df)} rows")
    else:
        print(f"File {filename} not found.")

def main():
    conn = sqlite3.connect(DB_PATH)

    load_csv_to_sql("stops.txt", "stops", conn)
    load_csv_to_sql("trips.txt", "trips", conn)
    load_csv_to_sql("stop_times.txt", "stop_times", conn, sample=SAMPLE_SIZE)

    conn.close()
    print("GTFS loading complete (sampled).")

if __name__ == "__main__":
    main()
