import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from fetch_weather import fetch_weather
import numpy as np

# Load stop_times with labels
conn = sqlite3.connect("smart_transit.db")
df = pd.read_sql_query("SELECT * FROM labeled_stop_times", conn)
conn.close()

# Clean up arrival_time format
df = df[df["arrival_time"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)]
df["hour"] = df["arrival_time"].str.slice(0, 2).astype(int)
df = df[df["hour"].between(0, 23)]

# Distribute stop times across a year for seasonal weather variation
# Use a deterministic approach based on trip_id hash to keep it reproducible
np.random.seed(42)
start_date = datetime(2025, 1, 1)
df["day_offset"] = df.index % 365  # Spread across full year
df["timestamp"] = df.apply(
    lambda row: start_date + timedelta(days=int(row["day_offset"]), hours=int(row["hour"])),
    axis=1
)

# Fetch weather for unique date+hour combinations
# Sample to reduce API calls (get ~50 representative dates across the year)
unique_timestamps = df["timestamp"].unique()
sample_timestamps = np.random.choice(unique_timestamps, min(50, len(unique_timestamps)), replace=False)

weather_cache = {}
print(f"Fetching weather for {len(sample_timestamps)} unique timestamps...")
for i, ts in enumerate(sample_timestamps):
    # Convert numpy.datetime64 to pandas Timestamp then to Python datetime
    ts_dt = pd.Timestamp(ts).to_pydatetime()
    weather = fetch_weather("chicago", ts_dt)
    if weather:
        weather_cache[ts] = weather
        if (i + 1) % 10 == 0:
            print(f"  Fetched {i + 1}/{len(sample_timestamps)}")
    else:
        print(f"  Warning: No weather data for {ts}")

# Map weather to all rows (use nearest timestamp from cache)
def get_nearest_weather(timestamp):
    """Find cached weather for nearest timestamp"""
    # Convert to pandas Timestamp for comparison
    ts = pd.Timestamp(timestamp)
    if timestamp in weather_cache:
        return weather_cache[timestamp]
    # Find nearest cached timestamp
    cached_times = list(weather_cache.keys())
    if not cached_times:
        return None
    nearest = min(cached_times, key=lambda t: abs((pd.Timestamp(t) - ts).total_seconds()))
    return weather_cache[nearest]

print("Mapping weather to all records...")
df["weather_data"] = df["timestamp"].apply(get_nearest_weather)

# Filter out rows without weather data
df = df[df["weather_data"].notna()]

# Extract weather features
df["temp"] = df["weather_data"].apply(lambda w: w["temp"] if w else None)
df["precip"] = df["weather_data"].apply(lambda w: w["precip"] if w else None)
df["wind_speed"] = df["weather_data"].apply(lambda w: w["wind_speed"] if w else None)
df["conditions"] = df["weather_data"].apply(lambda w: w["conditions"] if w else None)
df = df.drop(columns=["weather_data"])

# Save to new table
conn = sqlite3.connect("smart_transit.db")
df.to_sql("labeled_with_weather", conn, if_exists="replace", index=False)
conn.close()

print("Weather enrichment complete. Saved to labeled_with_weather.")
