import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fetch_weather import fetch_weather

def celsius_to_fahrenheit(c):
    """Convert Celsius to Fahrenheit"""
    return c * 9/5 + 32 if c is not None else None

def main():
    """
    Fetch city-wide weather for Chicago across multiple dates.
    Weather is universal for the entire city (not per-stop).
    """
    conn = sqlite3.connect("smart_transit.db")
    df = pd.read_sql_query("SELECT * FROM labeled_stop_times", conn)
    conn.close()

    # Clean up arrival_time format
    df = df[df["arrival_time"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)]
    df["hour"] = df["arrival_time"].str.slice(0, 2).astype(int) % 24

    # Distribute across full year for seasonal variation
    # Sample 30 representative dates across the year
    start_date = datetime(2025, 1, 1)
    date_samples = [start_date + timedelta(days=i*12) for i in range(30)]  # ~Every 12 days

    print(f"Fetching weather for {len(date_samples)} dates across the year...")
    print("This provides seasonal variation (Winter, Spring, Summer, Fall)")

    # Fetch weather for each date + hour combination
    weather_cache = {}
    api_calls = 0

    for date in date_samples:
        for hour in range(24):
            timestamp = date.replace(hour=hour)

            # Skip if we already have this date+hour
            date_hour_key = (date.date(), hour)
            if date_hour_key in weather_cache:
                continue

            weather = fetch_weather("chicago", timestamp)
            if weather:
                # Convert to Fahrenheit
                weather_f = {
                    "temp_f": celsius_to_fahrenheit(weather["temp"]),
                    "temp_c": weather["temp"],
                    "precip": weather["precip"],
                    "wind_speed": weather["wind_speed"],
                    "conditions": weather["conditions"],
                    "datetime": timestamp
                }
                weather_cache[date_hour_key] = weather_f
                api_calls += 1

                if api_calls % 50 == 0:
                    print(f"  Fetched {api_calls} weather records...")
            else:
                # Use default moderate weather if API fails
                weather_cache[date_hour_key] = {
                    "temp_f": 50.0,
                    "temp_c": 10.0,
                    "precip": 0.0,
                    "wind_speed": 10.0,
                    "conditions": "Clear",
                    "datetime": timestamp
                }

    print(f"Total weather records fetched: {len(weather_cache)}")

    # Assign dates to stop_times (cycle through dates)
    np.random.seed(42)
    df["date_index"] = df.index % len(date_samples)
    df["date"] = df["date_index"].apply(lambda i: date_samples[i].date())
    df["timestamp"] = df.apply(
        lambda row: datetime.combine(row["date"], datetime.min.time()) + timedelta(hours=int(row["hour"])),
        axis=1
    )

    # Map weather to each row based on date + hour
    def get_weather_for_datetime(dt):
        key = (dt.date(), dt.hour)
        return weather_cache.get(key, weather_cache.get((date_samples[0].date(), dt.hour), {}))

    df["weather_data"] = df["timestamp"].apply(get_weather_for_datetime)

    # Extract weather features (in Fahrenheit)
    df["temp"] = df["weather_data"].apply(lambda w: w.get("temp_f"))
    df["temp_c"] = df["weather_data"].apply(lambda w: w.get("temp_c"))
    df["precip"] = df["weather_data"].apply(lambda w: w.get("precip"))
    df["wind_speed"] = df["weather_data"].apply(lambda w: w.get("wind_speed"))
    df["conditions"] = df["weather_data"].apply(lambda w: w.get("conditions"))
    df = df.drop(columns=["weather_data", "date_index"])

    # Add date-based features
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Save to database
    conn = sqlite3.connect("smart_transit.db")
    df.to_sql("labeled_with_weather", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\n=== Weather Enrichment Complete ===")
    print(f"Temperature range: {df['temp'].min():.1f}°F - {df['temp'].max():.1f}°F")
    print(f"Dates covered: {df['date'].min()} to {df['date'].max()}")
    print(f"Records enriched: {len(df):,}")
    print("Saved to labeled_with_weather table.")

if __name__ == "__main__":
    main()
