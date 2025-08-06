import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

def time_to_minutes(t):
    try:
        h, m, s = map(int, t.split(":"))
        return h * 60 + m
    except:
        return None

def simulate_delay(hour, precipitation, wind_speed):
    prob = 0.1  # Base delay rate
    # Higher chance of delay during rush hours
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        prob += 0.1
    # More delays if it's raining or windy
    if precipitation > 0:
        prob += 0.1
    if wind_speed > 20:
        prob += 0.05

    if random.random() < prob:
        return random.randint(6, 15)
    return 0


def main():
    conn = sqlite3.connect("smart_transit.db")
    df = pd.read_sql_query("SELECT * FROM stop_times", conn)

    print("Simulating delays...")

    delays = []
    for i, row in df.iterrows():
        scheduled = time_to_minutes(row['arrival_time'])
        if scheduled is None:
            delays.append(None)
            continue

        hour = int(row['arrival_time'].split(":")[0])
        precip = row.get('precipitation', 0)
        wind = row.get('wind_speed', 0)

        delay_minutes = simulate_delay(hour, precip, wind)
        delayed = 1 if delay_minutes > 5 else 0
        delays.append(delayed)


    df["delayed"] = delays
    df.to_sql("labeled_stop_times", conn, if_exists="replace", index=False)

    conn.close()
    print("Delay labeling complete. Saved to labeled_stop_times.")

if __name__ == "__main__":
    main()
