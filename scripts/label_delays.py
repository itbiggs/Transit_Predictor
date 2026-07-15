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

def simulate_delay(hour, precipitation, wind_speed, upstream_delay_factor=0):
    """
    Simulate transit delays based on time of day, weather, and upstream delays.

    NOTE: This uses simulated delays, not real CTA data. The simulation incorporates
    realistic factors: rush hour congestion, weather impacts, and delay propagation
    along routes to demonstrate spatial ML modeling.
    """
    prob = 0.1  # Base delay rate

    # Higher chance of delay during rush hours
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        prob += 0.1

    # More delays if it's raining or windy
    if precipitation > 0:
        prob += 0.1
    if wind_speed > 20:
        prob += 0.05

    # Delay propagation: if there's an upstream delay, increase probability
    prob += upstream_delay_factor * 0.3  # upstream_delay_factor is 0-1 based on previous delays
    prob = min(prob, 0.8)  # Cap at 80% probability

    if random.random() < prob:
        return random.randint(6, 15)
    return 0


def main():
    conn = sqlite3.connect("smart_transit.db")
    df = pd.read_sql_query("SELECT * FROM stop_times ORDER BY trip_id, stop_sequence", conn)

    print("Simulating delays with spatial propagation...")
    print(f"Total stop times: {len(df)}")

    # Track delay state per trip to simulate propagation
    trip_delay_state = {}
    random.seed(42)  # For reproducibility

    delays = []
    delay_minutes_list = []

    for i, row in df.iterrows():
        scheduled = time_to_minutes(row['arrival_time'])
        if scheduled is None:
            delays.append(None)
            delay_minutes_list.append(0)
            continue

        hour = int(row['arrival_time'].split(":")[0]) % 24  # Handle 24+ hour times
        precip = row.get('precipitation', 0)
        wind = row.get('wind_speed', 0)
        trip_id = row.get('trip_id', '')

        # Get upstream delay factor for this trip
        upstream_factor = trip_delay_state.get(trip_id, 0)

        delay_minutes = simulate_delay(hour, precip, wind, upstream_factor)
        delayed = 1 if delay_minutes > 5 else 0
        delays.append(delayed)
        delay_minutes_list.append(delay_minutes)

        # Update trip delay state: decay over time but accumulate if delayed
        if delayed:
            trip_delay_state[trip_id] = min(1.0, upstream_factor + 0.3)
        else:
            trip_delay_state[trip_id] = max(0, upstream_factor - 0.1)

    df["delayed"] = delays
    df["delay_minutes"] = delay_minutes_list
    df.to_sql("labeled_stop_times", conn, if_exists="replace", index=False)

    conn.close()

    delay_rate = sum([1 for d in delays if d == 1]) / len([d for d in delays if d is not None])
    print(f"Delay labeling complete. Overall delay rate: {delay_rate:.1%}")
    print("Saved to labeled_stop_times.")

if __name__ == "__main__":
    main()
