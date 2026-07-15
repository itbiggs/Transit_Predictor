import sqlite3
import pandas as pd
import joblib
import json
from pathlib import Path

MODEL_PATH = "xgb_model.pkl"
OUTPUT_PATH = "predictions.geojson"

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training script"""
    df = df[df["arrival_time"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)]
    df["hour"] = df["arrival_time"].str.slice(0, 2).astype(int) % 24
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")

    # Day of week
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    else:
        df["day_of_week"] = df.index % 7

    # One-hot encode categorical variables
    if "conditions" in df.columns:
        df = pd.get_dummies(df, columns=["conditions"], drop_first=True)

    if "route_type" in df.columns:
        df = pd.get_dummies(df, columns=["route_type"], drop_first=True, prefix="route_type")

    # Sample stops (same top 50 as training)
    if "stop_id" in df.columns:
        top_stops = df["stop_id"].value_counts().nlargest(50).index
        df = df[df["stop_id"].isin(top_stops)]
        df = pd.get_dummies(df, columns=["stop_id"], drop_first=True)

    return df

def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading data...")
    conn = sqlite3.connect("smart_transit.db")
    df = pd.read_sql_query("SELECT * FROM labeled_with_spatial", conn)
    stops = pd.read_sql_query("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops", conn)
    conn.close()

    print(f"Loaded {len(df)} records")

    # Keep identifiers before preprocessing
    cols_to_keep = ["stop_id", "hour", "temp", "precip", "wind_speed",
                    "distance_from_loop", "is_transfer_hub"]
    if "conditions" in df.columns:
        cols_to_keep.append("conditions")
    identifiers = df[cols_to_keep].copy()

    # Preprocess for prediction
    df_processed = preprocess_for_prediction(df)

    # Prepare features (drop same columns as training)
    drop_cols = [
        "arrival_time", "departure_time", "trip_id", "delayed",
        "stop_headsign", "pickup_type", "shape_dist_traveled",
        "icon", "timestamp", "stop_lat", "stop_lon", "route_id", "delay_minutes"
    ]

    X = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns], errors="ignore")

    # Make predictions
    print("Making predictions...")
    predictions = model.predict_proba(X)[:, 1]  # Probability of delay

    # Combine predictions with identifiers
    result = identifiers.iloc[df_processed.index].copy()
    result["delay_probability"] = predictions

    # Group by stop and hour to get aggregate predictions
    print("Aggregating predictions by stop and hour...")
    agg_dict = {
        "delay_probability": "mean",
        "temp": "first",
        "precip": "first",
        "wind_speed": "first",
        "distance_from_loop": "first",
        "is_transfer_hub": "first"
    }
    if "conditions" in result.columns:
        agg_dict["conditions"] = lambda x: x.iloc[0] if len(x) > 0 else "Unknown"

    agg = result.groupby(["stop_id", "hour"]).agg(agg_dict).reset_index()

    # Merge with stops to get coordinates and names
    agg = agg.merge(stops, on="stop_id", how="left")

    # Filter out stops without coordinates
    agg = agg.dropna(subset=["stop_lat", "stop_lon"])

    print(f"Creating GeoJSON with {len(agg)} features...")

    # Create GeoJSON
    features = []
    for _, row in agg.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["stop_lon"]), float(row["stop_lat"])]
            },
            "properties": {
                "stop_id": str(row["stop_id"]),
                "stop_name": str(row["stop_name"]) if pd.notna(row["stop_name"]) else "Unknown",
                "hour": int(row["hour"]),
                "delay_probability": round(float(row["delay_probability"]), 3),
                "temp": round(float(row["temp"]), 1) if pd.notna(row["temp"]) else None,
                "precip": round(float(row["precip"]), 1) if pd.notna(row["precip"]) else None,
                "wind_speed": round(float(row["wind_speed"]), 1) if pd.notna(row["wind_speed"]) else None,
                "conditions": str(row["conditions"]) if "conditions" in row.index and pd.notna(row["conditions"]) else "Unknown",
                "distance_from_loop": round(float(row["distance_from_loop"]), 2) if pd.notna(row["distance_from_loop"]) else None,
                "is_transfer_hub": bool(row["is_transfer_hub"])
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # Save GeoJSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(geojson, f)

    print(f"\n=== GeoJSON Generation Complete ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total features: {len(features)}")
    print(f"Unique stops: {agg['stop_id'].nunique()}")
    print(f"Hour range: {agg['hour'].min()}-{agg['hour'].max()}")
    print(f"Delay probability range: {agg['delay_probability'].min():.2%} - {agg['delay_probability'].max():.2%}")

if __name__ == "__main__":
    main()
