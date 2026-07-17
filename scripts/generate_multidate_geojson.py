import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta

MODEL_PATH = "xgb_model.pkl"
OUTPUT_DIR = "docs/predictions"

def get_season(timestamp):
    """Determine season from timestamp"""
    if pd.isna(timestamp):
        return None
    month = timestamp.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training"""
    df = df[df["arrival_time"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)]
    df["hour"] = df["arrival_time"].str.slice(0, 2).astype(int) % 24

    # One-hot encode categorical variables (must match training)
    categorical_cols = []
    if "conditions" in df.columns:
        categorical_cols.append("conditions")
    if "route_name" in df.columns:
        categorical_cols.append("route_name")
    if "direction" in df.columns:
        categorical_cols.append("direction")

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

def main():
    print("="*60)
    print("GENERATING MULTI-DATE GEOJSON PREDICTIONS")
    print("="*60)

    # Load model
    print("\nLoading trained model...")
    model = joblib.load(MODEL_PATH)

    # Load feature importance
    try:
        with open("feature_importance.json", 'r') as f:
            importance_data = json.load(f)
        print(f"Model AUC: {importance_data['auc_score']:.3f}")
    except:
        importance_data = None
        print("Feature importance not available")

    # Load data
    print("Loading prediction data...")
    conn = sqlite3.connect("smart_transit.db")
    df = pd.read_sql_query("SELECT * FROM labeled_with_spatial", conn)
    stops = pd.read_sql_query("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops", conn)
    conn.close()

    print(f"Loaded {len(df):,} records for {df['stop_id'].nunique()} stops")

    # Group by date to generate predictions for multiple dates
    df['date'] = pd.to_datetime(df['date'])
    unique_dates = sorted(df['date'].unique())[:10]  # Sample 10 dates across the year

    print(f"\nGenerating predictions for {len(unique_dates)} dates...")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_features_by_date = {}

    for date in unique_dates:
        date_str = str(date.date())
        print(f"  Processing {date_str}...")

        # Filter data for this date and reset index
        date_df = df[df['date'] == date].copy().reset_index(drop=True)

        if len(date_df) == 0:
            continue

        # Keep identifiers before processing
        identifiers = date_df[['stop_id', 'hour', 'temp', 'precip', 'wind_speed',
                                'distance_from_loop', 'is_transfer_hub', 'date']].copy()

        # Preprocess
        date_df_processed = preprocess_for_prediction(date_df)

        # Drop non-feature columns
        drop_cols = [
            "arrival_time", "departure_time", "trip_id", "delayed",
            "stop_headsign", "pickup_type", "shape_dist_traveled",
            "icon", "timestamp", "stop_lat", "stop_lon", "route_id",
            "delay_minutes", "stop_id", "stop_name", "date", "temp_c",
            "route_short_name", "route_type", "direction_id"
        ]
        X = date_df_processed.drop(columns=[c for c in drop_cols if c in date_df_processed.columns], errors="ignore")
        X = X.select_dtypes(include=[np.number])

        # Make predictions
        try:
            predictions = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"    Error predicting: {e}")
            continue

        # Combine with predictions (indices should match now)
        result = identifiers.copy()
        result['delay_probability'] = predictions

        # Aggregate by stop and hour
        agg = result.groupby(['stop_id', 'hour']).agg({
            'delay_probability': 'mean',
            'temp': 'mean',
            'precip': 'mean',
            'wind_speed': 'mean',
            'distance_from_loop': 'first',
            'is_transfer_hub': 'first',
            'date': 'first'
        }).reset_index()

        # Merge with stop coordinates
        agg = agg.merge(stops, on='stop_id', how='left')
        agg = agg.dropna(subset=['stop_lat', 'stop_lon'])

        # Create GeoJSON features
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
                    "precip": round(float(row["precip"]), 2) if pd.notna(row["precip"]) else None,
                    "wind_speed": round(float(row["wind_speed"]), 1) if pd.notna(row["wind_speed"]) else None,
                    "distance_from_loop": round(float(row["distance_from_loop"]), 2) if pd.notna(row["distance_from_loop"]) else None,
                    "is_transfer_hub": bool(row["is_transfer_hub"]),
                    "date": date_str,
                    "season": get_season(pd.Timestamp(row["date"]))
                }
            }
            features.append(feature)

        all_features_by_date[date_str] = features
        print(f"    Generated {len(features)} features")

    # Save combined GeoJSON with all dates
    combined_geojson = {
        "type": "FeatureCollection",
        "features": [f for features in all_features_by_date.values() for f in features],
        "dates": list(all_features_by_date.keys())
    }

    output_path = Path(OUTPUT_DIR) / "predictions_multidate.geojson"
    with open(output_path, 'w') as f:
        json.dump(combined_geojson, f)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Total features: {len(combined_geojson['features']):,}")
    print(f"Dates: {len(all_features_by_date)}")
    print(f"Unique stops: {len(set(f['properties']['stop_id'] for f in combined_geojson['features']))}")

    # Also save feature importance for web display
    if importance_data:
        with open(Path(OUTPUT_DIR) / "feature_importance.json", 'w') as f:
            json.dump(importance_data, f, indent=2)
        print(f"Feature importance saved for web display")

if __name__ == "__main__":
    main()
