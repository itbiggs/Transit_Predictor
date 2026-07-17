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
    all_dates = sorted(df['date'].unique())

    # Select ~1 date per month across the year
    import numpy as np
    indices = np.linspace(0, len(all_dates)-1, 12, dtype=int)
    unique_dates = [all_dates[i] for i in indices]

    print(f"\nGenerating predictions for {len(unique_dates)} dates (monthly across the year)...")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_features_by_date = {}

    # Get all unique RAIL stops (only those in our labeled data)
    all_stops = df['stop_id'].unique()
    all_hours = list(range(24))

    print(f"Using {len(all_stops)} rail stops (not all {len(stops)} stops in database)")

    # Pre-compute stop spatial features (distance_from_loop, is_transfer_hub)
    stop_spatial_features = df[['stop_id', 'distance_from_loop', 'is_transfer_hub']].drop_duplicates('stop_id')

    for date in unique_dates:
        date_str = str(date.date())
        print(f"  Processing {date_str}...")

        # Filter data for this date
        date_df = df[df['date'] == date].copy()

        if len(date_df) == 0:
            continue

        # Create complete grid using pandas (much faster than loops)
        # Create cartesian product of all stops × all hours
        grid_stops = pd.DataFrame({'stop_id': all_stops})
        grid_hours = pd.DataFrame({'hour': all_hours})
        grid_stops['_key'] = 1
        grid_hours['_key'] = 1
        complete_grid = grid_stops.merge(grid_hours, on='_key').drop('_key', axis=1)

        # Get average weather for this date
        avg_weather = {
            'temp': date_df['temp'].mean(),
            'precip': date_df['precip'].mean(),
            'wind_speed': date_df['wind_speed'].mean(),
            'date': date
        }

        # Add date weather to grid
        for key, val in avg_weather.items():
            complete_grid[key] = val

        # Merge with stop spatial features
        complete_grid = complete_grid.merge(stop_spatial_features, on='stop_id', how='left')

        # Fill any missing spatial features with defaults
        complete_grid['distance_from_loop'] = complete_grid['distance_from_loop'].fillna(0)
        complete_grid['is_transfer_hub'] = complete_grid['is_transfer_hub'].fillna(0)

        # Merge with actual data where available (to get real values instead of averages)
        date_df_subset = date_df[['stop_id', 'hour', 'temp', 'precip', 'wind_speed']].copy()
        complete_grid = complete_grid.merge(
            date_df_subset,
            on=['stop_id', 'hour'],
            how='left',
            suffixes=('_avg', '')
        )

        # Use actual values where available, otherwise use averages
        complete_grid['temp'] = complete_grid['temp'].fillna(complete_grid['temp_avg'])
        complete_grid['precip'] = complete_grid['precip'].fillna(complete_grid['precip_avg'])
        complete_grid['wind_speed'] = complete_grid['wind_speed'].fillna(complete_grid['wind_speed_avg'])

        # Drop average columns
        complete_grid = complete_grid.drop(columns=[c for c in complete_grid.columns if c.endswith('_avg')])

        # Get a template row for other features needed
        template = date_df.iloc[0].to_dict()
        for key in template:
            if key not in complete_grid.columns:
                complete_grid[key] = template[key]

        # Override the key columns we just set
        complete_grid['date'] = date

        date_df = complete_grid.reset_index(drop=True)

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

        # Reorder columns to match model's training order
        model_feature_names = model.get_booster().feature_names
        # Only keep features that exist in both
        X = X[[col for col in model_feature_names if col in X.columns]]

        # Make predictions
        try:
            predictions = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"    Error predicting: {e}")
            continue

        # Combine with predictions (indices should match now)
        result = identifiers.copy()
        result['delay_probability'] = predictions

        # No aggregation needed since we have complete grid (each stop/hour once)
        # Just merge with stop coordinates
        result = result.merge(stops, on='stop_id', how='left')
        result = result.dropna(subset=['stop_lat', 'stop_lon'])

        # Create GeoJSON features
        features = []
        for _, row in result.iterrows():
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
                    "delay_probability": round(float(row["delay_probability"]), 2),  # Reduced precision
                    "temp": round(float(row["temp"]), 0) if pd.notna(row["temp"]) else None,  # Integer temps
                    "precip": round(float(row["precip"]), 1) if pd.notna(row["precip"]) else None,  # 1 decimal
                    "wind_speed": round(float(row["wind_speed"]), 0) if pd.notna(row["wind_speed"]) else None,  # Integer
                    "distance_from_loop": round(float(row["distance_from_loop"]), 1) if pd.notna(row["distance_from_loop"]) else None,  # 1 decimal
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
