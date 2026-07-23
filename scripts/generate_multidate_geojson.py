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

def ensure_fahrenheit(temp):
    if pd.isna(temp):
        return None
    if temp < 50:
        return (temp * 9/5) + 32
    return temp

def get_monthly_weather_averages(month):
    # Chicago monthly averages
    weather_by_month = {
        1:  {'temp': 26, 'precip': 51, 'wind_speed': 11},  # January
        2:  {'temp': 30, 'precip': 48, 'wind_speed': 11},  # February
        3:  {'temp': 41, 'precip': 68, 'wind_speed': 12},  # March
        4:  {'temp': 53, 'precip': 93, 'wind_speed': 12},  # April
        5:  {'temp': 63, 'precip': 99, 'wind_speed': 11},  # May
        6:  {'temp': 73, 'precip': 102, 'wind_speed': 10}, # June
        7:  {'temp': 77, 'precip': 99, 'wind_speed': 9},   # July
        8:  {'temp': 76, 'precip': 107, 'wind_speed': 9},  # August
        9:  {'temp': 68, 'precip': 85, 'wind_speed': 10},  # September
        10: {'temp': 56, 'precip': 83, 'wind_speed': 11},  # October
        11: {'temp': 42, 'precip': 76, 'wind_speed': 11},  # November
        12: {'temp': 31, 'precip': 63, 'wind_speed': 11},  # December
    }
    return weather_by_month.get(month, weather_by_month[1])

def apply_hourly_temperature_variation(base_temp, hour):
    # Hourly temperature adjustments
    hour_adjustment = {
        0: -8, 1: -9, 2: -9, 3: -10, 4: -10, 5: -9,  # Night (coldest)
        6: -7, 7: -5, 8: -2, 9: 1, 10: 3, 11: 5,     # Morning (warming)
        12: 6, 13: 7, 14: 7, 15: 7, 16: 6, 17: 4,    # Afternoon (warmest)
        18: 2, 19: 0, 20: -2, 21: -4, 22: -6, 23: -7  # Evening (cooling)
    }
    return base_temp + hour_adjustment.get(hour, 0)

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
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

    # Select the 15th of each month
    unique_dates = []
    month_names = []
    for month in range(1, 13):
        # Try to find data from the 15th of this month, or closest date
        month_data = df[df['date'].dt.month == month]
        if len(month_data) > 0:
            target_date = pd.Timestamp(year=df['date'].dt.year.mode()[0], month=month, day=15)
            available_dates = month_data['date'].unique()
            # Find the closest date to the 15th
            closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
            unique_dates.append(closest_date)
            month_names.append(pd.Timestamp(closest_date).strftime('%B'))

    print(f"\nGenerating predictions for {len(unique_dates)} months...")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_features_by_date = {}

    # Get all unique RAIL stops (only those in our labeled data)
    all_stops = df['stop_id'].unique()
    all_hours = list(range(24))

    print(f"Using {len(all_stops)} rail stops (not all {len(stops)} stops in database)")

    # Pre-compute stop spatial features (distance_from_loop, is_transfer_hub)
    stop_spatial_features = df[['stop_id', 'distance_from_loop', 'is_transfer_hub']].drop_duplicates('stop_id')

    for idx, date in enumerate(unique_dates):
        month_name = month_names[idx]
        date_str = str(date.date())
        print(f"  Processing {month_name} ({date_str})...")

        # Filter data for this date
        date_df = df[df['date'] == date].copy()

        if len(date_df) == 0:
            continue

        # Create grid of all stop/hour combinations
        grid_stops = pd.DataFrame({'stop_id': all_stops})
        grid_hours = pd.DataFrame({'hour': all_hours})
        grid_stops['_key'] = 1
        grid_hours['_key'] = 1
        complete_grid = grid_stops.merge(grid_hours, on='_key').drop('_key', axis=1)

        monthly_weather = get_monthly_weather_averages(date.month)
        avg_weather = {
            'temp': monthly_weather['temp'],
            'precip': monthly_weather['precip'],
            'wind_speed': monthly_weather['wind_speed'],
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

        # Preserve spatial features
        spatial_cols = ['stop_id', 'distance_from_loop', 'is_transfer_hub']
        preserved_spatial = complete_grid[spatial_cols + ['hour']].copy()

        # Merge with actual data, excluding spatial columns
        cols_to_merge = [c for c in date_df.columns
                        if c not in ['stop_lat', 'stop_lon', '_key', 'distance_from_loop', 'is_transfer_hub']]
        date_df_subset = date_df[cols_to_merge].copy()

        # Aggregate to one row per stop/hour
        numeric_cols = date_df_subset.select_dtypes(include=[np.number]).columns
        agg_dict = {col: 'mean' for col in numeric_cols if col not in ['stop_id', 'hour']}
        # Keep first value for non-numeric columns
        for col in date_df_subset.columns:
            if col not in ['stop_id', 'hour'] and col not in numeric_cols:
                agg_dict[col] = 'first'

        if agg_dict:
            date_df_subset = date_df_subset.groupby(['stop_id', 'hour'], as_index=False).agg(agg_dict)

        complete_grid = complete_grid.drop(columns=['distance_from_loop', 'is_transfer_hub'])
        complete_grid = complete_grid.merge(
            date_df_subset,
            on=['stop_id', 'hour'],
            how='left',
            suffixes=('_avg', '')
        )

        # Restore spatial features
        complete_grid = complete_grid.drop(columns=['distance_from_loop', 'is_transfer_hub'], errors='ignore')
        complete_grid = complete_grid.merge(
            preserved_spatial,
            on=['stop_id', 'hour'],
            how='left'
        )

        complete_grid['temp'] = complete_grid['temp'].fillna(complete_grid['temp_avg'])
        complete_grid['precip'] = complete_grid['precip'].fillna(complete_grid['precip_avg'])
        complete_grid['wind_speed'] = complete_grid['wind_speed'].fillna(complete_grid['wind_speed_avg'])
        complete_grid['date'] = complete_grid['date'].fillna(date)

        complete_grid = complete_grid.drop(columns=[c for c in complete_grid.columns if c.endswith('_avg')], errors='ignore')
        complete_grid = complete_grid.drop_duplicates(subset=['stop_id', 'hour'], keep='first')

        # Apply temperature variation by hour
        base_temp = monthly_weather['temp']
        complete_grid['temp'] = complete_grid['hour'].apply(
            lambda h: apply_hourly_temperature_variation(base_temp, h)
        )

        # Fill missing columns
        if 'day_of_week' not in complete_grid.columns:
            complete_grid['day_of_week'] = date.dayofweek
        if 'month' not in complete_grid.columns:
            complete_grid['month'] = date.month
        if 'is_weekend' not in complete_grid.columns:
            complete_grid['is_weekend'] = 1 if date.dayofweek >= 5 else 0

        # Fill other numeric columns with date means where missing
        for col in date_df.select_dtypes(include=[np.number]).columns:
            if col in complete_grid.columns and col not in ['stop_id', 'hour', 'temp', 'precip', 'wind_speed']:
                complete_grid[col] = complete_grid[col].fillna(date_df[col].mean())

        # Ensure all rows have valid arrival_time for preprocessing
        if 'arrival_time' not in complete_grid.columns or complete_grid['arrival_time'].isna().any():
            # Create arrival_time from hour field
            complete_grid['arrival_time'] = complete_grid['hour'].apply(lambda h: f"{int(h):02d}:00:00")

        date_df = complete_grid.reset_index(drop=True)

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

        # Keep identifiers BEFORE dropping columns but AFTER preprocessing
        identifiers = date_df_processed[['stop_id', 'hour', 'temp', 'precip', 'wind_speed',
                                          'distance_from_loop', 'is_transfer_hub', 'date']].copy()

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

        expected_count = len(all_stops) * 24
        actual_count = len(result)
        if actual_count != expected_count:
            print(f"    WARNING: Expected {expected_count} predictions, got {actual_count}")

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
                    "date": month_name,  # Use month name instead of date
                    "season": get_season(pd.Timestamp(row["date"]))
                }
            }
            features.append(feature)

        all_features_by_date[month_name] = features
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
