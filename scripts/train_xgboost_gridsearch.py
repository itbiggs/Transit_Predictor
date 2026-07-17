import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier # type: ignore
import joblib
import json

DB_PATH = "smart_transit.db"
TABLE = "labeled_with_spatial"  # Updated to use spatial features
MODEL_PATH = "xgb_model.pkl"
FEATURE_IMPORTANCE_PATH = "feature_importance.json"


def load_data(db_path: str, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess data with advanced spatial and ML features"""
    df = df[df["arrival_time"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)]
    df["hour"] = df["arrival_time"].str.slice(0, 2).astype(int) % 24
    df = df[df["delayed"].isin([0, 1])]

    # One-hot encode categorical variables
    categorical_cols = []
    if "conditions" in df.columns:
        categorical_cols.append("conditions")
    if "route_name" in df.columns:
        # One-hot encode all rail routes (not just top 50 - we have fewer rail routes)
        categorical_cols.append("route_name")
    if "direction" in df.columns:
        categorical_cols.append("direction")

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop non-feature columns
    drop_cols = [
        "arrival_time", "departure_time", "trip_id", "delayed",
        "stop_headsign", "pickup_type", "shape_dist_traveled",
        "icon", "timestamp", "stop_lat", "stop_lon", "route_id",
        "delay_minutes", "stop_id", "stop_name", "date", "temp_c",
        "route_short_name", "route_type", "direction_id"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["delayed"]

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    print(f"\n=== Advanced Feature Engineering Summary ===")
    print(f"Total samples: {len(X):,}")
    print(f"Total features: {len(X.columns)}")

    # Categorize features
    spatial_feats = [c for c in X.columns if any(x in c.lower() for x in ['distance', 'segment', 'transfer', 'upstream'])]
    temporal_feats = [c for c in X.columns if any(x in c.lower() for x in ['hour', 'day', 'week', 'month', 'rush', 'night'])]
    weather_feats = [c for c in X.columns if any(x in c.lower() for x in ['temp', 'precip', 'wind', 'weather', 'rain', 'cold'])]
    route_feats = [c for c in X.columns if any(x in c.lower() for x in ['route', 'direction'])]
    interaction_feats = [c for c in X.columns if any(x in c.lower() for x in ['cold_rush', 'rain_rush', 'weekend_evening', 'extreme'])]

    print(f"\nFeature Categories:")
    print(f"  Spatial: {len(spatial_feats)} ({', '.join(spatial_feats[:5])}...)")
    print(f"  Temporal: {len(temporal_feats)} ({', '.join([f for f in temporal_feats if '_' not in f])})")
    print(f"  Weather: {len(weather_feats)} ({len([f for f in weather_feats if 'conditions' not in f])} numeric, {len([f for f in weather_feats if 'conditions' in f])} categorical)")
    print(f"  Route: {len(route_feats)}")
    print(f"  Interactions: {len(interaction_feats)} ({', '.join(interaction_feats)})")

    print(f"\nDelay rate: {y.mean():.1%}")
    print(f"Class distribution: {y.value_counts().to_dict()}\n")

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[XGBClassifier, dict, dict]:
    """Train XGBoost model with advanced hyperparameter tuning"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_bal):,} samples")

    print("\nRunning GridSearchCV for hyperparameter tuning...")
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 7],
        "learning_rate": [0.1, 0.3],
        "min_child_weight": [1, 3],
        "subsample": [0.8, 1.0],
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='roc_auc')
    grid.fit(X_train_bal, y_train_bal)

    # Evaluate on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate comprehensive metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"\nBest Hyperparameters: {grid.best_params_}")
    print(f"Cross-Validation AUC: {grid.best_score_:.3f}")
    print(f"Test Set AUC: {auc_score:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    print("\n" + "="*60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:40s} {row['importance']:.4f}")

    # Save feature importance
    importance_dict = {
        'features': feature_importance.to_dict('records'),
        'auc_score': float(auc_score),
        'best_params': grid.best_params_
    }

    return best_model, grid.best_params_, importance_dict


def main():
    print("="*60)
    print("ADVANCED ML PIPELINE - CTA TRANSIT DELAY PREDICTION")
    print("="*60)

    df = load_data(DB_PATH, TABLE)
    X, y = preprocess(df)
    model, params, importance_dict = train_model(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save feature importance
    with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
