import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier # type: ignore
import joblib

DB_PATH = "smart_transit.db"
TABLE = "labeled_with_spatial"  # Updated to use spatial features
MODEL_PATH = "xgb_model.pkl"


def load_data(db_path: str, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess data with spatial features for delay prediction"""
    df = df[df["arrival_time"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)]
    df["hour"] = df["arrival_time"].str.slice(0, 2).astype(int) % 24  # Handle 24+ hour times
    df = df[df["delayed"].isin([0, 1])]
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")

    # Use timestamp to get real day of week if available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    else:
        # Fallback: distribute across week based on row index
        df["day_of_week"] = df.index % 7

    # One-hot encode categorical weather conditions
    if "conditions" in df.columns:
        df = pd.get_dummies(df, columns=["conditions"], drop_first=True)

    # One-hot encode route_type if available
    if "route_type" in df.columns:
        df = pd.get_dummies(df, columns=["route_type"], drop_first=True, prefix="route_type")

    # Limit stop_id to top 50 busiest stops and one-hot encode
    if "stop_id" in df.columns:
        top_stops = df["stop_id"].value_counts().nlargest(50).index
        df = df[df["stop_id"].isin(top_stops)]
        df = pd.get_dummies(df, columns=["stop_id"], drop_first=True)

    # Drop non-feature columns
    drop_cols = [
        "arrival_time",
        "departure_time",
        "trip_id",
        "delayed",
        "stop_headsign",
        "pickup_type",
        "shape_dist_traveled",
        "icon",
        "timestamp",
        "stop_lat",
        "stop_lon",
        "route_id",
        "delay_minutes",  # Don't use actual delay as feature, only the binary label
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["delayed"]

    print(f"\n=== Feature Engineering Summary ===")
    print(f"Total samples after preprocessing: {len(X)}")
    print(f"Total features: {len(X.columns)}")
    spatial_features = [c for c in X.columns if c in ["distance_from_loop", "segment_length",
                                                        "is_transfer_hub", "upstream_delay"]]
    print(f"Spatial features: {spatial_features}")
    print(f"Delay rate: {y.mean():.1%}\n")

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[XGBClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.3],
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)

    y_pred = grid.predict(X_test)
    print("Best params:", grid.best_params_)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return grid.best_estimator_, grid.best_params_


def main():
    df = load_data(DB_PATH, TABLE)
    X, y = preprocess(df)
    model, params = train_model(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
