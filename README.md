# CTA Transit Delay Prediction

A geospatial machine learning pipeline that predicts transit delays for the Chicago Transit Authority (CTA) using GTFS data, weather enrichment, and spatial propagation modeling. Demonstrates spatial feature engineering and interactive web mapping.

**[View Live Demo →](https://ibiggs24.github.io/TransitPredictor/)**

## Key Features

### Spatial ML Pipeline
- **Spatial features**: distance from Loop, upstream delay propagation, segment length, transfer hub detection
- **Route-level modeling**: Delays propagate along trips based on prior stops
- **Weather integration**: Hourly weather data from Visual Crossing API
- **XGBoost with SMOTE**: Handles class imbalance for better delay detection

### Interactive Visualization
- **MapLibre web map** with hour-by-hour delay predictions
- **GeoJSON output** compatible with ArcGIS Online, QGIS, and web mapping tools
- **11,000+ CTA stops** with lat/lon coordinates

### Model Performance
- **Delay detection recall**: 25% (vs 6% without spatial features)
- **Delay detection F1-score**: 0.29 (vs 0.06 baseline)
- **Overall accuracy**: 76%
- **59 features** including distance from Loop, upstream delay, segment length, and transfer hub status

> **Note**: This project uses **simulated delays** with realistic spatial propagation, not actual CTA data. The simulation incorporates rush hour patterns, weather impacts, and route-level delay persistence to demonstrate spatial ML techniques.

## Technologies

- **ML**: XGBoost, scikit-learn, imbalanced-learn
- **Geospatial**: geopy, GeoJSON
- **Data**: pandas, SQLite, GTFS
- **Visualization**: MapLibre GL JS
- **APIs**: Visual Crossing Weather API

## Setup Instructions

### 1. Clone and setup environment

```bash
git clone https://github.com/ibiggs24/TransitPredictor.git
cd TransitPredictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download CTA GTFS data

```bash
curl -L "https://www.transitchicago.com/downloads/sch_data/google_transit.zip" -o /tmp/google_transit.zip
unzip /tmp/google_transit.zip -d data/google_transit/
```

### 3. Set up Visual Crossing API key

Create a `.env` file in the root directory:

```env
VC_API_KEY=your_api_key_here
```

Get a free API key at [Visual Crossing](https://www.visualcrossing.com/weather-api).

### 4. Run the pipeline

```bash
# Load GTFS into SQLite (use sample version for faster testing)
python scripts/load_gtfs_sample.py

# Simulate delays with spatial propagation
python scripts/label_delays.py

# Enrich with weather data
python scripts/join_weather_features.py

# Add spatial features (distance from Loop, upstream delay, etc.)
python scripts/add_spatial_features.py

# Train XGBoost model with spatial features
python scripts/train_xgboost_gridsearch.py

# Generate GeoJSON predictions
python scripts/generate_geojson.py
```

### 5. View the map

Open `docs/index.html` in your browser or deploy to GitHub Pages:

1. Go to Settings > Pages in your GitHub repo
2. Select "Deploy from a branch"
3. Choose `main` branch and `/docs` folder
4. Visit `https://yourusername.github.io/TransitPredictor/`

## Pipeline Architecture

```
GTFS Data → SQLite → Delay Simulation → Weather Enrichment → Spatial Features → XGBoost → GeoJSON → MapLibre Map
```

**Spatial features added:**
- `distance_from_loop`: Distance in km from downtown Chicago (41.8781°N, 87.6298°W)
- `upstream_delay`: Cumulative delay minutes from previous stops on the same trip
- `segment_length`: Distance between consecutive stops
- `is_transfer_hub`: Binary flag for stops with transfers
- `route_type`: Bus vs rail classification

## Project Structure

```
TransitPredictor/
├── scripts/
│   ├── load_gtfs_sample.py       # Load GTFS into SQLite
│   ├── label_delays.py            # Simulate delays with propagation
│   ├── join_weather_features.py   # Add weather data
│   ├── add_spatial_features.py    # Calculate spatial features
│   ├── train_xgboost_gridsearch.py # Train model
│   └── generate_geojson.py        # Create predictions
├── docs/
│   ├── index.html                 # MapLibre web map
│   └── predictions.geojson        # Prediction output
├── data/google_transit/           # GTFS files (gitignored)
├── smart_transit.db               # SQLite database (gitignored)
└── requirements.txt
```

## Author

**Isaac Biggs**
CS + Geography/GIS, Data Science Minor
University of Illinois Urbana-Champaign

Data sources: CTA GTFS Feed, Visual Crossing Weather API
