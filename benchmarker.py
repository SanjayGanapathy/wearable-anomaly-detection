# benchmarker.py

import pandas as pd
import time
import config
from data_loader import load_data_range
from feature_engineering import create_features

# Import the models we want to compare
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def run_benchmark(start_date: str, end_date: str):
    """
    Loads data and runs multiple anomaly detection models to compare their
    performance and results.
    """
    print("--- Starting Anomaly Detection Benchmark ---")

    # 1. Load and prepare data
    print("\nLoading data...")
    df = load_data_range(
        config.BASE_PATH, config.SLEEP_PATH, config.HRV_PATH, start_date, end_date
    )
    df_featured = create_features(df, config.ROLLING_WINDOW_SIZE)

    # Define the features to use for all models
    train_features = [f for f in config.FEATURES if f in df_featured.columns]
    X = df_featured[train_features]

    print(f"\nData prepared. Shape: {X.shape}. Running models...")
    print("-" * 50)

    # 2. Define the models to benchmark
    # We use a consistent contamination/nu value for a fair comparison
    models = {
        "Isolation Forest": IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=config.RANDOM_STATE,
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20, contamination=config.ISOLATION_FOREST_CONTAMINATION
        ),
        "One-Class SVM": OneClassSVM(
            nu=config.ISOLATION_FOREST_CONTAMINATION, kernel="rbf", gamma="auto"
        ),
    }

    # 3. Run and time each model
    for name, model in models.items():
        print(f"Running {name}...")
        start_time = time.time()

        # --- FIX ---
        # We now fit every model. For these models, fit_predict is often used,
        # but we use fit() and predict() separately to time the prediction step.
        if name == "Local Outlier Factor":
            # LOF's predict method is only available after fitting.
            # It doesn't use the 'predict' method for outlier detection in the same way.
            # We use fit_predict to get the labels.
            predictions = model.fit_predict(X)
        else:
            # For Isolation Forest and One-Class SVM, we fit then predict.
            model.fit(X)
            predictions = model.predict(X)
        # --- END FIX ---

        end_time = time.time()

        # Count the number of anomalies found
        num_anomalies = (predictions == -1).sum()

        print(f"  -> Found {num_anomalies} anomalies.")
        print(f"  -> Time taken: {end_time - start_time:.2f} seconds.")
        print("-" * 50)

    print("Benchmark complete.")


if __name__ == "__main__":
    run_benchmark(config.START_DATE, config.END_DATE)
