# benchmarker.py

import pandas as pd
import time
import os
import config
from data_loader import load_data_range
from feature_engineering import create_features
from sklearn.ensemble import IsolationForest


def run_ab_test(start_date: str, end_date: str):
    """
    Performs an A/B test between a simple and a complex model and saves
    the detected anomalies to CSV files for qualitative review.
    """
    print("--- Starting Model A/B Test ---")

    # 1. Load and prepare data
    print("\nLoading and preparing data...")
    df = load_data_range(
        config.BASE_PATH,
        config.SLEEP_PATH,
        config.HRV_PATH,
        config.QUESTIONNAIRE_PATH,
        start_date,
        end_date,
    )
    df_featured = create_features(df, config.ROLLING_WINDOW_SIZE)
    print("Data preparation complete.")

    # 2. Define the two model configurations
    models_to_test = {
        "Simple_Model": {
            "features": ["heart_rate", "hour"],
            "output_file": "simple_model_anomalies.csv",
        },
        "Complex_Model": {
            "features": [f for f in config.FEATURES if f in df_featured.columns],
            "output_file": "complex_model_anomalies.csv",
        },
    }

    print("\nRunning A/B test...")
    print("-" * 50)

    # 3. Run each model and save its results
    for name, model_config in models_to_test.items():
        print(f"Testing: {name}")

        features = model_config["features"]
        output_file = model_config["output_file"]

        X = df_featured[features].select_dtypes(include="number")
        print(f"  -> Using {len(features)} features. Data shape: {X.shape}")

        model = IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=config.RANDOM_STATE,
        )

        start_time = time.time()
        model.fit(X)
        predictions = model.predict(X)
        end_time = time.time()

        # Isolate the anomalies and save them
        anomalies_df = df_featured[predictions == -1]
        anomalies_df.to_csv(output_file)

        num_anomalies = len(anomalies_df)
        print(f"  -> Found {num_anomalies} anomalies.")
        print(f"  -> Results saved to '{output_file}'")
        print(f"  -> Time taken: {end_time - start_time:.2f} seconds.")
        print("-" * 50)

    print("A/B Test complete.")
    print(
        "\nReview the CSV files to compare the quality of anomalies found by each model."
    )


if __name__ == "__main__":
    run_ab_test(config.START_DATE, config.END_DATE)
