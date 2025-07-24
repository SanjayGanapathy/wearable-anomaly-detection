# evaluation.py

import pandas as pd
import time
import os
import config
from data_loader import load_data_range
from feature_engineering import create_features
from sklearn.ensemble import IsolationForest

# Import our new deterministic model
from models import run_deterministic_model


def run_evaluation(start_date: str, end_date: str):
    """
    Runs a comprehensive evaluation of multiple anomaly detection models:
    1. Deterministic (Rule-Based)
    2. Simple Isolation Forest (ML)
    3. Complex Isolation Forest (ML with all features)

    The anomalies found by each model are saved to separate CSV files.
    """
    print("--- Starting Full Model Evaluation ---")

    # 1. Load and prepare the full dataset once
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
    print("-" * 50)

    # --- Model 1: Deterministic (Rule-Based) Model ---
    print("Running: 1. Deterministic Model")
    start_time = time.time()

    deterministic_results_df = run_deterministic_model(df_featured.copy())
    deterministic_anomalies = deterministic_results_df[
        deterministic_results_df["anomaly"] == -1
    ]

    end_time = time.time()

    output_file_1 = "deterministic_anomalies.csv"
    deterministic_anomalies.to_csv(output_file_1)

    print(f"  -> Found {len(deterministic_anomalies)} anomalies.")
    print(f"  -> Results saved to '{output_file_1}'")
    print(f"  -> Time taken: {end_time - start_time:.2f} seconds.")
    print("-" * 50)

    # --- Machine Learning Model Setups ---
    ml_models_to_test = {
        "Simple_Isolation_Forest": {
            "features": ["heart_rate", "hour"],
            "output_file": "simple_ml_model_anomalies.csv",
        },
        "Complex_Isolation_Forest": {
            "features": [f for f in config.FEATURES if f in df_featured.columns],
            "output_file": "complex_ml_model_anomalies.csv",
        },
    }

    # --- Run ML Models ---
    model_number = 2
    for name, model_config in ml_models_to_test.items():
        print(f"Running: {model_number}. {name}")

        features = model_config["features"]
        output_file = model_config["output_file"]

        X = df_featured[features].select_dtypes(include="number")
        print(f"  -> Using {len(features)} features. Data shape: {X.shape}")

        model = IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=config.RANDOM_STATE,
        )

        start_time = time.time()

        predictions = model.fit_predict(X)

        end_time = time.time()

        anomalies_df = df_featured[predictions == -1]
        anomalies_df.to_csv(output_file)

        print(f"  -> Found {len(anomalies_df)} anomalies.")
        print(f"  -> Results saved to '{output_file}'")
        print(f"  -> Time taken: {end_time - start_time:.2f} seconds.")
        print("-" * 50)
        model_number += 1

    print("Full evaluation complete.")
    print(
        "You now have three CSV files with anomalies from each model, ready for the next phase of analysis."
    )


if __name__ == "__main__":
    run_evaluation(config.START_DATE, config.END_DATE)
