# tuner.py

import warnings
import numpy as np
import config
from data_loader import load_data
from feature_engineering import create_features
from anomaly_model import detect_anomalies

# Suppress pandas FutureWarnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)


def tune_contamination():
    """
    Runs the anomaly detection pipeline with various contamination settings
    to help researchers choose the best value.
    """
    print(f"--- Starting Hyperparameter Tuning for {config.TARGET_DATE} ---")

    try:
        # Load and prepare the data once to save time
        df = load_data(config.BASE_PATH, config.TARGET_DATE)
        df_featured = create_features(df, config.ROLLING_WINDOW_SIZE)

        # Define the range of "sensitivity" settings to test
        # These values represent the percentage of data expected to be anomalous
        contamination_levels = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

        print("\nRunning analysis for different contamination values...")
        print("-" * 50)

        # Run the model for each setting and print the results
        for level in contamination_levels:
            # We only need to re-run the anomaly detection part
            anomalies = detect_anomalies(
                df_featured.copy(),  # Use a copy to avoid modifying the original df
                config.FEATURES,
                contamination=level,
                random_state=config.RANDOM_STATE,
            )

            # The detect_anomalies function returns the top 5, but the original
            # number of anomalies is what we want to report here.
            # To get the total count, we'll re-run a simplified version.
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                contamination=level, random_state=config.RANDOM_STATE
            )
            model.fit(df_featured[config.FEATURES])
            predictions = model.predict(df_featured[config.FEATURES])
            total_anomalies = np.sum(predictions == -1)

            print(
                f"Contamination: {level:<6} (or {level*100:.1f}%) -> Found {total_anomalies} anomalies."
            )

        print("-" * 50)
        print("\nTuning process complete.")
        print(
            "Please review the counts above to select the best sensitivity for your study."
        )

    except FileNotFoundError as e:
        print(f"\n[ERROR] Data file not found: {e}. Please check your config.py.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    tune_contamination()