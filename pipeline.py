# pipeline.py

import config
from data_loader import load_data_range
from feature_engineering import create_features
from anomaly_model import detect_anomalies
from llm_explainer import get_anomaly_explanations


def run_pipeline(start_date: str, end_date: str, target_feature: str) -> dict:
    """
    Runs the full anomaly detection pipeline for a given date range and target.
    """
    try:
        df = load_data_range(
            config.BASE_PATH, config.SLEEP_PATH, config.HRV_PATH, start_date, end_date
        )

        df_featured = create_features(df, config.ROLLING_WINDOW_SIZE)

        top_anomalies = detect_anomalies(
            df_featured,
            config.FEATURES,
            config.ISOLATION_FOREST_CONTAMINATION,
            config.RANDOM_STATE,
            target_feature,
        )

        results = get_anomaly_explanations(
            top_anomalies, config.GOOGLE_API_KEY, target_feature
        )

        return {
            "status": "success",
            "date_range_analyzed": f"{start_date} to {end_date}",
            "results": results,
        }

    except FileNotFoundError as e:
        error_message = f"Data file not found: {e}."
        print(f"\n[ERROR] {error_message}")
        return {"status": "error", "message": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"status": "error", "message": error_message}
