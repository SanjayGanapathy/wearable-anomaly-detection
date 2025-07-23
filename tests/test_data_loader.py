# tests/test_data_loader.py

import unittest
import pandas as pd
import os
import sys

# This block adds the main project directory to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from data_loader import load_and_summarize_sleep, load_daily_hrv, load_data_range


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Set up paths for all test data files."""
        self.test_data_path = os.path.join(os.path.dirname(__file__), "sample_data")
        self.sleep_path = os.path.join(self.test_data_path, "sleep-stages-2025.csv")
        self.hrv_path = os.path.join(
            self.test_data_path, "daily_heart_rate_variability_summary.csv"
        )
        # --- FIX: Add path for the test questionnaire ---
        self.questionnaire_path = os.path.join(self.test_data_path, "questionnaire.csv")

    def test_load_and_summarize_sleep(self):
        """Test that sleep data is summarized correctly."""
        target_date = pd.to_datetime("2025-07-01")
        summary = load_and_summarize_sleep(self.sleep_path, target_date)

        self.assertIsNotNone(summary, "Sleep summary should not be None")
        self.assertEqual(summary["sleep_deep_minutes"], 60)
        self.assertEqual(summary["sleep_awakenings"], 1)

    def test_load_daily_hrv(self):
        """Test that HRV data is loaded correctly."""
        hrv_df = load_daily_hrv(self.hrv_path)

        self.assertIsNotNone(hrv_df, "HRV DataFrame should not be None")
        self.assertEqual(hrv_df.shape[0], 1)
        self.assertEqual(hrv_df["hrv_rmssd"].iloc[0], 55.5)

    def test_load_data_range(self):
        """Test the main data loading function with all data sources."""
        start_date = "2025-07-01"
        end_date = "2025-07-01"

        # --- FIX: Pass the new questionnaire_path argument ---
        full_df = load_data_range(
            self.test_data_path,
            self.sleep_path,
            self.hrv_path,
            self.questionnaire_path,  # Added this argument
            start_date,
            end_date,
        )

        self.assertIsInstance(full_df, pd.DataFrame)
        self.assertEqual(full_df.shape[0], 2)
        # Check that questionnaire data was merged (it will be encoded)
        self.assertIn("primary_non_step_activity_stationary_bike", full_df.columns)
        self.assertEqual(
            full_df["primary_non_step_activity_stationary_bike"].iloc[0], 1.0
        )


if __name__ == "__main__":
    unittest.main()
