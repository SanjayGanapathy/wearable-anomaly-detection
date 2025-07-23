# tests/test_data_loader.py

import unittest
import pandas as pd
import os
import sys

# This block adds the main project directory to Python's path.
# This ensures that the test script can successfully import our modules like `data_loader`.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from data_loader import load_and_summarize_sleep, load_daily_hrv, load_data_range


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Set up paths for test data before each test."""
        self.test_data_path = os.path.join(os.path.dirname(__file__), "sample_data")
        self.sleep_path = os.path.join(self.test_data_path, "sleep-stages-2025.csv")
        self.hrv_path = os.path.join(
            self.test_data_path, "daily_heart_rate_variability_summary.csv"
        )

    def test_load_and_summarize_sleep(self):
        """Test that sleep data is summarized correctly."""
        target_date = pd.to_datetime("2025-07-01")
        summary = load_and_summarize_sleep(self.sleep_path, target_date)

        self.assertIsNotNone(summary, "Sleep summary should not be None")
        self.assertEqual(summary["sleep_deep_minutes"], 60)  # 3600000 ms = 60 mins
        self.assertEqual(summary["sleep_awakenings"], 1)

    def test_load_daily_hrv(self):
        """Test that HRV data is loaded correctly."""
        hrv_df = load_daily_hrv(self.hrv_path)

        self.assertIsNotNone(hrv_df, "HRV DataFrame should not be None")
        self.assertEqual(hrv_df.shape[0], 1)
        self.assertEqual(hrv_df["hrv_rmssd"].iloc[0], 55.5)

    def test_load_data_range(self):
        """Test the main data loading function for a range."""
        start_date = "2025-07-01"
        end_date = "2025-07-01"

        full_df = load_data_range(
            self.test_data_path, self.sleep_path, self.hrv_path, start_date, end_date
        )

        self.assertIsInstance(full_df, pd.DataFrame)
        self.assertEqual(full_df.shape[0], 2)  # Expect 2 rows from the heart rate file
        self.assertEqual(full_df["sleep_deep_minutes"].iloc[0], 60)
        self.assertEqual(full_df["hrv_rmssd"].iloc[0], 55.5)
        self.assertEqual(full_df["steps"].iloc[0], 10)


if __name__ == "__main__":
    unittest.main()
