# tests/test_feature_engineering.py

import unittest
import pandas as pd
import numpy as np
import os
import sys

# This block adds the main project directory to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from feature_engineering import create_features


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Create a sample DataFrame for testing."""
        # Create a simple time series DataFrame
        timestamps = pd.to_datetime(
            ["2025-07-22 12:00:00", "2025-07-22 12:00:01", "2025-07-22 12:00:02"]
        )
        data = {"heart_rate": [60, 62, 64]}
        self.df = pd.DataFrame(data, index=timestamps)

    def test_create_features(self):
        """Test that rolling average and standard deviation features are calculated correctly."""
        # We use a small window for predictable results in the test
        window_size = 2  # seconds

        featured_df = create_features(self.df, window_size)

        # --- Check Rolling Average ---
        # 1st row: avg of [60] = 60
        # 2nd row: avg of [60, 62] = 61
        # 3rd row: avg of [62, 64] = 63
        expected_avg = [60.0, 61.0, 63.0]
        # We use np.testing.assert_allclose for safe floating-point comparison
        np.testing.assert_allclose(featured_df["hr_rolling_avg"].tolist(), expected_avg)

        # --- Check Rolling Standard Deviation ---
        # 1st row: std of [60] = NaN -> filled with 0
        # 2nd row: std of [60, 62] = 1.414
        # 3rd row: std of [62, 64] = 1.414
        expected_std = [0.0, 1.41421356, 1.41421356]
        np.testing.assert_allclose(
            featured_df["hr_rolling_std"].tolist(), expected_std, rtol=1e-5
        )

        # --- Check Hour Feature ---
        self.assertEqual(featured_df["hour"].iloc[0], 12)


if __name__ == "__main__":
    unittest.main()
