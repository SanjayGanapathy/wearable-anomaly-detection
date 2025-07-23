# tests/test_anomaly_model.py

import unittest
import pandas as pd
import numpy as np
import os
import sys

# This block adds the main project directory to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from anomaly_model import detect_anomalies


class TestAnomalyModel(unittest.TestCase):

    def setUp(self):
        """Create a sample DataFrame with a clear anomaly."""
        timestamps = pd.to_datetime(
            pd.date_range(start="2025-07-22", periods=100, freq="T")
        )
        # Create 99 normal heart rate values and one clear anomaly
        normal_hr = np.random.normal(loc=70, scale=5, size=99)
        anomaly_hr = [150]  # This is our obvious outlier
        data = {
            "heart_rate": np.concatenate((normal_hr, anomaly_hr)),
            "steps": np.random.randint(0, 30, 100),
            "hour": timestamps.hour,
            "hr_rolling_avg": np.random.normal(loc=70, scale=2, size=100),
            "hr_rolling_std": np.random.normal(loc=5, scale=1, size=100),
        }
        self.df = pd.DataFrame(data, index=timestamps)
        # Add other columns to simulate the full feature set
        for col in ["sleep_deep_minutes", "hrv_rmssd"]:
            self.df[col] = 0

    def test_detect_anomalies_heart_rate(self):
        """Test that the model can find a known heart rate anomaly."""
        # Use a contamination that is high enough to find our single outlier
        top_anomalies = detect_anomalies(
            df=self.df,
            features=["heart_rate", "steps", "hour", "hr_rolling_avg"],
            contamination=0.01,  # Expects 1% of data to be anomalies (1 out of 100)
            random_state=55,
            target="heart_rate",
        )

        self.assertEqual(top_anomalies.shape[0], 1, "Should find exactly one anomaly")
        # Check that the anomaly it found is the one we created
        self.assertEqual(top_anomalies["heart_rate"].iloc[0], 150)


if __name__ == "__main__":
    unittest.main()
