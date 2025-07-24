# models.py

import pandas as pd
import numpy as np


def run_deterministic_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs a simple, rule-based (deterministic) anomaly detection model.

    An anomaly is flagged if any of the following rules are met:
    1. Absolute Threshold: Heart rate exceeds a defined maximum.
    2. Rapid Gradient: Heart rate increases too quickly in a short period.
    3. Statistical Deviation: Heart rate is a certain number of standard
       deviations away from the daily mean.

    Args:
        df: The input DataFrame with a 'heart_rate' column and a datetime index.

    Returns:
        A DataFrame with an added 'anomaly' column (-1 for anomalies, 1 for inliers).
    """
    print("Running Deterministic (Rule-Based) Model...")

    # --- Rule 1: Absolute Threshold ---
    # Anything above 185 bpm is flagged as an anomaly.
    rule1_threshold = 185
    df["anomaly_rule1"] = df["heart_rate"] > rule1_threshold

    # --- Rule 2: Rapid Gradient (Sudden Increase) ---
    # Calculate the change in heart rate from the previous data point
    hr_diff = df["heart_rate"].diff()
    # Calculate the time difference in seconds
    time_diff = df.index.to_series().diff().dt.total_seconds()
    # Calculate the gradient (change per second)
    hr_gradient = hr_diff / time_diff
    # Flag if the heart rate increases by more than 0.5 bpm per second (30 bpm per minute)
    gradient_threshold = 0.5
    df["anomaly_rule2"] = hr_gradient > gradient_threshold

    # --- Rule 3: Statistical Deviation from Daily Mean ---
    # Calculate the mean and standard deviation for each day
    daily_stats = df.groupby(df.index.date)["heart_rate"].agg(["mean", "std"])
    daily_stats.rename(columns={"mean": "daily_mean", "std": "daily_std"}, inplace=True)

    # Map the daily stats back to each row in the main dataframe
    df_with_stats = df.join(daily_stats, on=df.index.date)

    # Flag if heart rate is more than 3 standard deviations from the daily mean
    deviation_threshold = 3
    df["anomaly_rule3"] = (
        df_with_stats["heart_rate"] - df_with_stats["daily_mean"]
    ).abs() > (deviation_threshold * df_with_stats["daily_std"])

    # --- Combine the rules ---
    # An anomaly is flagged if ANY of the rules are true
    anomaly_mask = df["anomaly_rule1"] | df["anomaly_rule2"] | df["anomaly_rule3"]

    # Convert boolean mask to our standard format (-1 for anomaly, 1 for inlier)
    df["anomaly"] = np.where(anomaly_mask, -1, 1)

    # Clean up intermediate columns
    df.drop(columns=["anomaly_rule1", "anomaly_rule2", "anomaly_rule3"], inplace=True)

    print(f"  -> Found {np.sum(df['anomaly'] == -1)} anomalies based on the rules.")
    return df
