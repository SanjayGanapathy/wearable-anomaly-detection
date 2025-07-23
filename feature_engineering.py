# feature_engineering.py

import pandas as pd


def create_features(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Engineers time-based, rolling average, and rolling standard deviation features.
    """
    print("Creating features...")

    df["hour"] = df.index.hour

    rolling_window = df["heart_rate"].rolling(window=f"{window_size}s", min_periods=1)
    df["hr_rolling_avg"] = rolling_window.mean()
    df["hr_rolling_std"] = rolling_window.std()

    df.dropna(subset=["hr_rolling_avg"], inplace=True)

    # FIX: Modernized the fillna call to remove the warning
    df["hr_rolling_std"] = df["hr_rolling_std"].fillna(0)

    print("Feature creation complete.")
    return df
