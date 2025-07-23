# anomaly_model.py

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    features: list,
    contamination: float,
    random_state: int,
    target: str,
) -> pd.DataFrame:
    """
    Trains an IsolationForest model and identifies the top 5 anomalies
    ranked by the specified target feature.

    Args:
        df: The DataFrame with features.
        features: A list of feature names for the model.
        contamination: The proportion of anomalies in the data.
        random_state: The random state for reproducibility.
        target: The feature to focus on for ranking anomalies (e.g., 'heart_rate').

    Returns:
        A DataFrame containing the top 5 most significant anomalies.
    """
    print(f"Training model and predicting anomalies, ranking by '{target}'...")

    model = IsolationForest(contamination=contamination, random_state=random_state)
    train_features = [f for f in features if f in df.columns]

    model.fit(df[train_features])
    df["anomaly"] = model.predict(df[train_features])

    anomalies = df[df["anomaly"] == -1].copy()

    # Dynamic ranking based on the target feature
    if target == "heart_rate":
        # For heart rate, we use the Z-score we already engineered
        anomalies["z_score"] = (
            (anomalies["heart_rate"] - anomalies["hr_rolling_avg"])
            / anomalies["hr_rolling_std"]
        ).fillna(0)
        sort_key = "z_score"
    else:
        # For other features (like 'steps'), we rank by absolute deviation from the overall mean
        print(
            f"Note: Ranking '{target}' by deviation from the overall mean. For Z-score, engineer rolling stats for this feature."
        )
        overall_mean = df[target].mean()
        anomalies[f"{target}_deviation"] = (anomalies[target] - overall_mean).abs()
        sort_key = f"{target}_deviation"

    # Sort by the absolute value of our chosen ranking metric
    top_5_anomalies = anomalies.sort_values(by=sort_key, ascending=False, key=abs).head(
        5
    )

    print(
        f"Found {len(anomalies)} total anomalies. Focusing on the top 5 by '{target}'."
    )
    return top_5_anomalies
