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
    """
    print(f"Training model and predicting anomalies, ranking by '{target}'...")

    model = IsolationForest(contamination=contamination, random_state=random_state)

    # Ensure only columns that actually exist in the dataframe are used
    train_features = [f for f in features if f in df.columns]

    # Select only numeric data for the model
    numeric_df = df[train_features].select_dtypes(include="number")

    model.fit(numeric_df)
    df["anomaly"] = model.predict(numeric_df)

    anomalies = df[df["anomaly"] == -1].copy()

    if target == "heart_rate":
        anomalies["z_score"] = (
            (anomalies["heart_rate"] - anomalies["hr_rolling_avg"])
            / anomalies["hr_rolling_std"]
        ).fillna(0)
        sort_key = "z_score"
    else:
        print(f"Note: Ranking '{target}' by deviation from the overall mean.")
        overall_mean = df[target].mean()
        anomalies[f"{target}_deviation"] = (anomalies[target] - overall_mean).abs()
        sort_key = f"{target}_deviation"

    top_5_anomalies = anomalies.sort_values(by=sort_key, ascending=False, key=abs).head(
        5
    )

    print(
        f"Found {len(anomalies)} total anomalies. Focusing on the top 5 by '{target}'."
    )
    return top_5_anomalies
