# data_loader.py

import pandas as pd
import os
import sys
from datetime import timedelta


def load_questionnaire_data(questionnaire_path: str) -> dict:
    """Loads participant questionnaire data."""
    try:
        q_df = pd.read_csv(questionnaire_path)
        participant_data = q_df.iloc[0].to_dict()
        print(f"Loaded questionnaire data: {participant_data}")
        return participant_data
    except FileNotFoundError:
        print(f"Questionnaire file not found at {questionnaire_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error loading questionnaire data: {e}")
        return None


def load_and_summarize_sleep(sleep_file_path: str, target_date: pd.Timestamp) -> dict:
    """Loads and summarizes sleep data for the night prior to the target date."""
    try:
        sleep_df = pd.read_csv(sleep_file_path)
        sleep_df["startTime"] = pd.to_datetime(sleep_df["startTime"])
        sleep_df["endTime"] = pd.to_datetime(sleep_df["endTime"])
        night_sleep = sleep_df[sleep_df["endTime"].dt.date == target_date.date()]

        if night_sleep.empty:
            return None

        summary = {
            "sleep_deep_minutes": night_sleep[night_sleep["stage"] == "deep"][
                "duration"
            ].sum()
            / 60000,
            "sleep_light_minutes": night_sleep[night_sleep["stage"] == "light"][
                "duration"
            ].sum()
            / 60000,
            "sleep_rem_minutes": night_sleep[night_sleep["stage"] == "rem"][
                "duration"
            ].sum()
            / 60000,
            "sleep_awakenings": night_sleep[night_sleep["stage"] == "wake"].shape[0],
        }
        return summary
    except FileNotFoundError:
        return None


def load_daily_hrv(hrv_file_path: str) -> pd.DataFrame:
    """Loads and prepares the daily HRV data."""
    try:
        hrv_df = pd.read_csv(hrv_file_path)
        hrv_df["timestamp"] = pd.to_datetime(hrv_df["timestamp"])
        hrv_df["date"] = hrv_df["timestamp"].dt.date
        return hrv_df[["date", "rmssd", "coverage"]].rename(
            columns={"rmssd": "hrv_rmssd", "coverage": "hrv_coverage"}
        )
    except FileNotFoundError:
        return None


def load_data_range(
    base_path: str,
    sleep_path: str,
    hrv_path: str,
    questionnaire_path: str,
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    """
    Loads, merges, and cleans all data sources for a given date range.
    """
    print(f"Loading data from {start_date_str} to {end_date_str}...")

    all_dfs = []
    date_range = pd.to_datetime(pd.date_range(start=start_date_str, end=end_date_str))

    questionnaire_data = load_questionnaire_data(questionnaire_path)
    daily_hrv_data = load_daily_hrv(hrv_path)
    all_sleep_summaries = {}
    for date in date_range:
        sleep_summary = load_and_summarize_sleep(sleep_path, date)
        if sleep_summary:
            all_sleep_summaries[date.date()] = sleep_summary

    for date in date_range:
        current_date_str = date.strftime("%Y-%m-%d")

        hr_file = os.path.join(base_path, f"heart_rate_{current_date_str}.csv")
        if not os.path.exists(hr_file):
            continue

        hr_df = pd.read_csv(hr_file)
        hr_df.rename(columns={"beats per minute": "heart_rate"}, inplace=True)
        hr_df["timestamp"] = pd.to_datetime(hr_df["timestamp"])
        hr_df.set_index("timestamp", inplace=True)

        steps_month_str = date.strftime("%Y-%m-01")
        steps_file = os.path.join(base_path, f"steps_{steps_month_str}.csv")
        if not os.path.exists(steps_file):
            continue

        monthly_steps_df = pd.read_csv(steps_file)
        monthly_steps_df["timestamp"] = pd.to_datetime(monthly_steps_df["timestamp"])
        steps_df = monthly_steps_df[
            monthly_steps_df["timestamp"].dt.date == date.date()
        ].copy()
        steps_df.set_index("timestamp", inplace=True)
        steps_df.rename(columns={"value": "steps"}, inplace=True)

        daily_df = hr_df.join(steps_df, how="outer")
        daily_df.dropna(subset=["heart_rate"], inplace=True)

        daily_df["steps"] = daily_df["steps"].ffill().fillna(0)

        sleep_data = all_sleep_summaries.get(date.date(), {})
        for key, value in sleep_data.items():
            daily_df[key] = value

        if daily_hrv_data is not None:
            hrv_row = daily_hrv_data[daily_hrv_data["date"] == date.date()]
            if not hrv_row.empty:
                daily_df["hrv_rmssd"] = hrv_row["hrv_rmssd"].iloc[0]
                daily_df["hrv_coverage"] = hrv_row["hrv_coverage"].iloc[0]

        if questionnaire_data:
            for key, value in questionnaire_data.items():
                daily_df[key] = value

        all_dfs.append(daily_df)

    if not all_dfs:
        raise FileNotFoundError("No data could be loaded for the specified date range.")

    full_df = pd.concat(all_dfs).sort_index()

    expected_cols = {
        "sleep_deep_minutes": 0,
        "sleep_light_minutes": 0,
        "sleep_rem_minutes": 0,
        "sleep_awakenings": 0,
        "hrv_rmssd": 0,
        "hrv_coverage": 0,
        "primary_non_step_activity": "N/A",
        "caffeine_user": "N/A",
        "reports_high_stress": "N/A",
    }
    for col, default in expected_cols.items():
        if col not in full_df.columns:
            full_df[col] = default
        else:
            full_df[col].fillna(default, inplace=True)

    # --- ENCODING LOGIC ---
    print("Encoding questionnaire data for the model...")
    # One-Hot Encode categorical columns
    categorical_cols = [
        "primary_non_step_activity",
        "caffeine_user",
        "reports_high_stress",
    ]
    for col in categorical_cols:
        if col in full_df.columns:
            full_df = pd.get_dummies(full_df, columns=[col], prefix=col, dtype=float)

    full_df["heart_rate"] = full_df["heart_rate"].astype(int)
    full_df["steps"] = full_df["steps"].astype(int)

    print("Data loading and processing for range complete.")
    return full_df
