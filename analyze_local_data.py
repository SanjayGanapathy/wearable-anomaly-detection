import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import warnings
import google.generativeai as genai

# Suppress pandas FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- CONFIGURATION ---
TARGET_DATE = "2025-07-07"
GOOGLE_API_KEY = "AIzaSyAXeR1JhC9UUGf0D14iRiYNcYtQbTBWsVU"

# --- FILE PATHS ---
base_path = "Fitbit/Physical Activity_GoogleData/"
heart_rate_file = os.path.join(base_path, f"heart_rate_{TARGET_DATE}.csv")
steps_file = os.path.join(base_path, "steps-2025-05-01.csv")

try:
    # --- 1. DATA LOADING & PROCESSING ---
    print(f"Loading and processing data for {TARGET_DATE}...")
    # (The data loading code is the same as before)
    hr_df = pd.read_csv(heart_rate_file)
    hr_df.rename(columns={"beats per minute": "heart_rate"}, inplace=True)
    hr_df["timestamp"] = pd.to_datetime(hr_df["timestamp"])
    hr_df.set_index("timestamp", inplace=True)

    target_dt = pd.to_datetime(TARGET_DATE)
    steps_month_file = target_dt.strftime("%Y-%m-01")
    steps_file = os.path.join(base_path, f"steps_{steps_month_file}.csv")
    monthly_steps_df = pd.read_csv(steps_file)
    monthly_steps_df.rename(columns={"value": "steps"}, inplace=True)
    monthly_steps_df["timestamp"] = pd.to_datetime(monthly_steps_df["timestamp"])

    steps_df = monthly_steps_df[
        monthly_steps_df["timestamp"].dt.date == target_dt.date()
    ].copy()
    steps_df.set_index("timestamp", inplace=True)

    df = hr_df.join(steps_df, how="outer")
    df["steps"].fillna(method="ffill", inplace=True)
    df.dropna(subset=["heart_rate"], inplace=True)
    df["steps"].fillna(0, inplace=True)
    df["heart_rate"] = df["heart_rate"].astype(int)
    df["steps"] = df["steps"].astype(int)

    # --- 2. FEATURE ENGINEERING ---
    print("Creating features...")
    df["hour"] = df.index.hour
    df["hr_rolling_avg"] = df["heart_rate"].rolling(window=300, min_periods=1).mean()
    df.dropna(inplace=True)

    # --- 3. MODEL TRAINING & PREDICTION ---
    print("Training model and predicting anomalies...")
    features = ["heart_rate", "steps", "hour", "hr_rolling_avg"]
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df[features])
    df["anomaly"] = model.predict(df[features])

    anomalies = df[df["anomaly"] == -1].copy()

    # --- 4. FIND MOST SIGNIFICANT ANOMALIES ---
    # Calculate how much the heart rate deviated from the rolling average
    anomalies["hr_deviation"] = (
        anomalies["heart_rate"] - anomalies["hr_rolling_avg"]
    ).abs()
    # Sort by the biggest deviations and take the top 5
    top_5_anomalies = anomalies.sort_values(by="hr_deviation", ascending=False).head(5)

    print(
        f"\nFound {len(anomalies)} anomalies. Explaining the top 5 most significant..."
    )

    # --- 5. EXPLAIN ANOMALIES WITH LLM ---
    if not top_5_anomalies.empty and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel("gemini-2.5-flash")

        # Loop through the top 5 anomalies and explain each one
        for i, (timestamp, anomaly_row) in enumerate(top_5_anomalies.iterrows()):
            print(f"\n--- Explaining Anomaly #{i+1} ---")

            prompt = f"""
            You are a health data analyst for a medical researcher. Your task is to provide a detailed, multi-part explanation of heart rate anomalies from Fitbit wearable Data given by participants. This is medical data. Do not provide medical advice.

            **Anomaly Data:**
            - Timestamp: {timestamp}
            - Heart Rate: {anomaly_row['heart_rate']} bpm
            - Steps in the last minute: {anomaly_row['steps']}
            - Average Heart Rate (last 5 mins): {anomaly_row['hr_rolling_avg']:.1f} bpm

            **Analysis Required:**
            1.  **Summary:** Provide a one-sentence summary of the event.
            2.  **Reason for Flag:** Explain technically why this was flagged, focusing on the relationship between the measured heart rate, the rolling average, and physical activity (steps).
            3.  **Potential Correlations (Not Medical Advice):** Based on the data, what are some possible real-world events that could correlate with this type of anomaly? (e.g., sudden physical exertion, moments of stress or excitement, poor sleep, etc.).
            4.  **Key Observation for Researcher:** What is the most important or interesting takeaway from this specific data point for the researcher?
            """

            try:
                response = llm.generate_content(prompt)
                print(response.text)
            except Exception as e:
                print(f"Could not generate explanation. Error: {e}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
