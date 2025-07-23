# llm_explainer.py

import pandas as pd
import google.generativeai as genai


def get_anomaly_explanations(
    anomalies_df: pd.DataFrame, api_key: str, target_feature: str
) -> list:
    """
    Uses Google's Gemini LLM to generate explanations for each anomaly.
    """
    explanations = []
    if anomalies_df.empty:
        print("No anomalies to explain.")
        return explanations

    if not api_key or api_key == "REPLACE_WITH_YOUR_GOOGLE_API_KEY":
        print("\nSkipping LLM explanation: Google API key not provided.")
        for timestamp, row in anomalies_df.iterrows():
            anomaly_data = row.to_dict()
            anomaly_data["timestamp"] = timestamp.isoformat()
            explanations.append(
                {
                    "anomaly_data": anomaly_data,
                    "explanation": "Skipped: No Google API Key provided in config.py.",
                }
            )
        return explanations

    print("\n--- Generating LLM Explanations ---")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-1.5-flash")

    for i, (timestamp, row) in enumerate(anomalies_df.iterrows()):
        print(f"Explaining Anomaly #{i+1} at {timestamp}...")

        prompt = f"""
        You are a health data analyst. Your task is to explain a data anomaly from Fitbit data.
        The anomaly was selected because it was one of the most unusual data points for the **'{target_feature}'** metric.
        Analyze all the provided context to explain why this might have happened. Do not provide medical advice.

        **Daily Physiological Context:**
        - Previous Night's Deep Sleep: {row.get('sleep_deep_minutes', 'N/A'):.0f} minutes
        - Previous Night's REM Sleep: {row.get('sleep_rem_minutes', 'N/A'):.0f} minutes
        - Awakenings: {row.get('sleep_awakenings', 'N/A')} times
        - Daily HRV (RMSSD): {row.get('hrv_rmssd', 'N/A')} ms

        **Anomaly Data:**
        - Timestamp: {timestamp}
        - Heart Rate: {row['heart_rate']} bpm
        - Steps (last minute): {row['steps']}
        - Average Heart Rate (last 5 mins): {row['hr_rolling_avg']:.1f} bpm
        - Statistical Significance (Z-score for HR): {row.get('z_score', 'N/A'):.2f}

        **Analysis Required:**
        1.  **Summary:** Provide a one-sentence summary of the event, mentioning the target feature ('{target_feature}').
        2.  **Reason for Flag:** Explain why this was flagged, focusing on the '{target_feature}' value in the context of the other data.
        3.  **Potential Correlations:** Based on ALL provided data, what are possible real-world correlations?
        4.  **Key Observation for Researcher:** What is the most important takeaway, especially concerning the '{target_feature}'?
        """

        try:
            response = llm.generate_content(prompt)
            explanation_text = response.text
        except Exception as e:
            explanation_text = f"Could not generate explanation. Error: {e}"

        anomaly_data = row.to_dict()
        anomaly_data["timestamp"] = timestamp.isoformat()

        explanations.append(
            {"anomaly_data": anomaly_data, "explanation": explanation_text}
        )

    return explanations
