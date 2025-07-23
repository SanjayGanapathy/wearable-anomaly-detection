# compare_anomalies.py

import pandas as pd
import config
import google.generativeai as genai


def generate_comparison_report():
    """
    Loads anomalies from the A/B test, selects examples, and uses an LLM
    to generate a report comparing the quality of the findings.
    """
    print("--- Generating LLM Comparison Report for A/B Test ---")

    try:
        # 1. Load the results from the benchmark
        simple_anomalies = pd.read_csv("simple_model_anomalies.csv")
        complex_anomalies = pd.read_csv("complex_model_anomalies.csv")
    except FileNotFoundError:
        print("\n[ERROR] CSV files not found. Please run 'benchmarker.py' first.")
        return

    # 2. Find an anomaly that is unique to the complex model
    # We can do this by finding timestamps that exist in the complex set but not the simple one
    simple_timestamps = set(simple_anomalies["timestamp"])
    unique_complex_anomalies = complex_anomalies[
        ~complex_anomalies["timestamp"].isin(simple_timestamps)
    ]

    if unique_complex_anomalies.empty:
        print(
            "\nNo unique anomalies found in the complex model to compare. Try adjusting model parameters."
        )
        return

    # Select one interesting example from each set
    complex_example = unique_complex_anomalies.iloc[0]
    simple_example = simple_anomalies.iloc[0]

    print("\nSelected examples for comparison:")
    print("  -> Simple Model Example:", simple_example["timestamp"])
    print("  -> Complex Model Example:", complex_example["timestamp"])

    # 3. Configure the LLM
    if (
        not config.GOOGLE_API_KEY
        or config.GOOGLE_API_KEY == "REPLACE_WITH_YOUR_GOOGLE_API_KEY"
    ):
        print("\n[ERROR] Google API key not set in config.py. Cannot generate report.")
        return

    genai.configure(api_key=config.GOOGLE_API_KEY)
    llm = genai.GenerativeModel("gemini-1.5-flash")

    # 4. Create a detailed prompt for the comparison
    prompt = f"""
    You are a senior data scientist writing a report for a researcher as a part of A/B testing.
    Your goal is to explain how the complex and simple anomalies detected compare.

    You have two examples of anomalies found in a participant's Fitbit data.

    ---
    **Example 1: Found by the SIMPLE Model**
    This model only used 'heart_rate' and 'hour' as features.

    **Anomaly Data:**
    - Timestamp: {simple_example['timestamp']}
    - Heart Rate: {simple_example['heart_rate']} bpm
    - Steps (last minute): {simple_example['steps']}
    - Average Heart Rate (last 5 mins): {simple_example['hr_rolling_avg']:.1f} bpm
    - Previous Night's Deep Sleep: {simple_example.get('sleep_deep_minutes', 'N/A'):.0f} minutes
    - Daily HRV (RMSSD): {simple_example.get('hrv_rmssd', 'N/A')} ms
    ---
    **Example 2: Found ONLY by the COMPLEX Model**
    This model used all available features (heart rate, steps, sleep, HRV, etc.) to provide context.

    **Anomaly Data:**
    - Timestamp: {complex_example['timestamp']}
    - Heart Rate: {complex_example['heart_rate']} bpm
    - Steps (last minute): {complex_example['steps']}
    - Average Heart Rate (last 5 mins): {complex_example['hr_rolling_avg']:.1f} bpm
    - Previous Night's Deep Sleep: {complex_example.get('sleep_deep_minutes', 'N/A'):.0f} minutes
    - Daily HRV (RMSSD): {complex_example.get('hrv_rmssd', 'N/A')} ms
    ---

    **Report Required:**

    Write a brief report that addresses the following:
    1.  **Analyze the Simple Model's Finding:** Briefly describe the anomaly found by the simple model. Is it interesting or notable or show anything unique about the data?
    2.  **Analyze the Complex Model's Finding:** Describe the unique anomaly found by the complex model. Explain why this finding (if it exists) is different in comparison to the simple model's detected anomalies.
    3.  **Conclusion:** Conclude your thoughts on the two types of models."""

    # 5. Generate and print the report
    print("\nGenerating report...")
    try:
        response = llm.generate_content(prompt)
        print("\n--- A/B Test Qualitative Report ---")
        print(response.text)
        print("-----------------------------------")
    except Exception as e:
        print(f"\n[ERROR] Could not generate report. Error: {e}")


if __name__ == "__main__":
    generate_comparison_report()
