# Wearable Anomaly Detection

This project analyzes intraday time-series data from Fitbit wearables to detect and explain heart rate anomalies. It uses an `IsolationForest` machine learning model to identify unusual patterns and leverages Google's Gemini Large Language Model (LLM) to translate these findings into human-readable explanations suitable for non-technical researchers.

## Key Features

-   **Parses Local Data**: Ingests and processes high-resolution heart rate and steps data from a standard Fitbit data export.
-   **Feature Engineering**: Creates contextual features, such as the hour of the day and rolling heart rate averages, to improve model accuracy.
-   **Anomaly Detection**: Trains an `IsolationForest` model to learn a user's normal physiological patterns and flags significant deviations.
-   **Prioritized Results**: Identifies and ranks the top 5 most significant anomalies based on their deviation from the user's recent average heart rate.
-   **LLM-Powered Explanations**: For each top anomaly, the system generates a detailed, multi-part analysis that includes:
    1.  A one-sentence summary of the event.
    2.  The technical reason it was flagged.
    3.  Potential real-world correlations (without giving medical advice).
    4.  A key takeaway for the researcher.

## Tech Stack

-   **Python**
-   **Pandas**: For data manipulation and analysis.
-   **Scikit-learn**: For the Isolation Forest model.
-   **Google Generative AI**: For generating natural language explanations with the Gemini model.

## Setup and Installation

### 1. Download Your Fitbit Data
This script is designed to work with a local data export from Fitbit.
-   Go to [Fitbit's Data Export page](https://www.fitbit.com/settings/data/export).
-   Request your data in **JSON** format.
-   Download and unzip the `Takeout` folder.

### 2. Project Structure
Place the `analyze_local_data.py` script inside the unzipped `Takeout` folder. The script expects the following directory structure:
```
Takeout/
├── Fitbit/
│   └── Physical Activity_GoogleData/
│       ├── heart_rate_YYYY-MM-DD.csv
│       ├── steps_YYYY-MM-01.csv
│       └── ... (other data files)
└── analyze_local_data.py
```

### 3. Install Dependencies
Open your terminal and install the required Python libraries:
```bash
pip install pandas scikit-learn google-generativeai
```

## Usage

1.  **Configure the Script**: Open the `analyze_local_data.py` file and update the configuration section at the top:
    -   `TARGET_DATE`: Set the date you wish to analyze in `"YYYY-MM-DD"` format.
    -   `GOOGLE_API_KEY`: Paste your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

2.  **Run the Analysis**: Navigate to the `Takeout` directory in your terminal and run the script:
    ```bash
    python analyze_local_data.py
    ```

The script will load the data, train the model, and print the top 5 most significant anomalies along with their detailed explanations to the console.

## Roadmap to Production

This section outlines the necessary steps to evolve this script from a proof-of-concept into a reliable, scalable, and scientifically rigorous tool ready for dashboard integration and academic publication.

### 1. Dashboard Integration and Scalability

To plug this system into a dashboard like Wearipedia, the code needs to be more modular and accessible.

-   **Modularize the Code**: Refactor the single script into separate Python modules. Create distinct files for:
    -   `data_loader.py`: A module dedicated to finding and loading the specific Fitbit data files.
    -   `feature_engineering.py`: A module to handle the creation of time-based and rolling-average features.
    -   `anomaly_model.py`: A module that contains the `IsolationForest` model, including functions for training and prediction.
    -   `llm_explainer.py`: A module for interacting with the Google Gemini API to generate explanations.
-   **Create an API Endpoint**: Build a simple API using **Flask** or **FastAPI**. This API would accept a date as input, run the entire analysis pipeline using the modules above, and return the detected anomalies and their explanations in a structured JSON format that a dashboard can easily consume.
-   **Configuration Management**: Move all hardcoded variables (like `contamination`, `window` size, file paths, and feature lists) into a separate configuration file (e.g., `config.yaml`). This allows for easy adjustments without changing the core code.

---
### 2. Adjustments for Scientific Research

To ensure the results are valid and publishable, the methodology needs to be rigorous and reproducible.

-   **Hyperparameter Tuning**: Systematically test different values for the `IsolationForest` model's `contamination` parameter (e.g., from 0.001 to 0.05) to find the optimal sensitivity for your specific research question.
-   **Cross-Validation**: Implement a time-series cross-validation strategy to ensure the model's performance is consistent across different periods and not just on one particular day's data.
-   **Statistical Significance**: For each detected anomaly, calculate a significance score. This could be based on the Z-score of the heart rate's deviation from the rolling average, providing a quantitative measure of how unusual each event is.
-   **Expand Feature Set**: Incorporate more data from your export, such as sleep stages (`UserSleepStages.csv`) or heart rate variability (`DailyHeartRateVariabilitySummary.csv`), to create a richer feature set. This could help explain *why* an anomaly occurred (e.g., high heart rate spikes correlating with poor sleep the night before).
-   **Literature Review & Comparison**: Research existing academic papers on wearable heart rate anomaly detection to validate that our methods and findings are consistent with or improve upon established approaches.
- **Variable/Feature Analysis**: Consider what features or variables are the most important to be considered for the anomaly detection algorithm. Test these systematically.  

---
### 3. Rigorous Testing Plan

A robust testing suite is essential for reliability and to ensure you can trust the output.

-   **Unit Tests**: Write individual tests for your core functions. For example:
    -   A test to confirm that the data loading function correctly parses a sample CSV.
    -   A test to ensure the rolling average calculation is correct.
-   **Data Validation**: Add checks to your data loading pipeline to automatically flag issues with the input files, such as missing columns or incorrect data types.
-   **Integration Tests**: Create a test that runs the entire pipeline on a sample dataset with known, pre-defined anomalies to ensure the system correctly identifies them.
-   **Benchmarking**: Compare the `IsolationForest` model's results against at least two other unsupervised anomaly detection algorithms (e.g., **DBSCAN** or **Local Outlier Factor** or **OCSVM**) to benchmark its performance and justify its selection in your research paper.