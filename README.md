# Health Anomaly Detection with Wearable Data

This project analyzes intraday time-series data from Fitbit wearables to detect, rank, and explain physiological anomalies. It uses an Isolation Forest machine learning model to identify unusual patterns and leverages Google's Gemini Large Language Model (LLM) to translate these findings into human-readable explanations suitable for non-technical researchers.

The system is designed to be a flexible, scalable, and scientifically rigorous backend for clinical research platforms like "Easy Ants".

## Key Features

- **Multi-Day Analysis**: Processes continuous data over date ranges (weeks or months) to capture long-term trends.
- **Flexible Anomaly Targeting**: Researchers can specify which metric to target for anomaly detection (e.g., `heart_rate`, `steps`), tailoring the analysis to their study.
- **Rich Contextual Data**: Integrates multiple data streams for a holistic analysis, including:
    - Intraday Heart Rate
    - Step Count
    - Sleep Stages (Deep, Light, REM, Awakenings)
    - Heart Rate Variability (HRV)
- **Statistical Rigor**: Ranks anomalies using Z-scores to provide a quantitative measure of statistical significance.
- **Automated Explanations**: Utilizes a Large Language Model (LLM) to generate clear, context-aware explanations for each detected anomaly.
- **API-Driven**: Exposes a Flask API endpoint for easy integration with web dashboards and other applications.
- **Tested & Benchmarked**: Includes a suite of unit tests for reliability and a benchmarking script to validate the choice of the `IsolationForest` model against alternatives.

## Project Structure

The project is organized into a modular and maintainable structure:

-   `config.py`: Central configuration for file paths, API keys, and model parameters.
-   `data_loader.py`: Handles loading and merging of all raw data sources (heart rate, steps, sleep, HRV).
-   `feature_engineering.py`: Creates time-based and rolling-window features for the model.
-   `anomaly_model.py`: Contains the Isolation Forest model for detecting and ranking anomalies.
-   `llm_explainer.py`: Interacts with the Google Gemini API to generate explanations.
-   `pipeline.py`: Orchestrates the entire workflow from data loading to explanation.
-   `app.py`: Runs the Flask web server and defines the API endpoints.
-   `tuner.py`: A utility script to help researchers tune the model's sensitivity.
-   `benchmarker.py`: A utility script to compare different anomaly detection models.
-   `tests/`: Contains all unit tests to ensure code reliability.

## How to Use

### 1. Setup

First, install the required Python packages:

```bash
pip install pandas scikit-learn google-generativeai Flask
```

### 2. Configuration

Open `config.py` and set the following variables:
-   `GOOGLE_API_KEY`: Your API key for the Google Gemini service.
-   `BASE_PATH`, `SLEEP_PATH`, `HRV_PATH`: The correct paths to your data directories.
-   `START_DATE`, `END_DATE`: The default date range for analysis.
-   `ISOLATION_FOREST_CONTAMINATION`: Set this value based on the results of the `tuner.py` script.

### 3. Running the API Server

To start the anomaly detection service, run the following command from your terminal:

```bash
python app.py
```
The server will start and be accessible at `http://127.0.0.1:5000`.

### 4. Making an API Request

You can request an analysis by sending a GET request to the `/analyze_range` endpoint.

**Parameters:**
-   `start_date` (required): The start of the date range (e.g., `2025-07-01`).
-   `end_date` (required): The end of the date range (e.g., `2025-07-07`).
-   `target` (optional): The feature to rank anomalies by. Defaults to `heart_rate`. Can also be `steps`.

**Example Request (using curl):**
```bash
# Analyze heart rate anomalies for the first week of July
curl "[http://127.0.0.1:5000/analyze_range?start_date=2025-07-01&end_date=2025-07-07&target=heart_rate](http://127.0.0.1:5000/analyze_range?start_date=2025-07-01&end_date=2025-07-07&target=heart_rate)"

# Analyze step anomalies for the same period
curl "[http://127.0.0.1:5000/analyze_range?start_date=2025-07-01&end_date=2025-07-07&target=steps](http://127.0.0.1:5000/analyze_range?start_date=2025-07-01&end_date=2025-07-07&target=steps)"
```

### 5. Running Tests

To verify that all components are working correctly, run the unit test suite:

```bash
python -m unittest discover
```

## Future Work

-   **Dashboard Integration**: Build a front-end interface to consume the API.
-   **Production Deployment**: Deploy the Flask API using a production-grade server like Gunicorn.
-   **Expand Feature Set**: Integrate additional data sources like respiratory rate and skin temperature.