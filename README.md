# Health Anomaly Detection with Wearable Data

This project provides a backend service for detecting, ranking, and explaining physiological anomalies from intraday Fitbit data. It uses an Isolation Forest machine learning model and Google's Gemini LLM to translate complex data into human-readable explanations suitable for clinical researchers.

The system is designed as a flexible, scalable, and scientifically rigorous engine for research platforms like "Easy Ants", empowering researchers to monitor study participants effectively.

## Key Features

- **Multi-Day Analysis**: Processes continuous data over specified date ranges to capture long-term trends.
- **Flexible Anomaly Targeting**: Allows researchers to target specific metrics (e.g., `heart_rate`, `steps`) for anomaly ranking.
- **Rich Contextual Data**: Integrates multiple data streams for a holistic analysis, including:
    - Intraday Heart Rate & Step Count
    - Sleep Stages (Deep, Light, REM)
    - Heart Rate Variability (HRV)
    - Participant-specific questionnaire data (e.g., exercise habits, caffeine use).
- **Statistical Rigor**: Ranks anomalies using Z-scores for a quantitative measure of statistical significance.
- **Automated Explanations**: Leverages a Large Language Model (LLM) to generate clear, context-aware explanations for each detected anomaly.
- **API-Driven**: Exposes a Flask API for easy integration with web dashboards.
- **Tested & Benchmarked**: Includes a unit testing suite for reliability and a benchmarking script to validate the choice of the `IsolationForest` model.

---

## Methodology

Our pipeline is designed to be both powerful and interpretable.

#### 1. Feature Engineering
The model's performance relies on a rich feature set. We engineer several key features from the raw data:
-   **`hr_rolling_avg` & `hr_rolling_std`**: A 5-minute rolling average and standard deviation of the heart rate. This provides immediate, localized context for each data point.
-   **Sleep Metrics**: Data from the previous night's sleep (e.g., `sleep_deep_minutes`, `sleep_awakenings`) are included, as poor sleep can significantly impact next-day physiology.
-   **Heart Rate Variability (HRV)**: The daily `hrv_rmssd` provides insight into the participant's autonomic nervous system state and stress levels.
-   **Questionnaire Data**: Categorical data from participant questionnaires (e.g., exercise habits) is one-hot encoded to provide crucial lifestyle context.

#### 2. Anomaly Detection Model
We use the **Isolation Forest** algorithm. Our benchmarking showed it provides the best balance of speed and accuracy for this type of time-series data. It works by "isolating" observations by randomly selecting a feature and then randomly selecting a split value. The logic is that anomalous points are "easier" to isolate and will have shorter path lengths in the decision trees.

#### 3. Anomaly Ranking
Instead of just taking the model's raw output, we add a layer of statistical rigor. When targeting `heart_rate`, we rank the detected anomalies by their **Z-score**, which measures how many standard deviations a point is from its local rolling average. This provides a quantifiable and scientifically respected measure of how unusual each event is.

---

## Project Structure

The project is organized into a modular structure for maintainability and scalability.

-   `config.py`: Central configuration for file paths, API keys, and model parameters.
-   `data_loader.py`: Handles loading, merging, and encoding of all data sources.
-   `feature_engineering.py`: Creates time-based and rolling-window features.
-   `anomaly_model.py`: Contains the Isolation Forest model for detecting and ranking anomalies.
-   `llm_explainer.py`: Interacts with the Google Gemini API to generate explanations.
-   `pipeline.py`: Orchestrates the entire workflow from data loading to explanation.
-   `app.py`: Runs the Flask web server and defines the API endpoints.
-   `tuner.py`: A utility script to help researchers tune the model's sensitivity.
-   `benchmarker.py`: A utility script to compare different anomaly detection models.
-   `tests/`: Contains all unit tests to ensure code reliability.

---

## How to Use

### 1. Setup
Install the required Python packages:
```bash
pip install pandas scikit-learn google-generativeai Flask
```

### 2. Configuration
Open `config.py` and set the required variables, including your `GOOGLE_API_KEY` and the correct paths to your data files.

### 3. Running the API Server
To start the anomaly detection service, run the following command from your terminal:
```bash
python app.py
```
The server will start and be accessible at `http://127.0.0.1:5000`.

### 4. API Documentation

The service exposes one primary endpoint:

**`GET /analyze_range`**

Triggers the full analysis pipeline for a specified date range.

**Query Parameters:**
-   `start_date` (required): The start of the date range in `YYYY-MM-DD` format.
-   `end_date` (required): The end of the date range in `YYYY-MM-DD` format.
-   `target` (optional): The feature to rank anomalies by. Defaults to `heart_rate`. Can also be `steps`.

**Example Request (using curl):**
```bash
# Analyze heart rate anomalies for the first week of July
curl "[http://127.0.0.1:5000/analyze_range?start_date=2025-07-01&end_date=2025-07-07&target=heart_rate](http://127.0.0.1:5000/analyze_range?start_date=2025-07-01&end_date=2025-07-07&target=heart_rate)"
```

**Example Success Response (200 OK):**
```json
{
  "date_range_analyzed": "2025-07-01 to 2025-07-07",
  "results": [
    {
      "anomaly_data": {
        "anomaly": -1.0,
        "heart_rate": 150.0,
        "z_score": 4.5,
        "...": "..."
      },
      "explanation": "1. Summary: ...\n2. Reason for Flag: ...\n..."
    }
  ],
  "status": "success"
}
```

### 5. Running Tests
To verify that all components are working correctly, run the unit test suite:
```bash
python -m unittest discover
```

## How to Contribute
Contributions are welcome. Please feel free to submit a pull request or open an issue for any bugs or feature requests.