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
