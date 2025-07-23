# config.py

# -- FILE PATHS & API KEYS --
START_DATE = "2025-07-01"
END_DATE = "2025-07-07"

# IMPORTANT: Replace with your actual Google API Key
GOOGLE_API_KEY = "AIzaSyAXeR1JhC9UUGf0D14iRiYNcYtQbTBWsVU"

# Please adjust these paths to your actual data directories
BASE_PATH = "Fitbit/Physical Activity_GoogleData/"
SLEEP_PATH = "Fitbit/Sleep_GoogleData/sleep-stages-2025.csv"
HRV_PATH = (
    "Fitbit/Physical Activity_GoogleData/daily_heart_rate_variability_summary.csv"
)

# -- TARGET FEATURE --
# The default feature to focus on for anomaly ranking.
# Can be overridden by the API call. e.g., "heart_rate", "steps"
DEFAULT_TARGET_FEATURE = "heart_rate"

# -- MODEL PARAMETERS --
# This value should be set based on the tuner script's output
ISOLATION_FOREST_CONTAMINATION = 0.01
RANDOM_STATE = 42

# -- FEATURE ENGINEERING PARAMETERS --
ROLLING_WINDOW_SIZE = 300
FEATURES = [
    "heart_rate",
    "steps",
    "hour",
    "hr_rolling_avg",
    "sleep_deep_minutes",
    "sleep_light_minutes",
    "sleep_rem_minutes",
    "sleep_awakenings",
    "hrv_rmssd",
    "hrv_coverage",
]
