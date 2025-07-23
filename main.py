# main.py

import warnings
import config
from pipeline import run_pipeline
import json

# Suppress pandas FutureWarnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)

def main():
    """
    Runs the full anomaly detection and explanation pipeline from the command line.
    """
    print(f"--- Running anomaly detection for date: {config.TARGET_DATE} ---")
    results = run_pipeline(config.TARGET_DATE)
    
    # Pretty-print the JSON output
    print(json.dumps(results, indent=2))
    print("\n--- Pipeline finished ---")

if __name__ == "__main__":
    main()