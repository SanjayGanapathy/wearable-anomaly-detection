# app.py

from flask import Flask, request, jsonify
from pipeline import run_pipeline
import warnings
import config

warnings.simplefilter(action="ignore", category=FutureWarning)

app = Flask(__name__)


@app.route("/analyze_range", methods=["GET"])
def analyze_data_range():
    """
    API endpoint to trigger the anomaly detection pipeline for a date range.
    Accepts 'start_date', 'end_date', and an optional 'target' query parameter.
    e.g., /analyze_range?start_date=...&end_date=...&target=steps
    """
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    target_feature = request.args.get("target", config.DEFAULT_TARGET_FEATURE)

    if not all([start_date, end_date]):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Missing 'start_date' or 'end_date' parameter.",
                }
            ),
            400,
        )

    print(f"Received request to analyze data from: {start_date} to {end_date}")
    print(f"Target feature for ranking: {target_feature}")

    analysis_result = run_pipeline(start_date, end_date, target_feature)

    if analysis_result.get("status") == "error":
        return jsonify(analysis_result), 500

    return jsonify(analysis_result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
