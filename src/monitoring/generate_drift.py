import os

import mlflow
import pandas as pd
import yaml
from evidently import Report
from evidently.metrics import DataDriftPreset


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def log_drift_reports_to_mlflow(
    reference_path=None, current_path=None, config_path="config.yaml"
):
    """
    Detect data drift using Evidently and log results to MLflow.
    Returns a results dict suitable for Airflow XCom.
    """
    config = load_config(config_path)

    # Set default paths from config if not provided
    if reference_path is None:
        reference_path = config["data"]["reference_path"]
    if current_path is None:
        current_path = config["data"].get(
            "current_path", "data/current/drifted_data.parquet"
        )

    # Ensure files exist
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"Current file not found: {current_path}")

    # Load data
    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    # Run Evidently Data Drift detection
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    results_dict = report.as_dict()
    drift_metric = next(
        (m for m in results_dict["metrics"] if m.get("metric") == "DataDriftPreset"),
        None,
    )

    dataset_drift = drift_metric["result"]["dataset_drift"] if drift_metric else False
    drift_share = (
        drift_metric["result"]["share_of_drifted_columns"] if drift_metric else 0.0
    )

    # Log metrics to MLflow
    with mlflow.start_run(run_name="drift_detection", nested=True):
        mlflow.log_metric("dataset_drift", int(dataset_drift))
        mlflow.log_metric("drift_share", drift_share)

    print(
        f"Data drift detected: {dataset_drift}, Share of drifted columns: {drift_share:.2f}"
    )

    # Return results for Airflow XCom
    return {
        "data_drift_detected": dataset_drift,
        "drift_share": drift_share,
        "reference_size": len(reference_df),
        "current_size": len(current_df),
    }
