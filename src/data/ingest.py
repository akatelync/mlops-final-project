import os

import pandas as pd
import yaml


def load_config(config_path: str = "/opt/airflow/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        config_path = "config.yaml"

    with open(config_path) as file:
        return yaml.safe_load(file)


def ingest_data(config_path: str = "/opt/airflow/config.yaml") -> str:
    """
    Load raw data and prepare for processing.

    Args:
        config_path: Path to configuration file

    Returns:
        str: Path to the processed data file for passing to next Airflow task
    """
    config = load_config()
    raw_path = config["data"]["raw_path"]

    df = pd.read_csv(raw_path)

    df.columns = df.columns.str.replace(" ", "_").str.lower()

    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.drop(["date", "time"], axis=1, inplace=True)

    df = df[~df["booking_status"].isin(["No Driver Found", "Incomplete"])]
    df["is_cancelled"] = df["booking_status"].apply(
        lambda x: 1 if "Cancelled" in x else 0
    )

    df["payment_method"] = df["payment_method"].fillna("Unknown")

    os.makedirs("data/processed", exist_ok=True)
    processed_path = "data/processed/ingested_data.parquet"
    df.to_parquet(processed_path, index=False)

    return processed_path
