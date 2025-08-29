import os

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def ingest_data(config_path: str = "config.yaml") -> str:
    """
    Load raw data and prepare for processing.

    Args:
        config_path: Path to configuration file

    Returns:
        str: Path to the processed data file for passing to next Airflow task
    """
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]

    df = pd.read_csv(raw_path)

    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df.drop(["Date", "Time"], axis=1, inplace=True)

    df = df[df["Booking Status"] != "No Driver Found"]

    df["Is_Cancelled"] = df["Booking Status"].apply(
        lambda x: 1 if "Cancelled" in x else 0
    )

    df["Payment Method"] = df["Payment Method"].fillna("Unknown")

    os.makedirs("data/processed", exist_ok=True)
    processed_path = "data/processed/ingested_data.parquet"
    df.to_parquet(processed_path, index=False)

    return processed_path
