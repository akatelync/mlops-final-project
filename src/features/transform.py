import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "/opt/airflow/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def transform_data(
    input_path: str = None,
    config_path: str = "/opt/airflow/config.yaml",
    input_df: pd.DataFrame = None,
    for_inference: bool = False,
) -> str | pd.DataFrame:
    """
    Transform data for model training or inference.

    Args:
        input_path: Path to raw data (from config if None, ignored if input_df provided)
        config_path: Path to configuration file
        input_df: DataFrame to transform (for inference/single data point)
        for_inference: If True, returns transformed DataFrame instead of saving files

    Returns:
        str: Path to the transformed training data file (training mode)
        pd.DataFrame: Transformed DataFrame (inference mode)
    """
    config = load_config(config_path)

    if not for_inference:
        print(f"Config keys: {list(config.keys())}")
        print(f"Full config: {config}")

    # Check if features exists before processing
    if "features" not in config:
        raise KeyError(
            f"'features' key missing from config. Available keys: {list(config.keys())}"
        )

    # Load data - either from file or use provided DataFrame
    if input_df is not None:
        df = input_df.copy()
    else:
        if input_path is None:
            input_path = config["data"]["ingested_path"]
        df = pd.read_parquet(input_path)

    # Apply datetime transformations
    if "datetime_cols" in config["features"]:
        for col in config["features"]["datetime_cols"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_day_of_week"] = df[col].dt.dayofweek
                df[f"{col}_month"] = df[col].dt.month
                df = df.drop(columns=[col])

                # Only extend numerical_cols for training mode to avoid modifying config
                if not for_inference:
                    config["features"]["numerical_cols"].extend(
                        [f"{col}_hour", f"{col}_day_of_week", f"{col}_month"]
                    )

    # Get feature columns
    feature_cols = [
        col
        for col in config["features"]["feature_cols"]
        if col not in config["features"].get("datetime_cols", [])
    ]

    if "datetime_cols" in config["features"]:
        for col in config["features"]["datetime_cols"]:
            feature_cols.extend([f"{col}_hour", f"{col}_day_of_week", f"{col}_month"])

    # For inference mode, just return the transformed features
    if for_inference:
        return df[feature_cols]

    # Training mode - continue with train/test split and file saving
    target_col = config["model"]["target_column"]

    X = df[feature_cols]
    y = df[target_col]

    test_size = config["model"].get("test_size", 0.2)
    random_state = config["model"].get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    os.makedirs("data/processed", exist_ok=True)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_path = config["data"]["processed_path"]
    test_path = "data/processed/test_data.parquet"

    train_data.to_parquet(train_path, index=False)
    test_data.to_parquet(test_path, index=False)

    return train_path


if __name__ == "__main__":
    output_path = transform_data()
    print(f"Transformed training data saved to: {output_path}")
