import os

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def train_model(
    input_path: str = "data/processed/train_data.parquet",
    config_path: str = "config.yaml",
) -> str:
    """
    Train the machine learning model.

    Args:
        input_path: Path to training data
        config_path: Path to configuration file

    Returns:
        str: Path to the trained model
    """
    config = load_config(config_path)

    train_data = pd.read_parquet(input_path)
    preprocessor = joblib.load("data/processed/preprocessor.pkl")

    X_train = train_data.drop(columns=["Is_Cancelled"])
    y_train = train_data["Is_Cancelled"]

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    model_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    model_pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "models/trained_model.pkl"
    joblib.dump(model_pipeline, model_path)

    return model_path
