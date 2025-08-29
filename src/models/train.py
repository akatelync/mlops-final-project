import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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
    Train a RandomForest pipeline and log model + metrics to MLflow.
    """

    # Load config
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Set experiment
    try:
        experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
        if experiment is None:
            mlflow.create_experiment(config["mlflow"]["experiment_name"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
    except Exception as e:
        print(f"Could not set experiment: {e}")

    # Load preprocessed data
    train_data = pd.read_parquet(input_path)
    preprocessor = joblib.load("data/processed/preprocessor.pkl")

    X = train_data.drop(columns=[config["model"]["target_column"]])
    y = train_data[config["model"]["target_column"]]

    # Split for evaluation
    test_size = config["model"].get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config["model"]["random_state"]
    )

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    # Build full pipeline
    model_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    model_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log model + metrics to MLflow
    with mlflow.start_run():
        mlflow.set_tag("model_type", "RandomForestPipeline")
        mlflow.log_params(
            {
                "n_estimators": config["model"]["n_estimators"],
                "max_depth": config["model"]["max_depth"],
                "random_state": config["model"]["random_state"],
                "test_size": test_size,
            }
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model_pipeline, artifact_path="model")

    # Save locally as well
    os.makedirs("models", exist_ok=True)
    model_path = "models/trained_model.pkl"
    joblib.dump(model_pipeline, model_path)

    print(f"Model trained. Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(f"Local pipeline saved to: {model_path}")
    print("MLflow run logged.")

    return model_path
