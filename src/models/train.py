import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import shap
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
    Train the machine learning model with MLflow tracking.

    Args:
        input_path: Path to training data
        config_path: Path to configuration file

    Returns:
        str: Path to the trained model
    """
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.set_tag("model_type", "RandomForestClassifier")

        train_data = pd.read_parquet(input_path)
        preprocessor = joblib.load("data/processed/preprocessor.pkl")

        X_train = train_data.drop(columns=["Is_Cancelled"])
        y_train = train_data["Is_Cancelled"]

        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["model"]["random_state"],
        )

        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])
        mlflow.log_param("random_state", config["model"]["random_state"])
        mlflow.log_param("test_size", config["model"]["test_size"])

        model_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        model_pipeline.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        model_path = "models/trained_model.pkl"
        joblib.dump(model_pipeline, model_path)

        mlflow.sklearn.log_model(model_pipeline, "model")

        generate_shap_plot(model_pipeline, X_train.sample(100))

        return model_path


def generate_shap_plot(model_pipeline, sample_data):
    """Generate and log SHAP plot."""
    try:
        X_transformed = model_pipeline.named_steps["preprocessor"].transform(
            sample_data
        )

        explainer = shap.TreeExplainer(model_pipeline.named_steps["model"])
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, show=False, max_display=10)
        plt.tight_layout()

        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        shap_plot_path = os.path.join(results_dir, "shap_summary.png")
        plt.savefig(shap_plot_path)
        plt.close()

        mlflow.log_artifact(shap_plot_path)
        os.remove(shap_plot_path)

    except Exception as e:
        print(f"Could not generate SHAP plot: {e}")
