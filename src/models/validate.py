import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def validate_model(
    model_path: str = "models/trained_model.pkl",
    test_data_path: str = "data/processed/test_data.parquet",
    config_path: str = "config.yaml",
) -> dict:
    """
    Validate the trained model performance with MLflow logging.

    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        config_path: Path to configuration file

    Returns:
        dict: Model evaluation metrics
    """
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    with mlflow.start_run():
        model_pipeline = joblib.load(model_path)
        test_data = pd.read_parquet(test_data_path)

        X_test = test_data.drop(columns=["Is_Cancelled"])
        y_test = test_data["Is_Cancelled"]

        y_pred = model_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        cm_plot_path = "confusion_matrix.png"
        plt.savefig(cm_plot_path)
        plt.close()

        mlflow.log_artifact(cm_plot_path)

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        return metrics
