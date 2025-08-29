import joblib
import pandas as pd
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
    Validate the trained model performance.

    Args:
        model_path: Path to trained model
        test_data_path: Path to test data
        config_path: Path to configuration file

    Returns:
        dict: Model evaluation metrics
    """

    model_pipeline = joblib.load(model_path)
    test_data = pd.read_parquet(test_data_path)

    X_test = test_data.drop(columns=["Is_Cancelled"])
    y_test = test_data["Is_Cancelled"]

    y_pred = model_pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


if __name__ == "__main__":
    metrics = validate_model()
    print("\nValidation metrics:")
    for k, v in metrics.items():
        if k not in ["confusion_matrix", "classification_report"]:
            print(f"{k}: {v}")
