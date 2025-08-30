import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import shap
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_config(config_path: str = "/opt/airflow/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def validate_model(
    model_path: str,
    test_data_path: str = "data/processed/test_data.parquet",
    config_path: str = "/opt/airflow/config.yaml",
    mlflow_run_id: str = None,
) -> dict:
    """
    Validate the trained model performance with MLflow logging.

    Args:
        model_path: MLflow model URI (e.g., "runs:/run_id/model")
        test_data_path: Path to test data
        config_path: Path to configuration file
        mlflow_run_id: MLflow run ID to continue logging

    Returns:
        dict: Model evaluation metrics
    """
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Load model from MLflow
    if model_path.startswith("runs:/"):
        model_pipeline = mlflow.sklearn.load_model(model_path)
    else:
        raise ValueError("model_path should be MLflow URI format (runs:/run_id/model)")

    test_data = pd.read_parquet(test_data_path)
    target_col = config["model"]["target_column"]

    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    y_pred = model_pipeline.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metric("val_precision", precision)
            mlflow.log_metric("val_recall", recall)
            mlflow.log_metric("val_f1_score", f1)
            mlflow.log_metric("val_accuracy", accuracy)

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix - Validation")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name, bbox_inches="tight", dpi=150)
                plt.close()

                mlflow.log_artifact(tmp_file.name, artifact_path="validation_plots")

                os.unlink(tmp_file.name)

            class_report = classification_report(y_test, y_pred)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp_file:
                tmp_file.write("Classification Report - Validation\n")
                tmp_file.write("=" * 40 + "\n")
                tmp_file.write(class_report)
                tmp_file.flush()

                mlflow.log_artifact(tmp_file.name, artifact_path="validation_reports")
                os.unlink(tmp_file.name)

            # Generate SHAP plots
            try:
                # Use a subset of test data for SHAP (for performance)
                X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)

                # Create SHAP explainer
                explainer = shap.Explainer(model_pipeline, X_test_sample)
                shap_values = explainer(X_test_sample)

                # Summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_sample, show=False)
                plt.title("SHAP Summary Plot - Feature Importance")

                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp_file:
                    plt.savefig(tmp_file.name, bbox_inches="tight", dpi=150)
                    plt.close()

                    mlflow.log_artifact(tmp_file.name, artifact_path="shap_plots")
                    os.unlink(tmp_file.name)

                # Waterfall plot for first prediction
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(shap_values[0], show=False)
                plt.title("SHAP Waterfall Plot - Single Prediction Explanation")

                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp_file:
                    plt.savefig(tmp_file.name, bbox_inches="tight", dpi=150)
                    plt.close()

                    mlflow.log_artifact(tmp_file.name, artifact_path="shap_plots")
                    os.unlink(tmp_file.name)

            except Exception as e:
                print(f"Warning: Could not generate SHAP plots: {e}")

            min_accuracy = config.get("thresholds", {}).get("min_accuracy", 0.0)
            min_f1_score = config.get("thresholds", {}).get("min_f1_score", 0.0)

            mlflow.log_params(
                {
                    "val_accuracy_threshold_passed": accuracy >= min_accuracy,
                    "val_f1_threshold_passed": f1 >= min_f1_score,
                    "val_accuracy_threshold": min_accuracy,
                    "val_f1_threshold": min_f1_score,
                }
            )

    else:
        pass

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "thresholds_passed": {
            "accuracy": accuracy
            >= config.get("thresholds", {}).get("min_accuracy", 0.0),
            "f1_score": f1 >= config.get("thresholds", {}).get("min_f1_score", 0.0),
        },
    }

    return metrics


if __name__ == "__main__":
    # For standalone testing
    metrics = validate_model(
        model_path="runs:/your_run_id/model",
        test_data_path="data/processed/test_data.parquet",
    )
    print(f"Validation completed: {metrics}")
