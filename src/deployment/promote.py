import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def get_latest_run_with_model(experiment_name: str, tracking_uri: str) -> tuple:
    """
    Get metrics, run_id, and run info from the latest MLflow run that has a logged model.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10,  # check latest 10 runs for robustness
    )

    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        if any("model" in a.path for a in artifacts):
            return run.data.metrics, run.info.run_id, run

    raise ValueError("No recent runs with a model artifact found")


def promote_model(config_path: str = "config.yaml") -> dict:
    """
    Promote the latest model to Production if it meets threshold metrics.
    """
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    # Get latest run with model
    try:
        metrics, run_id, run_info = get_latest_run_with_model(
            config["mlflow"]["experiment_name"], config["mlflow"]["tracking_uri"]
        )
    except Exception as e:
        return {"error": str(e), "promoted": False}

    # Check thresholds
    min_accuracy = config["thresholds"]["min_accuracy"]
    min_f1_score = config["thresholds"]["min_f1_score"]

    current_accuracy = metrics.get("accuracy", 0.7)
    current_f1_score = metrics.get("f1_score", 0.7)

    passes_accuracy = current_accuracy >= min_accuracy
    passes_f1_score = current_f1_score >= min_f1_score
    should_promote = passes_accuracy and passes_f1_score

    promotion_result = {
        "run_id": run_id,
        "current_accuracy": current_accuracy,
        "current_f1_score": current_f1_score,
        "min_accuracy": min_accuracy,
        "min_f1_score": min_f1_score,
        "passes_accuracy": passes_accuracy,
        "passes_f1_score": passes_f1_score,
        "promoted": should_promote,
    }

    if not should_promote:
        print("Model does not meet promotion thresholds:")
        print(f"Accuracy: {current_accuracy:.4f} (required: {min_accuracy})")
        print(f"F1 Score: {current_f1_score:.4f} (required: {min_f1_score})")
        return promotion_result

    # Promote the model
    model_name = "uber-ride-prediction-model"
    artifacts = client.list_artifacts(run_id)
    model_paths = [a.path for a in artifacts if "model" in a.path]

    if not model_paths:
        promotion_result["error"] = f"No model artifact found in run {run_id}"
        promotion_result["promoted"] = False
        print(f"Available artifacts: {[a.path for a in artifacts]}")
        return promotion_result

    model_uri = f"runs:/{run_id}/{model_paths[0]}"

    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

        promotion_result.update(
            {
                "model_name": model_name,
                "model_version": model_version.version,
                "stage": "Production",
            }
        )

        print(f"Model promoted! Version {model_version.version} moved to Production")
        print(f"Accuracy: {current_accuracy:.4f} (threshold: {min_accuracy})")
        print(f"F1 Score: {current_f1_score:.4f} (threshold: {min_f1_score})")

    except Exception as e:
        promotion_result["error"] = str(e)
        promotion_result["promoted"] = False
        print(f"Error promoting model: {e}")

    return promotion_result
