import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def get_latest_run_metrics(experiment_name: str, tracking_uri: str) -> dict:
    """Get metrics from the latest MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in experiment")

    latest_run = runs[0]
    return latest_run.data.metrics, latest_run.info.run_id, latest_run


def promote_model(config_path: str = "config.yaml") -> dict:
    """
    Check model performance against thresholds and promote if it passes.
    """
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    # Get latest run metrics
    metrics, run_id, run_info = get_latest_run_metrics(
        config["mlflow"]["experiment_name"], config["mlflow"]["tracking_uri"]
    )

    # Check thresholds
    min_accuracy = config["thresholds"]["min_accuracy"]
    min_f1_score = config["thresholds"]["min_f1_score"]

    current_accuracy = metrics.get("accuracy", 0.0)
    current_f1_score = metrics.get("f1_score", 0.0)

    # Determine if model should be promoted
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

    if should_promote:
        model_name = "uber-ride-prediction-model"

        try:
            # Check if model artifact exists
            artifacts = client.list_artifacts(run_id, path="")
            model_artifacts = [a for a in artifacts if a.path == "model"]

            if not model_artifacts:
                promotion_result["error"] = f"No model artifact found in run {run_id}"
                promotion_result["promoted"] = False
                print(f"Available artifacts: {[a.path for a in artifacts]}")
                return promotion_result

            # Get model URI from the run
            model_uri = f"runs:/{run_id}/model"

            # Register the model
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

            # Transition to Production stage
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )

            promotion_result["model_name"] = model_name
            promotion_result["model_version"] = model_version.version
            promotion_result["stage"] = "Production"

            print(
                f"Model promoted! Version {model_version.version} moved to Production"
            )
            print(f"Accuracy: {current_accuracy:.4f} (threshold: {min_accuracy})")
            print(f"F1 Score: {current_f1_score:.4f} (threshold: {min_f1_score})")

        except Exception as e:
            promotion_result["error"] = str(e)
            promotion_result["promoted"] = False
            print(f"Error promoting model: {e}")

    else:
        print("Model does not meet promotion thresholds:")
        print(f"Accuracy: {current_accuracy:.4f} (required: {min_accuracy})")
        print(f"F1 Score: {current_f1_score:.4f} (required: {min_f1_score})")

    return promotion_result
