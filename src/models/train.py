import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "/opt/airflow/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def train_model(
    input_path: str = "data/processed/train_data.parquet",
    config_path: str = "/opt/airflow/config.yaml",
) -> str:
    """
    Train a LogisticRegression pipeline and log model + metrics to MLflow.
    """
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    try:
        experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
        if experiment is None:
            mlflow.create_experiment(config["mlflow"]["experiment_name"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
    except Exception as e:
        print(f"Could not set experiment: {e}")

    if input_path is None:
        input_path = config["data"]["processed_path"]

    train_data = pd.read_parquet(input_path)

    target_col = config["model"]["target_column"]
    X = train_data.drop(columns=[target_col])
    y = train_data[target_col]

    test_size = config["model"].get("test_size", 0.2)
    random_state = config["model"].get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression(random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tag("model_type", "LogisticRegressionPipeline")
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "random_state": random_state,
                "test_size": test_size,
            }
        )

        mlflow.log_metric("train_accuracy", accuracy)
        mlflow.log_metric("train_precision", precision)
        mlflow.log_metric("train_recall", recall)
        mlflow.log_metric("train_f1_score", f1)

        signature = mlflow.models.infer_signature(X_train, y_pred)
        input_example = X_train.iloc[:5]

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    return f"runs:/{run_id}/model"


if __name__ == "__main__":
    model_uri = train_model()
    print(f"Model trained and logged: {model_uri}")
