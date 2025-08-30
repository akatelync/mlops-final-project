import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
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

    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    try:
        experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
        if experiment is None:
            mlflow.create_experiment(config["mlflow"]["experiment_name"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
    except Exception as e:
        print(f"Could not set experiment: {e}")

    train_data = pd.read_parquet(input_path)
    preprocessor = joblib.load("data/processed/preprocessor.pkl")

    X = train_data.drop(columns=[config["model"]["target_column"]])
    y = train_data[config["model"]["target_column"]]

    test_size = config["model"].get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config["model"]["random_state"]
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    model_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    model_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model_pipeline.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

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

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model_pipeline, name="model")
