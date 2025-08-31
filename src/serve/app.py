from datetime import datetime
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from src.features.transform import transform_data


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        with open("/opt/airflow/config.yaml") as file:
            return yaml.safe_load(file)


config = load_config()

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

app = FastAPI(
    title="Uber Ride Cancellation Prediction API",
    description="API for serving ML model predictions for ride cancellation",
    version="1.0.0",
    docs_url="/docs",
)


class InputData(BaseModel):
    """Input schema for prediction requests."""

    avg_vtat: float = Field(
        ..., description="Average vehicle turnaround time", example=15.5
    )
    timestamp: str = Field(
        ..., description="Timestamp in ISO format", example="2023-01-01T12:00:00"
    )

    model_config = {
        "json_schema_extra": {
            "example": {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}
        }
    }


def preprocess_input(data: dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess input data using the same transformations as training pipeline.
    Uses the transform_data function from transform.py to ensure consistency.

    Args:
        data: Dictionary containing input features

    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction
    """
    df = pd.DataFrame([data])

    processed_df = transform_data(
        input_df=df, config_path="config.yaml", for_inference=True
    )

    return processed_df


def load_champion_model():
    """Load the champion model from MLflow Model Registry."""
    try:
        model_name = config["mlflow"]["model_name"]
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load champion model: {str(e)}"
        )


def get_model_metadata():
    """Get model metadata from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = config["mlflow"]["model_name"]

        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise HTTPException(
                status_code=404,
                detail="No production model found in MLflow Model Registry",
            )

        model_version = versions[0]
        run = client.get_run(model_version.run_id)

        return {
            "model_version": model_version,
            "run": run,
            "params": run.data.params,
            "metrics": run.data.metrics,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve model metadata: {str(e)}"
        )


def get_feature_importance(model, feature_names: list[str]) -> list[dict[str, Any]]:
    """Extract feature importance from the model."""
    try:
        if hasattr(model, "_model_impl") and hasattr(model._model_impl, "coef_"):
            importances = np.abs(model._model_impl.coef_[0])

            feature_importance = []
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importance.append(
                        {"feature": feature_names[i], "importance": float(importance)}
                    )

            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            return feature_importance[:5]

        return []

    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
        return []


@app.post("/predict")
async def predict(input_data: InputData):
    """
    Make predictions using the champion model.

    Args:
        input_data: Input features for prediction

    Returns:
        Dict containing the prediction result
    """
    try:
        model = load_champion_model()

        processed_data = preprocess_input(input_data.model_dump())

        prediction = model.predict(processed_data)

        prediction_proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(processed_data)
                prediction_proba = float(proba[0][1])
            except (AttributeError, IndexError, ValueError, TypeError) as e:
                print(f"Warning: Could not get prediction probability: {e}")
                pass

        response = {"prediction": int(prediction[0])}

        if prediction_proba is not None:
            response["probability"] = prediction_proba

        return response

    except ValidationError as ve:
        raise HTTPException(
            status_code=400, detail=f"Invalid input data: {ve.errors()}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model")
async def get_model_info():
    """
    Get information about the currently deployed model.

    Returns:
        Dict containing model hyperparameters, feature importance, and input schema
    """
    try:
        metadata = get_model_metadata()

        model = load_champion_model()

        expected_features = []
        for col in config["features"]["feature_cols"]:
            if col not in config["features"].get("datetime_cols", []):
                expected_features.append(col)

        if "datetime_cols" in config["features"]:
            for col in config["features"]["datetime_cols"]:
                expected_features.extend(
                    [f"{col}_hour", f"{col}_day_of_week", f"{col}_month"]
                )

        top_features = get_feature_importance(model, expected_features)

        input_schema = InputData.model_json_schema()

        return {
            "hyperparameters": metadata["params"],
            "metrics": metadata["metrics"],
            "top_features": top_features,
            "input_schema": {
                "properties": input_schema["properties"],
                "required": input_schema["required"],
                "type": input_schema["type"],
            },
            "model_info": {
                "model_name": config["mlflow"]["model_name"],
                "version": metadata["model_version"].version,
                "stage": metadata["model_version"].current_stage,
                "run_id": metadata["model_version"].run_id,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try to load the model to ensure it's accessible
        load_champion_model()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": True,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": False,
            "error": str(e),
        }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Uber Ride Cancellation Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict (POST)",
            "model_info": "/model (GET)",
            "health": "/health (GET)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
