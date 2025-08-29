import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -----------------------------
# Load configuration
# -----------------------------
def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


config = load_config()

MLFLOW_TRACKING_URI = config["mlflow"]["tracking_uri"]
MODEL_NAME = config["mlflow"]["model_name"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# -----------------------------
# Load Champion Model
# -----------------------------
def load_champion_model():
    """
    Load the production ('champion') model from MLflow Model Registry
    """
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading champion model: {e}")
        return None


model = load_champion_model()
if model is None:
    raise RuntimeError("Champion model could not be loaded from MLflow.")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Uber Ride Cancellation Prediction",
    description="Serve ML model for predicting ride cancellations",
    version="1.0.0",
)


# -----------------------------
# Input Schema
# -----------------------------
class PredictRequest(BaseModel):
    # Replace these with your actual features
    BookingID: int = Field(..., example=12345)
    CustomerID: int = Field(..., example=54321)
    VehicleType: str = Field(..., example="Sedan")
    PaymentMethod: str = Field(..., example="Credit Card")
    Distance: float = Field(..., example=10.5)
    # Add more features based on your model


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/model")
def get_model_info():
    """Return model metadata"""
    try:
        return {
            "model_name": MODEL_NAME,
            "stage": "Production",
            "tracking_uri": MLFLOW_TRACKING_URI,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(payload: PredictRequest):
    """Predict Is_Cancelled for incoming ride request"""
    try:
        # Convert payload to DataFrame
        df = pd.DataFrame([payload.dict()])

        # Predict using champion model
        prediction = model.predict(df)
        proba = (
            model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
        )

        return {
            "prediction": int(prediction[0]),
            "probability": float(proba[0]) if proba is not None else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
