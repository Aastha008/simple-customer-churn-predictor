"""FastAPI application routes, error handling, and scoring endpoints."""

import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.models import CustomerRecord, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from src.core import ChurnModelPipeline


app = FastAPI(
    title="Simple Customer Churn Predictor API",
    description="Lightweight ML pipeline for real-time and batch customer churn scoring with local feature explanations.",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/churn_model.joblib")
pipeline_manager = ChurnModelPipeline(model_path=MODEL_PATH)

# Attempt loading model at startup; if missing, endpoints will train or handle gracefully
if not pipeline_manager.load_model():
    try:
        df, y = pipeline_manager.generate_synthetic_data(500)
        pipeline_manager.train(df, y)
    except Exception:
        pass


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict:
    """Health check endpoint."""
    model_loaded = pipeline_manager.pipeline is not None
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_churn(customer: CustomerRecord) -> PredictionResponse:
    """Evaluate real-time churn risk and feature contributions for a single customer."""
    try:
        result = pipeline_manager.predict_single(customer)
        return result
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_churn_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Evaluate churn risk for a batch of customers."""
    try:
        predictions = [pipeline_manager.predict_single(cust) for cust in payload.customers]
        return BatchPredictionResponse(predictions=predictions)
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Batch prediction error: {str(e)}")
