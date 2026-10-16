"""Pydantic schemas for data validation, predictions, and explainability."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field, conint, confloat


class CustomerRecord(BaseModel):
    """Customer usage and billing metrics input schema."""
    tenure_months: conint(ge=0) = Field(..., description="Number of months the customer has been subscribed")
    monthly_charges: confloat(ge=0.0) = Field(..., description="Current monthly billing amount in USD")
    total_charges: confloat(ge=0.0) = Field(..., description="Total amount billed to the customer to date")
    support_tickets: conint(ge=0) = Field(..., description="Number of customer support tickets raised in the last 6 months")
    contract_type: str = Field(..., description="Contract type: 'month-to-month', 'one-year', or 'two-year'")
    payment_method: str = Field(..., description="Payment method: 'electronic_check', 'mailed_check', 'bank_transfer', or 'credit_card'")
    login_frequency_weekly: confloat(ge=0.0) = Field(..., description="Average weekly logins to the platform")


class FeatureContribution(BaseModel):
    """Individual feature contribution to the churn score."""
    feature: str
    value: Any
    coefficient: float
    contribution: float


class PredictionResponse(BaseModel):
    """Prediction result response schema including risk tier and feature breakdown."""
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of customer churn")
    churn_prediction: int = Field(..., description="Binary prediction: 1 for churn, 0 for retain")
    risk_tier: str = Field(..., description="Categorical risk level: Low, Medium, High")
    explanations: List[FeatureContribution] = Field(..., description="Local feature contributions explaining the prediction")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    customers: List[CustomerRecord] = Field(..., description="List of customer records to score")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[PredictionResponse]
