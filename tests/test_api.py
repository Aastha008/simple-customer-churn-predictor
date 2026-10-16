"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app, pipeline_manager

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def setup_model():
    # Ensure model is trained for API tests
    df, y = pipeline_manager.generate_synthetic_data(300)
    pipeline_manager.train(df, y)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "tenure_months": 12,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "support_tickets": 1,
        "contract_type": "one-year",
        "payment_method": "credit_card",
        "login_frequency_weekly": 4.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_tier" in data
    assert "explanations" in data


def test_predict_validation_error():
    # Invalid negative tenure
    payload = {
        "tenure_months": -5,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "support_tickets": 1,
        "contract_type": "one-year",
        "payment_method": "credit_card",
        "login_frequency_weekly": 4.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_endpoint():
    payload = {
        "customers": [
            {
                "tenure_months": 12,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "support_tickets": 1,
                "contract_type": "one-year",
                "payment_method": "credit_card",
                "login_frequency_weekly": 4.5
            },
            {
                "tenure_months": 1,
                "monthly_charges": 100.0,
                "total_charges": 100.0,
                "support_tickets": 6,
                "contract_type": "month-to-month",
                "payment_method": "electronic_check",
                "login_frequency_weekly": 0.2
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
