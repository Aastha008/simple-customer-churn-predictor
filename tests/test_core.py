"""Unit tests for core machine learning pipeline and explainability."""

import os
import pytest
import pandas as pd
import numpy as np

from src.core import ChurnModelPipeline
from src.models import CustomerRecord


@pytest.fixture
def temp_model_path(tmp_path):
    return str(tmp_path / "test_model.joblib")


def test_pipeline_training(temp_model_path):
    pipeline = ChurnModelPipeline(model_path=temp_model_path)
    df, y = pipeline.generate_synthetic_data(n_samples=100)
    metrics = pipeline.train(df, y)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] >= 0.5
    assert os.path.exists(temp_model_path)


def test_single_prediction_and_explainability(temp_model_path):
    pipeline = ChurnModelPipeline(model_path=temp_model_path)
    df, y = pipeline.generate_synthetic_data(n_samples=200)
    pipeline.train(df, y)

    customer = CustomerRecord(
        tenure_months=2,
        monthly_charges=85.5,
        total_charges=171.0,
        support_tickets=4,
        contract_type="month-to-month",
        payment_method="electronic_check",
        login_frequency_weekly=1.0
    )

    response = pipeline.predict_single(customer)
    assert 0.0 <= response.churn_probability <= 1.0
    assert response.churn_prediction in [0, 1]
    assert response.risk_tier in ["Low", "Medium", "High"]
    assert len(response.explanations) > 0
    assert hasattr(response.explanations[0], "contribution")
