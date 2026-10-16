"""Core Machine Learning Pipeline: training, preprocessing, inference, and explainability."""

import os
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models import CustomerRecord, FeatureContribution, PredictionResponse


NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "login_frequency_weekly"
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method"
]


class ChurnModelPipeline:
    """Encapsulates Scikit-Learn pipeline and persistence logic for churn prediction."""

    def __init__(self, model_path: str = "artifacts/churn_model.joblib"):
        self.model_path = model_path
        self.pipeline: Pipeline | None = None
        self.feature_names_out: List[str] = []

    def build_pipeline(self) -> Pipeline:
        """Constructs the scikit-learn preprocessing and logistic regression pipeline."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERICAL_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ]
        )

        classifier = LogisticRegression(class_weight="balanced", random_state=42)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        return pipeline

    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generates synthetic customer data for training and testing."""
        np.random.seed(42)
        tenure = np.random.randint(1, 72, n_samples)
        monthly = np.random.uniform(20.0, 120.0, n_samples)
        total = tenure * monthly * np.random.uniform(0.9, 1.1, n_samples)
        tickets = np.random.poisson(1.5, n_samples)
        logins = np.random.exponential(3.0, n_samples)

        contracts = np.random.choice(["month-to-month", "one-year", "two-year"], n_samples, p=[0.5, 0.3, 0.2])
        payments = np.random.choice(["electronic_check", "mailed_check", "bank_transfer", "credit_card"], n_samples)

        df = pd.DataFrame({
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "contract_type": contracts,
            "payment_method": payments,
            "login_frequency_weekly": logins
        })

        # Churn logic heuristic + noise
        churn_score = (
            (df["contract_type"] == "month-to-month").astype(int) * 1.5 +
            (df["support_tickets"] > 3).astype(int) * 1.2 -
            (df["tenure_months"] / 20.0) -
            (df["login_frequency_weekly"] / 5.0) +
            np.random.normal(0, 0.5, n_samples)
        )
        churn = (churn_score > 0.5).astype(int)
        return df, churn

    def train(self, df: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Trains the ML pipeline and saves it to disk."""
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(df, y)

        # Extract transformed feature names for explainability
        cat_encoder = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        encoded_cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names_out = NUMERICAL_FEATURES + encoded_cat_features

        preds = self.pipeline.predict(df)
        probs = self.pipeline.predict_proba(df)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "roc_auc": float(roc_auc_score(y, probs))
        }

        self.save_model()
        return metrics

    def save_model(self) -> None:
        """Saves the pipeline to disk via joblib."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_names_out": self.feature_names_out}, self.model_path)

    def load_model(self) -> bool:
        """Loads the pipeline from disk if available."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.pipeline = data["pipeline"]
            self.feature_names_out = data["feature_names_out"]
            return True
        return False

    def predict_single(self, customer: CustomerRecord) -> PredictionResponse:
        """Scores a single customer record and generates local feature contributions."""
        if self.pipeline is None:
            success = self.load_model()
            if not success:
                raise RuntimeError("Model not trained or found. Please train the model first.")

        df = pd.DataFrame([customer.model_dump()])
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        pred = int(self.pipeline.predict(df)[0])

        # Risk Tier calculation
        if prob < 0.3:
            risk_tier = "Low"
        elif prob < 0.7:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        # Local Feature Contribution Extraction
        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]

        Xt = preprocessor.transform(df)
        coefficients = classifier.coef_[0]

        explanations: List[FeatureContribution] = []
        for idx, feature_name in enumerate(self.feature_names_out):
            val = float(Xt[0, idx])
            coef = float(coefficients[idx])
            contribution = val * coef
            # Map back to original input where possible for readability
            orig_val = None
            for num_feat in NUMERICAL_FEATURES:
                if feature_name == num_feat:
                    orig_val = getattr(customer, num_feat)
            if orig_val is None:
                orig_val = feature_name

            explanations.append(FeatureContribution(
                feature=feature_name,
                value=orig_val,
                coefficient=coef,
                contribution=round(contribution, 4)
            ))

        # Sort explanations by absolute contribution magnitude descending
        explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=pred,
            risk_tier=risk_tier,
            explanations=explanations
        )
