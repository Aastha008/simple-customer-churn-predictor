<div align="center">

# Simple Customer Churn Predictor

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI Status](https://github.com/daily-project-factory/simple-customer-churn-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/daily-project-factory/simple-customer-churn-predictor/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)

  <p align="center">
    <strong>A lightweight, production-ready machine learning pipeline and REST API for evaluating subscriber retention metrics and predicting customer churn risk in real time.</strong>
  </p>
</div>

<br>

---

## 📋 Table of Contents
1. [Problem Statement & Target Audience](#-problem-statement--target-audience)
2. [Key Features](#-key-features)
3. [Architecture & System Design](#-architecture--system-design)
4. [Technology Stack](#-technology-stack)
5. [Installation & Virtual Environment Setup](#-installation--virtual-environment-setup)
6. [Environment Variables Configuration](#-environment-variables-configuration)
7. [Usage: CLI Commands & API Endpoints](#-usage-cli-commands--api-endpoints)
8. [Automated Testing Instructions](#-automated-testing-instructions)
9. [Project Structure Tree](#-project-structure-tree)
10. [Roadmap / Future Enhancements](#-roadmap--future-enhancements)
11. [License](#-license)

---

## 🎯 Problem Statement & Target Audience

### The Problem
Subscription-based businesses frequently struggle to identify at-risk customers before they cancel their subscriptions. Without proactive intervention signals, companies lose valuable recurring revenue due to delayed insights, opaque churn drivers, and a lack of real-time behavioral monitoring.

### Target Audience
* **Data Scientists & ML Engineers:** Seeking a clean, extensible pattern for lightweight Scikit-Learn training pipelines paired with production FastAPI wrappers.
* **Product & Growth Teams:** Needing immediate, explainable risk-scoring endpoints to integrate inside CRM workflows or internal customer success dashboards.
* **Engineering Teams:** Requiring strict input schema validation via Pydantic and robust automated testing (`pytest`) for ML inference pipelines.

---

## ✨ Key Features

* **Automated Model Training:** Train classification models on historical customer usage and billing records via clean CLI commands.
* **Real-Time & Batch Scoring:** Execute low-latency churn probability scoring via high-performance FastAPI endpoints.
* **Explainable AI (XAI):** Generate local feature contribution breakdowns using lightweight coefficient extraction to explain *why* an account is flagged at risk.
* **Rigorous Payload Validation:** Validate input feature distributions and payload schemas deterministically with Pydantic models.

---

## 🏗️ Architecture & System Design

The system follows a modular, decoupled design separating model lifecycle management from API routing and input validation.

```
                  +-----------------------------------+
                  |            Client / CLI           |
                  +-----------------------------------+
                        |                        |
             (CLI Train/Eval)             (HTTP Requests)
                        |                        v
                        |             +---------------------+
                        |             |    src/api.py       |
                        v             | (FastAPI Endpoints) |
            +---------------------+   +---------------------+
            |     src/main.py     |              |
            | (Uvicorn / Runner)  |              v
            +---------------------+   +---------------------+
                       |              |    src/models.py    |
                       v              | (Pydantic Schemas)  |
            +---------------------+   +---------------------+
            |     src/core.py     |--------------+
            | (Scikit-Learn/Joblib)              |
            +---------------------+              v
                       |              +---------------------+
                       +------------->| Model Persistence   |
                                      |    (artifacts/)     |
                                      +---------------------+
```

### Core Architecture Modules
* **`src/models.py`**: Houses Pydantic models defining customer usage records, validation schemas, prediction requests, and explainability responses.
* **`src/core.py`**: Encapsulates the machine learning pipeline implementation, model training, persistence via `joblib`, and real-time inference with feature contribution extraction.
* **`src/api.py`**: Defines FastAPI HTTP routes, middleware, global error handling, and endpoints for single and batch churn scoring.
* **`src/main.py`**: Serves as the Uvicorn server startup script and CLI operations orchestrator for model training and evaluation.

---

## 💻 Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| **Language** | Python 3.9+ | Core application logic and data structures |
| **API Framework** | FastAPI | High-performance asynchronous HTTP REST endpoints |
| **ML Engine** | Scikit-Learn | Classification estimators and preprocessing pipelines |
| **Data Processing** | Pandas / NumPy | Tabular data manipulation and vector operations |
| **Model Serialization**| Joblib | Fast disk persistence of trained ML pipelines |
| **Data Validation** | Pydantic | Strict runtime type checking and schema enforcement |
| **Testing** | Pytest | Automated unit and integration testing suite |

---

## 📦 Installation & Virtual Environment Setup

Follow these steps to set up your development environment locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/daily-project-factory/simple-customer-churn-predictor.git
   cd simple-customer-churn-predictor
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On macOS / Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🔐 Environment Variables Configuration

Create a `.env` file in the root directory by copying the provided example template.

```bash
cp .env.example .env
```

### `.env.example` Reference:
```ini
# Application Configuration
APP_ENV=development
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Model Artifact Paths
MODEL_PATH=artifacts/churn_model.joblib
SCALER_PATH=artifacts/scaler.joblib
```

---

## 🚀 Usage: CLI Commands & API Endpoints

### 1. Model Training via CLI
Before starting the API server, ensure you have trained and persisted a model instance:
```bash
python src/main.py train --data-path path/to/customer_data.csv
```

### 2. Run the FastAPI Server
Launch the development server via Uvicorn:
```bash
python src/main.py serve
```
* Interactive API Documentation (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
* Alternative Documentation (ReDoc): [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 3. API Endpoints

#### **`POST /predict`**
Evaluate a single subscriber's churn risk and obtain explainable feature contributions.

* **Request Body Example:**
  ```json
  {
    "tenure_months": 12,
    "monthly_charges": 65.50,
    "total_charges": 786.00,
    "support_tickets_per_month": 2,
    "contract_type": 1,
    "is_active_user": true
  }
  ```

* **Response Example:**
  ```json
  {
    "churn_probability": 0.78,
    "is_at_risk": true,
    "top_contributing_features": [
      {"feature": "support_tickets_per_month", "contribution": 0.45},
      {"feature": "tenure_months", "contribution": -0.22}
    ]
  }
  ```

#### **`POST /predict/batch`**
Score multiple user accounts simultaneously in a single HTTP payload.

---

## 🧪 Automated Testing Instructions

The test suite covers data validation schemas, pipeline training logic, and API endpoint behaviors. Run all tests using `pytest`:

```bash
# Run complete test suite
pytest -v

# Run with code coverage report
pytest --cov=src --cov-report=term-missing
```

---

## 📁 Project Structure Tree

```
simple-customer-churn-predictor/
├── .env.example
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── core.py
│   ├── main.py
│   └── models.py
└── tests/
    ├── __init__.py
    ├── test_api.py
    └── test_core.py
```

---

## 🗺️ Roadmap / Future Enhancements

* [ ] **Advanced Algorithms:** Integrate Gradient Boosting (XGBoost / LightGBM) alongside Scikit-Learn linear models.
* [ ] **Model Registry Integration:** Add MLflow or Weights & Biases tracking for experiment version control.
* [ ] **Automated Retraining Webhooks:** Implement scheduled background jobs (via Celery or APScheduler) to retrain models on fresh database snapshots.
* [ ] **Dockerization:** Provide optimized multi-stage `Dockerfile` and `docker-compose.yml` configurations.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` or choose options in standard repository headers for more information.