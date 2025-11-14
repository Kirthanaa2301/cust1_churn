Customer Churn Prediction — MLOps Deployment (MLflow + BentoML + Docker + CI/CD)

This repository implements a complete MLOps workflow for Customer Churn Prediction using MLflow for experiment tracking, BentoML for model serving, Docker for reproducible deployment, and GitHub Actions for CI/CD automation.
The system supports automatic retraining whenever new data is added or preprocessing logic changes.

1. Overview

This project provides:

Automated training and evaluation pipeline

MLflow Tracking Server and Model Registry

BentoML model serving

FastAPI prediction service

Docker-based deployment environment

CI/CD workflow for automated model retraining and redeployment

Batch prediction support for multiple uploaded files

Automatic model versioning and registry updates

2. Repository Structure
my_repo/
├── cust1_train.ipynb
├── cust1_prediction.ipynb
├── preprocessing_utils.py
├── customer_churn_dataset-training-master.csv
├── customer_churn_dataset-testing-master.csv
├── model.pkl

├── train.py
├── predict_service.py

├── bento_service/
│   ├── service.py
│   ├── bentofile.yaml
│   └── requirements.txt

├── fastapi_app/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt

├── mlflow_server/
│   ├── Dockerfile
│   └── requirements.txt

├── infra/
│   └── docker-compose.yml

└── .github/workflows/
    └── ci-cd.yml

3. System Architecture

The MLflow server manages experiment logs, metrics, and registered models.

The training pipeline loads data, applies preprocessing, trains the model, evaluates it, and pushes results to MLflow.

The CI/CD workflow automatically runs training whenever data or training logic changes.

BentoML builds a production-ready model service using the latest registered model.

A FastAPI service exposes prediction endpoints, including multi-file batch prediction.

Docker Compose orchestrates the entire system locally.

4. Running the System (Local Deployment)
Step 1: Start all services

From the infra directory:

docker-compose up --build


This starts:

MLflow Tracking Server at http://localhost:5000

BentoML model server at http://localhost:3000

FastAPI service at http://localhost:8000

5. Training the Model

To train manually:

python train.py


This will:

Load datasets

Apply preprocessing

Train the churn prediction model

Log metrics and artifacts to MLflow

Register a new model version in MLflow Model Registry

6. Automatic Retraining (CI/CD)

The workflow in .github/workflows/ci-cd.yml performs:

Notebook-to-Python conversion

Training and evaluation

MLflow logging

Model version registration

BentoML bundle build

Deployment refresh

It triggers automatically when any of the following change:

Training or testing datasets

Training notebook

Preprocessing functions

Training script

7. Model Deployment with BentoML

The model is served using a BentoML service defined in:

bento_service/service.py
bento_service/bentofile.yaml


To serve manually (optional):

bentoml serve bento_service/

8. Prediction API (FastAPI)

The FastAPI application exposes:

/predict-batch — accepts multiple CSV files

/healthz — service status

Documentation is available at:

http://localhost:8000/docs


Example batch request:

curl -X POST "http://localhost:8000/predict-batch" \
  -F "files=@data1.csv" \
  -F "files=@data2.csv"

9. MLflow Model Registry

The MLflow interface is available at:

http://localhost:5000


It provides:

Experiment lists

Parameters, metrics, artifacts

Registered model versions

Transition controls (e.g., staging, production)

10. Environment Variables

Create a .env file:

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123

11. Useful Commands

Stop all containers:

docker-compose down


Rebuild all:

docker-compose up --build


Show logs for a specific service:

docker-compose logs -f bento

12. Summary

This repository provides a production-grade MLOps pipeline that includes:

Automated retraining

Full experiment tracking

Versioned model management

Reproducible environment

Containerized deployment

Real-time and batch prediction APIs

If you need enhancements such as cloud deployment, DVC integration, or a Streamlit UI, support can be added easily.
