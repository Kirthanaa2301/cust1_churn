# train/register_mlflow.py
import mlflow
import joblib
import os
from mlflow.exceptions import MlflowException

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_path = os.path.join("models", "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found. Run train.py first.")

model = joblib.load(model_path)

with mlflow.start_run() as run:
    mlflow.log_param("registered_via", "register_mlflow.py")
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="churn_model")
    print("Registered model with MLflow")
