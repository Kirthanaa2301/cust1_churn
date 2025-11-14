# bento_service/service.py
import bentoml
from bentoml.io import JSON, PandasDataFrame
import pandas as pd
import os

# Service name
SERVICE_NAME = "churn_service"

# We'll load the latest model that Bento has in its model store under tag 'churn_model'
# In CI we will import MLflow model into Bento store with name 'churn_model_mlflow'
MODEL_BENTO_NAME = os.environ.get("BENTO_MODEL_NAME", "churn_model_mlflow:latest")

svc = bentoml.Service(SERVICE_NAME)

# Create a runner from the imported model
try:
    runner = bentoml.get(MODEL_BENTO_NAME).to_runner()
    svc = bentoml.Service(SERVICE_NAME, runners=[runner])
except Exception:
    # If not present at import time, define service without runner;
    # runner will be available once CI imports/builder has run
    svc = bentoml.Service(SERVICE_NAME)

@svc.api(input=JSON(), output=JSON())
def predict(json_obj):
    """
    Accepts JSON body: { "rows": [ {col1: val, col2: val, ...}, ... ] }
    Returns: { "predictions": [...] }
    """
    # Lazy fetch runner if not yet available
    global runner
    try:
        runner = runner
    except NameError:
        # attempt to get latest model runner
        runner = bentoml.get(MODEL_BENTO_NAME).to_runner()
        svc.runners = [runner]

    df = pd.DataFrame(json_obj.get("rows", []))
    # Use runner to predict
    preds = runner.predict.run(df)
    return {"predictions": preds.tolist()}
