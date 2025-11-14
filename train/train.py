# train/train.py
"""
Train script — used by CI. Reads CSVs, runs preprocessing_utils, trains a model,
saves model locally and logs it to MLflow (via register step).
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing_utils import preprocess  # your existing functions

# Change these paths if needed
TRAIN_CSV = "customer_churn_dataset-training-master.csv"
TEST_CSV = "customer_churn_dataset-testing-master.csv"
OUT_MODEL_DIR = "models"
OUT_MODEL_PATH = os.path.join(OUT_MODEL_DIR, "model.pkl")

# Use MLflow tracking URI from env var if provided
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return train, test

def prepare(df):
    # Expect preprocess to return (X, y) or modified dataframe.
    # We'll attempt common patterns: if preprocess returns tuple, use it.
    result = preprocess(df)  # from your preprocessing_utils.py
    if isinstance(result, tuple) and len(result) == 2:
        X, y = result
    else:
        # If preprocess returns df with target column named 'target' or 'churn'
        df_pre = result
        if 'churn' in df_pre.columns:
            y = df_pre['churn']
            X = df_pre.drop(columns=['churn'])
        elif 'target' in df_pre.columns:
            y = df_pre['target']
            X = df_pre.drop(columns=['target'])
        else:
            raise ValueError("preprocess must return (X, y) or dataframe with 'churn' or 'target'.")
    return X, y

def train_and_log():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    train_df, test_df = load_data()
    X_train, y_train = prepare(train_df)
    X_test, y_test = prepare(test_df)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    # Save local copy
    joblib.dump(clf, OUT_MODEL_PATH)
    print(f"Saved model to {OUT_MODEL_PATH}")

    # Log to MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("train_acc", float(train_acc))
        mlflow.log_metric("test_acc", float(test_acc))
        mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name="churn_model")
        print("Logged model to MLflow (and attempted register). Run ID:", run.info.run_id)

if __name__ == "__main__":
    train_and_log()
