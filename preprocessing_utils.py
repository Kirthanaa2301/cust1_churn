import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_churn_data(
    df: pd.DataFrame,
    is_train: bool = True,
    scaler: StandardScaler = None,
    encoders: dict = None,
    expected_columns: list = None
):
    """
    Complete preprocessing pipeline for customer churn project.
    Handles missing values, encoding, scaling, and ensures the same
    feature order/columns between training & inference.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    is_train : bool
        Whether this is training phase.
    scaler : StandardScaler
        Previously fitted scaler (in inference).
    encoders : dict
        Dict of fitted LabelEncoders (in inference).
    expected_columns : list
        Feature column order saved during training.

    Returns
    -------
    X : pd.DataFrame
        Fully processed features.
    y : pd.Series or None
        Target column (only in training).
    scaler : StandardScaler
        Fitted scaler (in training).
    encoders : dict
        Fitted encoders (in training).
    customer_ids : pd.Series
        For attaching to predictions.
    expected_columns : list
        Final feature order to be reused in inference.
    """

    df = df.copy()

    # Extract ID for merging back after prediction
    customer_ids = df["CustomerID"] if "CustomerID" in df.columns else None

    # -------------------------------------------------------------
    # 1. Handle Missing Values Safely
    # -------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Fill numeric missing with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing with most frequent
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -------------------------------------------------------------
    # 2. Encode Categorical Columns Consistently
    # -------------------------------------------------------------
    if is_train:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        # Inference: use trained encoders
        for col in categorical_cols:
            if col in encoders:
                known_classes = set(encoders[col].classes_)
                df[col] = df[col].astype(str).apply(
                    lambda val: encoders[col].transform([val])[0]
                    if val in known_classes else -1      # unseen category handling
                )
            else:
                # Categorical col missing in model → drop it
                df[col] = -1

    # -------------------------------------------------------------
    # 3. Extract Target Variable
    # -------------------------------------------------------------
    y = None
    if "Churn" in df.columns:
        y = df["Churn"]
        df.drop(columns=["Churn"], inplace=True)

    # -------------------------------------------------------------
    # 4. Remove CustomerID from features
    # -------------------------------------------------------------
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    # -------------------------------------------------------------
    # 5. Fix Column Order (critical for MLOps)
    # -------------------------------------------------------------
    if is_train:
        expected_columns = df.columns.tolist()
    else:
        # Reorder and add missing columns (set to 0)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

    # -------------------------------------------------------------
    # 6. Scale numeric features
    # -------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if is_train:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, y, scaler, encoders, customer_ids, expected_columns
