"""
Preprocessing Module
====================
Provides end-to-end data transformation, categorical encoding,
feature alignment, and train-validation splitting.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CATEGORICAL_COLUMNS = [
    'job', 'marital', 'education', 'default', 'housing',
    'loan', 'contact', 'month', 'poutcome'
]

NUMERICAL_COLUMNS = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
]


def prepare_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    drop_id: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], List[str]]:
    """
    Encode categorical features and align columns across train and test sets.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training data.
    test_df : Optional[pd.DataFrame]
        Raw test data.
    drop_id : bool, default=True
        Whether to exclude the 'ID' column from model features.

    Returns
    -------
    X : pd.DataFrame
        Processed training feature matrix.
    y : pd.Series
        Binary target vector (1 for 'yes', 0 for 'no').
    X_test : Optional[pd.DataFrame]
        Processed and aligned test feature matrix.
    feature_names : List[str]
        Final list of aligned feature column names.
    """
    df_train = train_df.copy()
    
    # Extract target
    if 'subscribed' not in df_train.columns:
        raise KeyError("Column 'subscribed' missing from training dataframe.")
        
    y = (df_train['subscribed'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    features_to_drop = ['subscribed']
    if drop_id and 'ID' in df_train.columns:
        features_to_drop.append('ID')

    X_train_raw = df_train.drop(columns=features_to_drop)
    
    # One-hot encode train features
    X = pd.get_dummies(X_train_raw, drop_first=True)
    feature_names = list(X.columns)

    X_test = None
    if test_df is not None:
        df_test = test_df.copy()
        if drop_id and 'ID' in df_test.columns:
            df_test = df_test.drop(columns=['ID'])
        X_test_raw = pd.get_dummies(df_test, drop_first=True)
        # Align test features to match train features exactly
        X_test = X_test_raw.reindex(columns=feature_names, fill_value=0)

    return X, y, X_test, feature_names


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and target vector into training and validation sets.
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )


def scale_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """
    Standardize features using StandardScaler fitted only on X_train.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
