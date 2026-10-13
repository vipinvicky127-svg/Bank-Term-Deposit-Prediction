"""
Models Module
=============
Provides model initialization, training routines, and feature importance
utilities across multiple classification algorithms.
"""

from typing import Dict, Any, Optional
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


MODEL_REGISTRY: Dict[str, Any] = {
    'logistic_regression': {
        'class': LogisticRegression,
        'default_params': {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': None
        }
    },
    'decision_tree': {
        'class': DecisionTreeClassifier,
        'default_params': {
            'max_depth': 4,
            'random_state': 42
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'default_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'default_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 4,
            'random_state': 42
        }
    }
}


def get_model(name: str, **kwargs) -> BaseEstimator:
    """
    Instantiate a classifier by name with optional hyperparameter overrides.

    Parameters
    ----------
    name : str
        Model name: 'logistic_regression', 'decision_tree',
        'random_forest', or 'gradient_boosting'.
    **kwargs : dict
        Hyperparameter overrides.

    Returns
    -------
    BaseEstimator
        Configured scikit-learn estimator instance.
    """
    key = name.lower().replace('-', '_').replace(' ', '_')
    if key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    config = MODEL_REGISTRY[key]
    params = config['default_params'].copy()
    params.update(kwargs)
    return config['class'](**params)


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> BaseEstimator:
    """
    Fit the model on training data.
    """
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model: BaseEstimator, feature_names: list) -> pd.DataFrame:
    """
    Extract and rank feature importances or coefficients.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        col_name = 'Importance'
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
        col_name = 'Coefficient'
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        col_name: importances
    })
    df_imp['AbsValue'] = df_imp[col_name].abs()
    df_imp = df_imp.sort_values(by='AbsValue', ascending=False).drop(columns=['AbsValue'])
    df_imp.reset_index(drop=True, inplace=True)
    return df_imp
