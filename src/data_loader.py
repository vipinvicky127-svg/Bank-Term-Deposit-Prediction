"""
Data Loader Module
==================
Handles ingestion, schema verification, and summary statistics
for training and test datasets.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd


EXPECTED_FEATURE_COLUMNS = [
    'ID', 'age', 'job', 'marital', 'education', 'default',
    'balance', 'housing', 'loan', 'contact', 'day', 'month',
    'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

TARGET_COLUMN = 'subscribed'


def load_train_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the training dataset from a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the train.csv file.

    Returns
    -------
    pd.DataFrame
        Loaded and verified training dataframe.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path.resolve()}")

    df = pd.read_csv(path)
    
    # Check expected columns
    missing = [c for c in EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {missing}")
        
    return df


def load_test_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the test dataset from a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the test.csv file.

    Returns
    -------
    pd.DataFrame
        Loaded and verified test dataframe.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path.resolve()}")

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Test dataset is missing required columns: {missing}")

    return df


def print_dataset_summary(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> None:
    """
    Print formatted summary statistics of the loaded datasets.
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Train Shape: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    if test_df is not None:
        print(f"Test Shape : {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
    
    if TARGET_COLUMN in train_df.columns:
        counts = train_df[TARGET_COLUMN].value_counts()
        pcts = train_df[TARGET_COLUMN].value_counts(normalize=True) * 100
        print("-" * 60)
        print("Target Distribution ('subscribed'):")
        for val, count in counts.items():
            print(f"  {val:5s}: {count:,} ({pcts[val]:.2f}%)")
    print("=" * 60)
