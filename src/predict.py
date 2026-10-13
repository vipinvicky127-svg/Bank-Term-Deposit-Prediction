"""
Prediction & Submission Generation Module
=========================================
Generates predictions on the test dataset and exports formatted
CSV submission files compatible with solution_checker.xlsx.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


def generate_submission(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str | Path = "outputs/submission.csv"
) -> pd.DataFrame:
    """
    Generate predictions for the test dataset and save as a submission CSV.

    Parameters
    ----------
    model : BaseEstimator
        Fitted classifier.
    X_test : pd.DataFrame
        Aligned feature matrix for the test dataset.
    test_df : pd.DataFrame
        Raw test dataframe containing the original 'ID' column.
    output_path : str or Path
        Destination path for the generated submission CSV.

    Returns
    -------
    pd.DataFrame
        Generated submission dataframe containing 'ID' and 'subscribed'.
    """
    if 'ID' not in test_df.columns:
        raise KeyError("Original test_df must contain the 'ID' column.")

    # Generate binary predictions
    preds_binary = model.predict(X_test)

    # Map numeric labels to expected categorical values
    label_map = {0: 'no', 1: 'yes'}
    preds_text = [label_map.get(p, 'no') for p in preds_binary]

    submission_df = pd.DataFrame({
        'ID': test_df['ID'].values,
        'subscribed': preds_text
    })

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(dest, index=False)
    print(f"[SUCCESS] Submission saved to: {dest.resolve()}")
    print(f"          Total Records: {len(submission_df):,}")
    print(f"          Distribution : {dict(submission_df['subscribed'].value_counts())}")

    return submission_df


def verify_against_solution_checker(
    submission_csv: str | Path,
    solution_checker_xlsx: str | Path
) -> Optional[float]:
    """
    Calculate the exact test accuracy by comparing submission.csv
    against ground truth in solution_checker.xlsx (Sheet2).
    """
    import zipfile
    import xml.etree.ElementTree as ET

    sub_path = Path(submission_csv)
    xlsx_path = Path(solution_checker_xlsx)

    if not sub_path.exists() or not xlsx_path.exists():
        print("[WARNING] Cannot verify: file(s) not found.")
        return None

    sub_df = pd.read_csv(sub_path)
    preds = sub_df['subscribed'].tolist()

    with zipfile.ZipFile(xlsx_path) as z:
        ss_xml = ET.fromstring(z.read('xl/sharedStrings.xml'))
        strings = [t.text for t in ss_xml.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')]
        s2_xml = ET.fromstring(z.read('xl/worksheets/sheet2.xml'))
        
        actuals = []
        for row in s2_xml.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')[1:]:
            cells = {c.attrib.get('r')[0]: (strings[int(c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v').text)] if c.attrib.get('t') == 's' else c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v').text) for c in row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c')}
            actuals.append(cells.get('A'))

    if len(preds) != len(actuals):
        print(f"[ERROR] Length mismatch: preds={len(preds)}, actuals={len(actuals)}")
        return None

    correct = sum(1 for p, a in zip(preds, actuals) if p == a)
    accuracy = correct / len(actuals)
    print(f"[SOLUTION CHECKER RESULT] Correct: {correct:,} / {len(actuals):,} | Accuracy: {accuracy:.4%}")
    return accuracy
