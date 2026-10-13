#!/usr/bin/env python3
"""
Bank Term Deposit Prediction - End-to-End Pipeline
===================================================
Executes data loading, preprocessing, model training, validation
evaluation, test prediction, and solution checker verification.

Usage:
------
    python run_pipeline.py --model gradient_boosting
    python run_pipeline.py --model decision_tree
    python run_pipeline.py --compare-all
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_train_data, load_test_data, print_dataset_summary
from src.preprocess import prepare_features, split_data
from src.models import get_model, train_model, get_feature_importances, MODEL_REGISTRY
from src.evaluate import evaluate_model, print_evaluation_report
from src.predict import generate_submission, verify_against_solution_checker


def run(
    model_name: str = "gradient_boosting",
    compare_all: bool = False,
    data_dir: str = "data",
    output_dir: str = "outputs"
) -> None:
    data_path = Path(data_dir)
    train_file = data_path / "train.csv"
    test_file = data_path / "test.csv"
    
    print("\n" + "#" * 65)
    print("  BANK TERM DEPOSIT PREDICTION - MACHINE LEARNING PIPELINE")
    print("#" * 65)
    
    # 1. Ingestion
    print("\n[Step 1/5] Loading Datasets...")
    train_df = load_train_data(train_file)
    test_df = load_test_data(test_file)
    print_dataset_summary(train_df, test_df)

    # 2. Preprocessing & Alignment
    print("\n[Step 2/5] Preprocessing and Feature Encoding...")
    X, y, X_test, feature_names = prepare_features(train_df, test_df, drop_id=True)
    print(f"Features Generated: {len(feature_names)} columns")

    # 3. Train-Validation Split
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    # 4. Model Training & Evaluation
    if compare_all:
        print("\n[Step 3/5] Comparing All Available Models on Validation Set...")
        best_model_name = None
        best_acc = 0.0
        results = []

        for name in MODEL_REGISTRY.keys():
            clf = get_model(name)
            clf = train_model(clf, X_train, y_train)
            val_preds = clf.predict(X_val)
            val_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
            metrics = evaluate_model(y_val, val_preds, val_prob)
            print_evaluation_report(name, metrics)
            results.append((name, metrics['accuracy'], metrics.get('roc_auc')))
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_model_name = name

        print("\n" + "=" * 65)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 65)
        for name, acc, auc in results:
            auc_str = f", ROC-AUC: {auc:.4f}" if auc else ""
            print(f"  * {name:<22}: Accuracy = {acc:.4%}{auc_str}")
        print("=" * 65)
        selected_model_name = best_model_name
    else:
        selected_model_name = model_name

    print(f"\n[Step 4/5] Training Final Selected Model: '{selected_model_name}'...")
    final_model = get_model(selected_model_name)
    # Fit on full training set for optimal test performance
    final_model = train_model(final_model, X, y)

    # Top feature importances
    imp_df = get_feature_importances(final_model, feature_names)
    if not imp_df.empty:
        print("\nTop 10 Most Important Features:")
        print(imp_df.head(10).to_string(index=False))

    # 5. Prediction & Solution Checking
    print("\n[Step 5/5] Generating Test Predictions and Saving Submission...")
    sub_dest = Path(output_dir) / "submission.csv"
    generate_submission(final_model, X_test, test_df, output_path=sub_dest)

    checker_file = Path(output_dir) / "solution_checker.xlsx"
    if checker_file.exists():
        print("\nVerifying Predictions with solution_checker.xlsx...")
        verify_against_solution_checker(sub_dest, checker_file)

    print("\n" + "#" * 65)
    print("  PIPELINE EXECUTION COMPLETE SUCCESSFULLY")
    print("#" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Bank Term Deposit Prediction Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model algorithm to use (default: gradient_boosting)"
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all models on the validation split before selecting the best"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing train.csv and test.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save submission.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model_name=args.model,
        compare_all=args.compare_all,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
