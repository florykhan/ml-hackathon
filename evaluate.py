from __future__ import annotations
import argparse
from dataclasses import asdict
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from model import (
    ModelConfig,
    apply_survey_missing_codes,
    build_pipeline,
    split_test_x,
    split_xy,
)
def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
    n_splits: int = 5,
    seeds: List[int] = [42, 123, 777, 2026, 999],
) -> pd.DataFrame:
    """
    Runs Stratified K-Fold ROC AUC across multiple seeds.
    Returns a tidy dataframe with per-seed results.
    """
    pipeline = build_pipeline(X, config)
    rows = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        rows.append({
            "seed": seed,
            "n_splits": n_splits,
            "mean_auc": float(np.mean(fold_scores)),
            "std_auc": float(np.std(fold_scores)),
            "fold_scores": ",".join([f"{s:.6f}" for s in fold_scores]),
        })
    return pd.DataFrame(rows)
def summarize_cv(results: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize per-seed CV results into one "official" estimate for the experiment.
    """
    return {
        "mean_auc_over_seeds": float(results["mean_auc"].mean()),
        "std_of_mean_auc_over_seeds": float(results["mean_auc"].std()),
        "avg_fold_std_auc": float(results["std_auc"].mean()),
    }
def append_log(results: pd.DataFrame, experiment: str, log_path: str = "cv_results.csv") -> None:
    """
    Append results to a CSV log for comparability across experiments.
    """
    results = results.copy()
    results.insert(0, "experiment", experiment)

    try:
        old = pd.read_csv(log_path)
        out = pd.concat([old, results], ignore_index=True)
    except Exception:
        out = results

    out.to_csv(log_path, index=False)
def train_full_and_write_submission(
    train_path: str,
    test_path: str,
    config: ModelConfig,
    out_path: str = "submission.csv",
) -> None:
    """
    Fits pipeline on full training set and writes submission probabilities for test set.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X, y = split_xy(train_df, config)
    test_ids, X_test = split_test_x(test_df, config)
    X = apply_survey_missing_codes(X, config)
    X_test = apply_survey_missing_codes(X_test, config)
    pipeline = build_pipeline(X, config)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({
        config.id_col: test_ids,
        config.target_col: proba,
    })
    submission.to_csv(out_path, index=False)
def main():
    parser = argparse.ArgumentParser(description="Evaluate model with CV and optionally write submission.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--experiment", default="baseline_lr_onehot_missingcodes", help="Experiment name for logging")
    parser.add_argument("--splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seeds", default="42,123,777,2026,999", help="Comma-separated list of random seeds")
    parser.add_argument("--log", default="cv_results.csv", help="Path to CV log file")
    parser.add_argument("--write-submission", action="store_true", help="Also train full and write submission.csv")
    parser.add_argument("--submission-out", default="submission.csv", help="Submission output path")

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    config = ModelConfig()
    train_df = pd.read_csv(args.train)
    X, y = split_xy(train_df, config)
    X = apply_survey_missing_codes(X, config)
    results = run_cv(X, y, config, n_splits=args.splits, seeds=seeds)
    summary = summarize_cv(results)
    print("\nPer-seed CV results:")
    print(results[["seed", "mean_auc", "std_auc"]].to_string(index=False))
    print("\nExperiment summary:")
    print(f"  experiment: {args.experiment}")
    print(f"  mean_auc_over_seeds: {summary['mean_auc_over_seeds']:.6f}")
    print(f"  std_of_mean_auc_over_seeds: {summary['std_of_mean_auc_over_seeds']:.6f}")
    print(f"  avg_fold_std_auc: {summary['avg_fold_std_auc']:.6f}")
    append_log(results, experiment=args.experiment, log_path=args.log)
    print(f"\nLogged results to: {args.log}")
    if args.write_submission:
        train_full_and_write_submission(
            train_path=args.train,
            test_path=args.test,
            config=config,
            out_path=args.submission_out,
        )
        print(f"Wrote submission to: {args.submission_out}")
if __name__ == "__main__":
    main()
