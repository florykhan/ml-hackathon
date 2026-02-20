# Technical Report: Overqualification Prediction (NGS Dataset)

This document describes the methodology, pipeline, and results for the **SFU Data Science Student Society ML Hackathon** project on overqualification prediction using the NGS (National Graduate Survey) structured hiring dataset.

---

## 1. Problem and Goal

**Goal:** Build a robust model that accurately estimates **overqualification probability** (candidates whose profiles exceed role requirements) based on candidate attributes: education level, years of experience, skill composition, prior roles, and demographics.

**Dataset:** NGS structured hiring data with one row per respondent; target is binary `overqualified` (1 = overqualified, 0 = not). Features are survey-derived codes (education, program, employment, demographics, etc.).

**Evaluation:** Accuracy on the hackathon Public and Private leaderboards. Our tuned model achieved:
- **0.75174** accuracy on the Public leaderboard (best: 0.76623)
- **0.70511** accuracy on the Private leaderboard (best: 0.71304)

This placed the solution close to the top-performing teams and demonstrated strong generalization on unseen data.

---

## 2. Pipeline Overview

The pipeline is modular and lives under `src/`:

| Component | Role |
|-----------|------|
| `config.py` | Paths, target/id columns, validation settings |
| `data.py` | Load train/test CSVs, split X/y, train/val split |
| `preprocess.py` | Clean NGS codes (6, 9, 99 → missing), normalize mixed-type columns (GENDER2, DDIS_FL, VISBMINP) |
| `features.py` | Convert code columns to categorical strings for CatBoost; expose categorical feature list |
| `model.py` | CatBoost binary classifier with configurable iterations, depth, learning_rate, l2_leaf_reg |
| `evaluate.py` | Stratified K-fold CV with accuracy (and F1); early stopping per fold |
| `hyperparameter_tuning.py` | Grid search over depth, learning_rate, l2_leaf_reg |
| `train.py` | End-to-end: load → clean → add_features → validate (optional) → retrain on full train → save model and artifacts |
| `predict.py` | Load model and artifacts, preprocess test data, predict, write submission CSV |

**Training flow:**  
`load_train()` → `clean()` → `add_features()` → `split_X_y()` → (optional) train/val split and CV → `build_model()` → `fit()` with `cat_features` and `eval_set` → save `model.cbm` and `artifacts.pkl`.

**Inference flow:**  
Load model and artifacts → `load_test()` → `clean()` → `add_features()` → `split_X_y()` (no target) → align columns → `model.predict()` → write `submissions/submission.csv`.

---

## 3. Preprocessing and Feature Engineering

- **NGS missing codes:** 6 (valid skip), 9 / 99 (refused / not stated) are replaced with `NaN` in numeric columns, then treated as a single `"missing"` category in categorical encoding.
- **Mixed-type columns:** GENDER2, DDIS_FL, VISBMINP sometimes contain text (e.g. "Female", "With disability"). These are normalized to numeric codes, then all code columns are converted to string and NaN is mapped to `"missing"` so CatBoost can use them as categoricals natively.
- **No scaling:** Tree-based CatBoost does not require feature scaling.
- **No one-hot encoding:** CatBoost accepts categorical indices; we pass `cat_features=cat_indices` and keep columns as object/string.

---

## 4. Model: CatBoost

- **Algorithm:** CatBoostClassifier (gradient boosting on decision trees with ordered target encoding for categoricals).
- **Loss:** Log loss (default); evaluation metric: Accuracy.
- **Key hyperparameters:** `iterations=500`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3.0`, `early_stopping_rounds=20`.
- **Tuning:** Optional grid search over `depth`, `learning_rate`, `l2_leaf_reg` via `src/hyperparameter_tuning.py` and notebook `03_catboost_training_tuning.ipynb`.

---

## 5. Validation and Iteration

- **Stratified train/val split** (e.g. 80/20) for a single validation accuracy.
- **Stratified K-fold cross-validation** (e.g. 5 folds) for mean and std accuracy across folds.
- **Early stopping** on the validation set to reduce overfitting.
- Iteration on the model was driven by validation feedback and leaderboard results (Public/Private), focusing on both predictive performance and interpretability.

---

## 6. Interpretability

- **Feature importance:** CatBoost built-in (e.g. PredictionValuesChange) used to rank features; see notebook `04_evaluation_interpretability.ipynb`.
- **SHAP:** Optional SHAP (TreeExplainer) for global and instance-level explanations when `shap` is installed.

---

## 7. Results Summary

| Metric | Value |
|--------|--------|
| Public leaderboard accuracy | 0.75174 |
| Private leaderboard accuracy | 0.70511 |
| Validation accuracy (example run) | ~0.66–0.75 (depends on split and tuning) |
| CV accuracy (5-fold) | ~0.67 ± 0.01 (example) |

The gap between Public and Private accuracy indicates some overfitting to the public test set; regularization (l2_leaf_reg, depth, early stopping) and CV help improve generalization.

---

## 8. Reproducibility

- **Data:** Place `train.csv` and `test.csv` in `data/raw/`.
- **Environment:** `pip install -r requirements.txt` (pandas, numpy, scikit-learn, catboost, matplotlib, seaborn).
- **Train:** `python3 -m src.train` (optionally with different hyperparameters inside `train.py` or via notebooks).
- **Predict:** `python3 -m src.predict` → generates `submissions/submission.csv`.

Random seeds are fixed (`RANDOM_STATE=42`) in config for reproducible splits and model training.

---

## 9. References

- NGS (National Graduate Survey) structured hiring dataset — SFU Data Science ML Hackathon.
- CatBoost documentation: [catboost.ai](https://catboost.ai).
- Hackathon theme: overqualification prediction in recruitment.
