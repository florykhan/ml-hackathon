from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
@dataclass(frozen=True)
class ModelConfig:
    target_col: str = "overqualified"
    id_col: str = "id"
    missing_codes: Tuple[int, ...] = (6, 9, 99)
    numeric_cols: Tuple[str, ...] = ("BEF_160",)
    max_iter: int = 3000
    class_weight: str = "balanced"
    random_state: Optional[int] = None
def apply_survey_missing_codes(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """
    Replace survey special codes (e.g., 6/9/99) with np.nan.
    Important: do this BEFORE preprocessing/encoding.
    """
    return df.replace(list(config.missing_codes), np.nan)
def infer_feature_columns(X: pd.DataFrame, config: ModelConfig) -> Tuple[List[str], List[str]]:
    """
    Decide numeric vs categorical features.
    Rule:
    - If column is in config.numeric_cols and exists -> numeric.
    - Everything else -> categorical (even if it looks numeric).
    """
    numeric_cols = [c for c in config.numeric_cols if c in X.columns]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols
def build_pipeline(X: pd.DataFrame, config: ModelConfig) -> Pipeline:
    """
    Build a leakage-safe sklearn Pipeline:
      preprocess (impute + onehot) -> model
    No fitting happens here.
    """
    numeric_cols, categorical_cols = infer_feature_columns(X, config)
    transformers = []
    if numeric_cols:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", numeric_transformer, numeric_cols))
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(
        max_iter=config.max_iter,
        class_weight=config.class_weight,
    )
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipeline
def split_xy(train_df: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    From a labeled dataframe, return X (features) and y (target).
    Drops id and target from X.
    """
    if config.target_col not in train_df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found.")
    if config.id_col not in train_df.columns:
        raise ValueError(f"ID column '{config.id_col}' not found.")

    y = train_df[config.target_col].astype(int)
    X = train_df.drop(columns=[config.target_col, config.id_col]).copy()
    return X, y


def split_test_x(test_df: pd.DataFrame, config: ModelConfig) -> Tuple[pd.Series, pd.DataFrame]:
    """
    From test dataframe, return (ids, X_test) where X_test excludes id column.
    """
    if config.id_col not in test_df.columns:
        raise ValueError(f"ID column '{config.id_col}' not found in test set.")
    ids = test_df[config.id_col].copy()
    X_test = test_df.drop(columns=[config.id_col]).copy()
    return ids, X_test
