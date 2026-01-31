# Import required libraries
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

# =============================================================================
# FEATURE DEFINITIONS BASED ON NGS 2020 PUMF DATA DICTIONARY
# =============================================================================
# Careful analysis of the data dictionary reveals:
# - Most features are CATEGORICAL (including ordinal ranges)
# - Only BEF_160 (months of work experience) is truly CONTINUOUS
# - Different features have different special codes

# Missing value encoding for CatBoost
MISSING_VALUE_CAT = "MISSING"      # Not stated / unknown (code 9 or 99)
VALID_SKIP_VALUE_CAT = "SKIP"      # Valid skip (code 6)
MISSING_VALUE_NUM = np.nan         # For numeric features - CatBoost handles natively

# =============================================================================
# CATEGORICAL FEATURES
# =============================================================================
# Based on data dictionary, almost ALL features are categorical.
# Even ordinal ranges (debt, scholarships, age) should be treated as categorical
# because the numeric values don't represent actual quantities.

CATEGORICAL_FEATURES = [
    # Education & Program Features
    'CERTLEVP',      # 2020 Program Level (1=College, 2=Bachelor, 3=Master+, 9=NS)
    'PGMCIPAP',      # Field of Study CIP code (1-23, 99=NS) - HIGH CARDINALITY categorical
    'PGM_P034',      # Full-time/Part-time (1=FT, 2=PT, 6=skip, 9=NS)
    'PGM_P036',      # Reason for part-time (1-5, 6=skip, 9=NS)
    'HLOSGRDP',      # Highest edu at graduation (1-7, 9=NS) - NO code 6!
    'PREVLEVP',      # Edu before 2020 program (1-4, 9=NS) - NO code 6!
    
    # Program Experience Features (Entrepreneurial)
    'PGM_280A',      # Started business (1=Y, 2=N, 6=skip, 9=NS)
    'PGM_280B',      # Completed courses (1=Y, 2=N, 6=skip, 9=NS)
    'PGM_280C',      # Pitch/business plan (1=Y, 2=N, 6=skip, 9=NS)
    'PGM_280F',      # None of above (1=Y, 2=N, 6=skip, 9=NS)
    'PGM_P401',      # Online education (1=All, 2=Some, 3=None, 6=skip, 9=NS)
    
    # Financial Features (ORDINAL RANGES - treat as categorical)
    'STULOANS',      # Gov student loans (1=Y, 2=N, 6=skip, 9=NS)
    'DBTOTGRD',      # Non-gov debt RANGES (0-5, 6=skip, 9=NS) - NOT continuous!
    'SCHOLARP',      # Scholarship RANGES (0-5, 6=skip, 9=NS) - NOT continuous!
    
    # Demographic Features
    'GRADAGEP',      # Age at graduation GROUPS (1-5) - NO special codes per dict
    'GENDER2',       # Gender (1=M, 2=F) - NO special codes per dict
    'CTZSHIPP',      # Citizenship (1-3) - NO special codes per dict
    'VISBMINP',      # Visible minority (1=Y, 2=N, 9=NS) - NO code 6!
    'DDIS_FL',       # Disability (1=Y, 2=N) - NO special codes per dict
    
    # Parental Education
    'PAR1GRD',       # Parent 1 edu comparison (1-3, 6=skip, 9=NS)
    'PAR2GRD',       # Parent 2 edu comparison (1-3, 6=skip, 9=NS)
    
    # Pre-Program Activity
    'BEF_P140',      # Main activity before (1-3, 9=NS) - NO code 6!
]

# =============================================================================
# NUMERIC FEATURES
# =============================================================================
# Only truly continuous feature - actual count of months
NUMERIC_FEATURES = [
    'BEF_160',       # Months of work experience (0-97 actual months, 99=NS)
]

# =============================================================================
# SPECIAL CODE MAPPINGS BY FEATURE
# =============================================================================
# Critical: Not all features have the same special codes!
# NOTE: Actual data may have additional codes not in documentation (e.g., 0, 6, 9)
#       We handle these as MISSING since they likely represent unknown/undefined

# Features where 6 = Valid Skip (NOT a valid data value)
# IMPORTANT: HLOSGRDP uses 6 as a VALID value (Master's degree), not skip!
FEATURES_WITH_CODE_6 = {
    'PGM_P034', 'PGM_P036',
    'PGM_280A', 'PGM_280B', 'PGM_280C', 'PGM_280F',
    'PGM_P401',
    'STULOANS', 'DBTOTGRD', 'SCHOLARP',
    'PAR1GRD', 'PAR2GRD',
    'PREVLEVP',  # Data shows code 6 exists
    'GRADAGEP',  # Data shows code 6 exists
    'BEF_160',   # Data shows code 6 exists
    # Note: HLOSGRDP is NOT here - 6 = Master's degree (valid value)
}

# Features where 9 = Not Stated
FEATURES_WITH_CODE_9 = {
    'CERTLEVP', 'PGM_P034', 'PGM_P036',
    'HLOSGRDP', 'PREVLEVP',
    'PGM_280A', 'PGM_280B', 'PGM_280C', 'PGM_280F',
    'PGM_P401',
    'STULOANS', 'DBTOTGRD', 'SCHOLARP',
    'VISBMINP',
    'PAR1GRD', 'PAR2GRD',
    'BEF_P140',
    # Additional based on actual data analysis:
    'GRADAGEP', 'GENDER2', 'CTZSHIPP',
}

# Features where 99 = Not Stated (continuous-style coding)
FEATURES_WITH_CODE_99 = {
    'PGMCIPAP',  # Field of study (1-23, 99=NS)
    'BEF_160',   # Work experience months (0-97, 99=NS)
}

# Features with NO special codes (data dictionary shows complete categories)
FEATURES_NO_SPECIAL_CODES = {
    'GRADAGEP',  # Age groups 1-5 only
    'GENDER2',   # Gender 1,2 only
    'CTZSHIPP',  # Citizenship 1,2,3 only
    'DDIS_FL',   # Disability 1,2 only
}


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean the dataset according to NGS 2020 PUMF Data Dictionary.
    
    Handles:
    1. Mixed-type columns (string/numeric) conversion
    2. Special codes based on feature-specific rules:
       - Code 6 (Valid Skip): Only for features that have it
       - Code 9 (Not Stated): For most categorical features
       - Code 99 (Not Stated): For PGMCIPAP and BEF_160
    3. Creates clean versions with appropriate sentinel values
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data (for logging)
    
    Returns:
        Cleaned dataframe with _clean suffix columns
    """
    df_clean = df.copy()
    
    # ==========================================================================
    # STEP 1: Handle mixed-type columns (string values in numeric columns)
    # ==========================================================================
    # These columns may have string values that need mapping to numeric
    string_mappings = {
        'GENDER2': {'Male': 1.0, 'Female': 2.0},
        'VISBMINP': {'Yes': 1.0, 'No': 2.0},
        'DDIS_FL': {'With disability': 1.0, 'Without disability': 2.0}
    }
    
    for col, mapping in string_mappings.items():
        if col in df_clean.columns:
            # Check if there are string values
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].replace(mapping)
    
    # ==========================================================================
    # STEP 2: Get feature columns (exclude id and target)
    # ==========================================================================
    feature_cols = [col for col in df_clean.columns if col not in ['id', 'overqualified']]
    
    # Convert all feature columns to numeric first
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # ==========================================================================
    # STEP 3: Create cleaned versions with feature-specific special code handling
    # ==========================================================================
    for col in feature_cols:
        clean_col_name = f"{col}_clean"
        df_clean[clean_col_name] = df_clean[col].copy()
        
        if col in NUMERIC_FEATURES:
            # ===== NUMERIC FEATURES: Use NaN for missing =====
            # BEF_160: 0-97 valid, 99 = not stated
            if col == 'BEF_160':
                df_clean.loc[df_clean[clean_col_name] == 99, clean_col_name] = MISSING_VALUE_NUM
            # Keep existing NaN as NaN (CatBoost handles natively)
            
        else:
            # ===== CATEGORICAL FEATURES: Use string sentinels =====
            # Convert to string to allow string sentinel values
            df_clean[clean_col_name] = df_clean[clean_col_name].astype(str)
            
            # Handle code 99 (for PGMCIPAP - field of study)
            if col in FEATURES_WITH_CODE_99:
                df_clean.loc[df_clean[col] == 99, clean_col_name] = MISSING_VALUE_CAT
            
            # Handle code 9 (not stated) - only for features that use it
            if col in FEATURES_WITH_CODE_9:
                df_clean.loc[df_clean[col] == 9, clean_col_name] = MISSING_VALUE_CAT
            
            # Handle code 6 (valid skip) - only for features that use it
            if col in FEATURES_WITH_CODE_6:
                df_clean.loc[df_clean[col] == 6, clean_col_name] = VALID_SKIP_VALUE_CAT
            
            # Handle any NaN values that existed in original data
            df_clean.loc[df_clean[col].isna(), clean_col_name] = MISSING_VALUE_CAT
            
            # Clean up "nan" strings that might have been created during conversion
            df_clean.loc[df_clean[clean_col_name] == 'nan', clean_col_name] = MISSING_VALUE_CAT
            df_clean.loc[df_clean[clean_col_name] == 'None', clean_col_name] = MISSING_VALUE_CAT
    
    return df_clean


def get_catboost_feature_indices(feature_names: List[str]) -> List[int]:
    """
    Get indices of categorical features for CatBoost's cat_features parameter.
    
    CatBoost handles categorical features specially - it uses ordered target 
    statistics instead of one-hot encoding, which is more efficient and often
    more effective for high-cardinality features like PGMCIPAP.
    
    Args:
        feature_names: List of feature column names
    
    Returns:
        List of indices for categorical features
    """
    cat_indices = []
    for idx, name in enumerate(feature_names):
        # All features except BEF_160 are categorical
        if name in CATEGORICAL_FEATURES:
            cat_indices.append(idx)
    return cat_indices


def prepare_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for CatBoost by ensuring correct data types.
    
    For CatBoost:
    - Categorical features: keep as string (handles high cardinality well)
    - Numeric features: keep as float with NaN (native NaN handling)
    
    Args:
        df: Cleaned dataframe
    
    Returns:
        Dataframe ready for CatBoost
    """
    df_cb = df.copy()
    
    for col in df_cb.columns:
        if col in NUMERIC_FEATURES:
            # Numeric: keep as float (NaN preserved for CatBoost native handling)
            df_cb[col] = pd.to_numeric(df_cb[col], errors='coerce')
        else:
            # Categorical: ensure string type
            df_cb[col] = df_cb[col].astype(str)
    
    return df_cb


def preprocess_dataset(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Complete preprocessing pipeline for train and test datasets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        verbose: Whether to print progress
    
    Returns:
        Tuple of (train_features, test_features, categorical_indices)
    """
    # Clean both datasets
    train_clean = clean_data(train_df, is_train=True)
    test_clean = clean_data(test_df, is_train=False)
    
    if verbose:
        print(f"Training set shape after cleaning: {train_clean.shape}")
        print(f"Test set shape after cleaning: {test_clean.shape}")
    
    # Extract clean columns only
    clean_cols = [col for col in train_clean.columns if col.endswith('_clean')]
    train_final = train_clean[clean_cols].copy()
    test_final = test_clean[clean_cols].copy()
    
    # Remove _clean suffix from column names
    train_final.columns = [col.replace('_clean', '') for col in clean_cols]
    test_final.columns = [col.replace('_clean', '') for col in clean_cols]
    
    # Prepare for CatBoost (correct data types)
    train_final = prepare_for_catboost(train_final)
    test_final = prepare_for_catboost(test_final)
    
    # Add target column to train dataframe (if it exists)
    if 'overqualified' in train_df.columns:
        train_final['overqualified'] = train_df['overqualified'].values
    
    # Get categorical feature indices for CatBoost
    cat_indices = get_catboost_feature_indices(train_final.columns.tolist())
    
    if verbose:
        print(f"\nFinal training shape: {train_final.shape}")
        print(f"Final test shape: {test_final.shape}")
        
        # List categorical vs numeric features
        cat_feature_names = [train_final.columns[i] for i in cat_indices]
        num_feature_names = [c for c in train_final.columns if c in NUMERIC_FEATURES]
        print(f"\nCategorical features ({len(cat_indices)}): {cat_feature_names}")
        print(f"Numeric features ({len(num_feature_names)}): {num_feature_names}")
        
        # Report missing value summary
        print("\n" + "="*60)
        print("MISSING VALUE SUMMARY:")
        print("="*60)
        
        for col in train_final.columns:
            if col in NUMERIC_FEATURES:
                # Numeric: count NaN
                nan_count = train_final[col].isna().sum()
                pct = nan_count / len(train_final) * 100
                if nan_count > 0:
                    print(f"  {col}: {nan_count} NaN ({pct:.1f}%) - CatBoost native handling")
            else:
                # Categorical: count string sentinels
                missing_count = (train_final[col] == MISSING_VALUE_CAT).sum()
                skip_count = (train_final[col] == VALID_SKIP_VALUE_CAT).sum()
                missing_pct = missing_count / len(train_final) * 100
                skip_pct = skip_count / len(train_final) * 100
                if missing_count > 0 or skip_count > 0:
                    parts = []
                    if missing_count > 0:
                        parts.append(f"{missing_count} MISSING ({missing_pct:.1f}%)")
                    if skip_count > 0:
                        parts.append(f"{skip_count} SKIP ({skip_pct:.1f}%)")
                    print(f"  {col}: {', '.join(parts)}")
    
    return train_final, test_final, cat_indices
