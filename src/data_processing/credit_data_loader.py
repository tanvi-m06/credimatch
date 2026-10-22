"""
Data loader module for CrediMatch AI.
Optimized for memory efficiency, chunked streaming, and feature engineering.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    HOME_CREDIT_PATH,
    SEGMENTATION_PATH,
    CATALOGUE_CSV_PATH,
    CATALOGUE_XLSX_PATH,
    HOME_CREDIT_SAMPLE_SIZE,
    HOME_CREDIT_RANDOM_STATE
)

# Core features for the supervised Home Credit risk model
HOME_CREDIT_FEATURES = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'CNT_CHILDREN',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

# Core features for unsupervised customer segmentation
SEGMENTATION_FEATURES = [
    'total_trans_amount',
    'total_trans_count',
    'avg_utilization_ratio',
    'credit_limit',
    'estimated_income'
]

def load_home_credit_data(sample_size: int = HOME_CREDIT_SAMPLE_SIZE) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads Home Credit dataset using column filtering and row sampling
    to avoid reading the entire 166MB file into memory simultaneously.
    """
    if not HOME_CREDIT_PATH.exists():
        raise FileNotFoundError(f"Home credit dataset not found at {HOME_CREDIT_PATH}")

    use_cols = ['TARGET'] + HOME_CREDIT_FEATURES
    
    # Read sample efficiently
    df = pd.read_csv(
        HOME_CREDIT_PATH,
        usecols=use_cols,
        nrows=sample_size,
        low_memory=False
    )
    
    # Feature Engineering
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).round(1)
    # DAYS_EMPLOYED has positive anomalous values (365243) for unemployed/retired; clip them
    employed_clean = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else -x)
    df['YEARS_EMPLOYED'] = (employed_clean / 365.25).round(1)
    
    income_safe = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['DEBT_TO_INCOME'] = (df['AMT_CREDIT'] / income_safe).clip(upper=20.0).round(3)
    df['ANNUITY_TO_INCOME'] = (df['AMT_ANNUITY'] / income_safe).clip(upper=5.0).round(3)
    
    engineered_cols = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'YEARS_EMPLOYED',
        'CNT_CHILDREN', 'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    
    X = df[engineered_cols]
    y = df['TARGET']
    return X, y

def load_segmentation_data() -> pd.DataFrame:
    """Loads customer segmentation dataset and derives average ticket size."""
    if not SEGMENTATION_PATH.exists():
        raise FileNotFoundError(f"Customer segmentation dataset not found at {SEGMENTATION_PATH}")
    
    df = pd.read_csv(SEGMENTATION_PATH)
    df['avg_ticket_size'] = (df['total_trans_amount'] / df['total_trans_count'].replace(0, 1)).round(2)
    return df

def load_card_catalogue() -> pd.DataFrame:
    """Loads and standardizes the credit card product catalogue."""
    if CATALOGUE_CSV_PATH.exists():
        df = pd.read_csv(CATALOGUE_CSV_PATH)
    elif CATALOGUE_XLSX_PATH.exists():
        df = pd.read_excel(CATALOGUE_XLSX_PATH)
    else:
        raise FileNotFoundError("Card catalogue not found in CSV or XLSX format.")
    
    # Fill defaults for essential numerical calculations
    defaults = {
        'joining_fee_inr': 0.0,
        'renewal_fee_inr': 0.0,
        'waiver_threshold_inr': np.nan,
        'base_reward_pct': 1.0,
        'max_reward_pct': 2.0,
        'domestic_lounge_annual': 0.0,
        'intl_lounge_annual': 0.0,
        'forex_markup_pct': 3.5,
        'monthly_interest_pct': 3.5,
        'is_lifetime_free': False
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            if col == 'is_lifetime_free':
                df[col] = df[col].fillna(False).astype(bool)
            elif col != 'waiver_threshold_inr':
                df[col] = df[col].fillna(val)
                
    return df
