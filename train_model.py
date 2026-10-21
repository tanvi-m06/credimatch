"""
================================================================================
CrediMatch AI - Supervised Credit Risk Model Training Script
================================================================================
Author: Senior ML Engineer
Target Audience: Third-Year IT Project / Viva Examination
Description:
    Trains a supervised XGBoost classifier on the Home Credit Default Risk dataset
    to estimate applicant Probability of Default (PD).
    
Key Steps:
    1. Memory-efficient data loading (stratified sampling).
    2. Feature engineering (Age, Employment Duration, Debt-to-Income, Annuity-to-Income).
    3. Leak-free preprocessing with SimpleImputer (median).
    4. Stratified Train/Test split (80/20).
    5. Class imbalance handling via scale_pos_weight (~11.4).
    6. Evaluation: ROC-AUC, Precision, Recall, F1, Confusion Matrix.
    7. Artifact persistence for instant inference in Streamlit.
================================================================================
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import xgboost as xgb
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "home_credit" / "application_train.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "credit_risk_xgb.joblib"
IMPUTER_FILE = MODELS_DIR / "risk_imputer.joblib"
METRICS_FILE = MODELS_DIR / "risk_metrics.json"

FEATURE_NAMES = [
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AGE_YEARS',
    'YEARS_EMPLOYED',
    'CNT_CHILDREN',
    'DEBT_TO_INCOME',
    'ANNUITY_TO_INCOME',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

def load_and_preprocess_data(sample_rows=75000):
    """
    Loads Home Credit dataset using column filtering and row sampling.
    Prevents loading the entire 166MB dataset all at once to protect system memory.
    """
    print(f"[*] Reading {sample_rows:,} records from: {DATA_PATH}")
    
    use_cols = [
        'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    
    df = pd.read_csv(DATA_PATH, usecols=use_cols, nrows=sample_rows, low_memory=False)
    
    # Feature Engineering (Converting days to years & calculating financial debt ratios)
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).round(1)
    
    # DAYS_EMPLOYED has positive values (e.g., 365243) representing pensioners/unemployed
    employed_days = df['DAYS_EMPLOYED'].apply(lambda d: 0 if d > 0 else -d)
    df['YEARS_EMPLOYED'] = (employed_days / 365.25).round(1)
    
    # Debt ratios
    income_nonzero = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['DEBT_TO_INCOME'] = (df['AMT_CREDIT'] / income_nonzero).clip(upper=20.0).round(3)
    df['ANNUITY_TO_INCOME'] = (df['AMT_ANNUITY'] / income_nonzero).clip(upper=5.0).round(3)
    
    X = df[FEATURE_NAMES]
    y = df['TARGET']
    
    print(f"[*] Loaded features shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    return X, y

def train_credit_risk_model():
    """
    Trains, evaluates, and serializes the XGBoost credit risk classifier.
    """
    X, y = load_and_preprocess_data()
    
    # 1. Stratified Train-Test Split (80% train, 20% test) to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    
    # 2. Fit Imputer on Train set ONLY (no leakage)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_NAMES)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_NAMES)
    
    # 3. Address class imbalance (default rate ~8%)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    print(f"[*] Imbalance ratio: {scale_pos_weight:.2f} (applied to scale_pos_weight)")
    
    # 4. Initialize and fit XGBoost Classifier
    print("[*] Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.07,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    model.fit(X_train_imp, y_train)
    
    # 5. Evaluate on test set
    y_pred_proba = model.predict_proba(X_test_imp)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.50).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred_binary, zero_division=0)
    rec = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_binary).tolist()
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (Test Set):")
    print("=" * 50)
    print(f"ROC-AUC Score    : {auc:.4f}")
    print(f"Precision Score  : {prec:.4f}")
    print(f"Recall Score     : {rec:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Confusion Matrix : [TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]")
    print("=" * 50)
    
    # 6. Save model, imputer, and metrics
    joblib.dump(model, MODEL_FILE)
    joblib.dump(imputer, IMPUTER_FILE)
    
    feature_importances = dict(zip(FEATURE_NAMES, [float(v) for v in model.feature_importances_]))
    metrics_payload = {
        'roc_auc': round(float(auc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': cm,
        'feature_importances': feature_importances,
        'feature_names': FEATURE_NAMES,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
        
    print(f"[+] Saved model to   : {MODEL_FILE}")
    print(f"[+] Saved imputer to : {IMPUTER_FILE}")
    print(f"[+] Saved metrics to : {METRICS_FILE}")
    return model, imputer, metrics_payload

if __name__ == "__main__":
    train_credit_risk_model()
