"""
Credit Risk Model for CrediMatch AI.
Supervised XGBoost classifier trained on Home Credit Default Risk data.
Predicts Probability of Default (PD), maps to Credit Score (300-900),
Risk Tier, Approval Status, and Maximum Credit Limit.
"""

from typing import Dict, Any, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import xgboost as xgb

from src.config import (
    RISK_MODEL_PATH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MEDIUM
)
from src.data_processing.credit_data_loader import load_home_credit_data

FEATURE_NAMES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'YEARS_EMPLOYED',
    'CNT_CHILDREN', 'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]

class CreditRiskPipeline:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.model = None
        self.feature_names = FEATURE_NAMES
        self.metrics = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit imputer
        X_train_imp = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=self.feature_names
        )
        X_test_imp = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=self.feature_names
        )
        
        # Calculate class weight balance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / max(1, pos_count)
        
        self.model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.07,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        self.model.fit(X_train_imp, y_train)
        
        y_pred_proba = self.model.predict_proba(X_test_imp)[:, 1]
        y_pred_binary = (y_pred_proba >= 0.50).astype(int)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_binary, average='binary')
        
        self.metrics = {
            'roc_auc': round(float(auc), 4),
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1_score': round(float(f1), 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
    def predict_risk(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates an individual credit applicant.
        """
        # Feature defaults if applicant data lacks certain scores
        income = float(applicant.get('annual_income', 600000.0))
        requested_credit = float(applicant.get('requested_credit_limit', 150000.0))
        estimated_annuity = requested_credit * 0.12 # approx monthly repayment * 12
        age = float(applicant.get('age', 32.0))
        years_employed = float(applicant.get('years_employed', 4.0))
        children = int(applicant.get('children', 0))
        
        # Normalized external scores (0.0 to 1.0)
        ext1 = float(applicant.get('ext_score_1', 0.50))
        ext2 = float(applicant.get('ext_score_2', 0.55))
        ext3 = float(applicant.get('ext_score_3', 0.52))
        
        dti = round(requested_credit / max(1000.0, income), 3)
        ati = round(estimated_annuity / max(1000.0, income), 3)
        
        row_dict = {
            'AMT_INCOME_TOTAL': income,
            'AMT_CREDIT': requested_credit,
            'AMT_ANNUITY': estimated_annuity,
            'AGE_YEARS': age,
            'YEARS_EMPLOYED': years_employed,
            'CNT_CHILDREN': children,
            'DEBT_TO_INCOME': dti,
            'ANNUITY_TO_INCOME': ati,
            'EXT_SOURCE_1': ext1,
            'EXT_SOURCE_2': ext2,
            'EXT_SOURCE_3': ext3
        }
        
        df_row = pd.DataFrame([row_dict], columns=self.feature_names)
        df_imp = pd.DataFrame(self.imputer.transform(df_row), columns=self.feature_names)
        
        prob_default = float(self.model.predict_proba(df_imp)[0, 1])
        
        # Scale to 300 - 900 CIBIL-like credit score
        # 0.0 PD -> 880 score, 0.5 PD -> 620 score, 1.0 PD -> 320 score
        credit_score = int(np.clip(880 - (prob_default * 560), 300, 900))
        
        # Risk Tiers
        if prob_default < RISK_THRESHOLD_LOW:
            risk_tier = "Low Risk (Prime)"
            approval_status = "Approved"
            limit_multiplier = 0.35 # up to 35% of annual income
        elif prob_default < RISK_THRESHOLD_MEDIUM:
            risk_tier = "Moderate Risk (Near-Prime)"
            approval_status = "Conditional Approval"
            limit_multiplier = 0.20 # up to 20% of annual income
        else:
            risk_tier = "High Risk (Subprime)"
            approval_status = "Declined (Secured Cards Only)"
            limit_multiplier = 0.08 # restricted / secured limit
            
        max_credit_limit = round((income * limit_multiplier) / 5000.0) * 5000
        max_credit_limit = max(25000, min(max_credit_limit, 1500000))
        
        # Feature importance ranking
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'default_probability': round(prob_default, 4),
            'credit_score': credit_score,
            'risk_tier': risk_tier,
            'approval_status': approval_status,
            'max_recommended_limit': max_credit_limit,
            'top_risk_drivers': [k for k, v in sorted_imp[:4]]
        }

def train_or_load_risk_model(force_retrain: bool = False) -> CreditRiskPipeline:
    """Loads existing trained risk model or trains and persists a new pipeline."""
    if RISK_MODEL_PATH.exists() and not force_retrain:
        try:
            pipeline = joblib.load(RISK_MODEL_PATH)
            return pipeline
        except Exception:
            pass
            
    print("Training Supervised XGBoost Credit-Risk Model on Home Credit Data...")
    X, y = load_home_credit_data()
    pipeline = CreditRiskPipeline()
    pipeline.fit(X, y)
    
    # Save model
    RISK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, RISK_MODEL_PATH)
    print(f"XGBoost Model persisted to {RISK_MODEL_PATH}. Test ROC-AUC: {pipeline.metrics.get('roc_auc')}")
    return pipeline
