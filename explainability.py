"""
================================================================================
CrediMatch AI - Model Explainability Module (SHAP & Recommendation Insights)
================================================================================
Author: Senior ML Engineer
Target Audience: Third-Year IT Project / Viva Examination
Description:
    Provides transparent explainability for both:
    1. The supervised XGBoost credit risk prediction using SHAP (TreeExplainer).
    2. The card recommendation ranking using itemized financial reward attribution.
    
Key Rule:
    Strictly distinguishes genuine SHAP feature attributions from deterministic
    card reward breakdowns.
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "credit_risk_xgb.joblib"
IMPUTER_FILE = MODELS_DIR / "risk_imputer.joblib"

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

FEATURE_LABELS = {
    'AMT_INCOME_TOTAL': 'Annual Income',
    'AMT_CREDIT': 'Requested Credit Limit',
    'AMT_ANNUITY': 'Annual Repayment Burden',
    'AGE_YEARS': 'Applicant Age',
    'YEARS_EMPLOYED': 'Employment Experience (Years)',
    'CNT_CHILDREN': 'Number of Children/Dependents',
    'DEBT_TO_INCOME': 'Debt-to-Income Ratio (DTI)',
    'ANNUITY_TO_INCOME': 'Annuity-to-Income Ratio',
    'EXT_SOURCE_1': 'Bureau Score Source 1',
    'EXT_SOURCE_2': 'Bureau Score Source 2',
    'EXT_SOURCE_3': 'Bureau Score Source 3'
}

class RiskExplainer:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.explainer = None
        self._load()
        
    def _load(self):
        if MODEL_FILE.exists() and IMPUTER_FILE.exists():
            self.model = joblib.load(MODEL_FILE)
            self.imputer = joblib.load(IMPUTER_FILE)
            # TreeExplainer is lightning fast for XGBoost trees
            self.explainer = shap.TreeExplainer(self.model)
            
    def explain_applicant(self, applicant_dict):
        """
        Calculates exact SHAP values for an applicant input.
        Returns positive and negative risk contributors.
        """
        if self.model is None or self.explainer is None:
            return {"status": "error", "message": "Model not loaded"}
            
        # Build dataframe row
        income = float(applicant_dict.get('annual_income', 600000.0))
        credit = float(applicant_dict.get('requested_credit_limit', 150000.0))
        annuity = credit * 0.12
        age = float(applicant_dict.get('age', 32.0))
        years_emp = float(applicant_dict.get('years_employed', 4.0))
        children = int(applicant_dict.get('children', 0))
        ext1 = float(applicant_dict.get('ext_score_1', 0.50))
        ext2 = float(applicant_dict.get('ext_score_2', 0.55))
        ext3 = float(applicant_dict.get('ext_score_3', 0.52))
        
        dti = round(credit / max(1000.0, income), 3)
        ati = round(annuity / max(1000.0, income), 3)
        
        row = {
            'AMT_INCOME_TOTAL': income,
            'AMT_CREDIT': credit,
            'AMT_ANNUITY': annuity,
            'AGE_YEARS': age,
            'YEARS_EMPLOYED': years_emp,
            'CNT_CHILDREN': children,
            'DEBT_TO_INCOME': dti,
            'ANNUITY_TO_INCOME': ati,
            'EXT_SOURCE_1': ext1,
            'EXT_SOURCE_2': ext2,
            'EXT_SOURCE_3': ext3
        }
        
        df = pd.DataFrame([row], columns=FEATURE_NAMES)
        df_imp = pd.DataFrame(self.imputer.transform(df), columns=FEATURE_NAMES)
        
        # Calculate SHAP values
        shap_values = self.explainer(df_imp)
        values = shap_values.values[0] # 1D array of feature attributions
        
        results = []
        for feat, val, raw_val in zip(FEATURE_NAMES, values, df_imp.iloc[0]):
            human_name = FEATURE_LABELS.get(feat, feat)
            impact = "Increases Risk" if val > 0 else "Decreases Risk"
            results.append({
                'feature': feat,
                'feature_label': human_name,
                'raw_value': round(float(raw_val), 2),
                'shap_value': round(float(val), 4),
                'impact': impact,
                'abs_shap': abs(float(val))
            })
            
        results.sort(key=lambda x: x['abs_shap'], reverse=True)
        return {
            'status': 'success',
            'base_value': round(float(self.explainer.expected_value), 4),
            'top_drivers': results[:6],
            'all_drivers': results
        }

def explain_card_recommendation(card_result, user_spending):
    """
    Generates human-understandable breakdown of why a card was recommended.
    (Distinct from SHAP, based on transparent financial reward calculations).
    """
    reasons = []
    
    # Check top reward category contribution
    cat_breakdown = card_result.get('category_rewards_inr', {})
    if cat_breakdown:
        top_earning_cat = max(cat_breakdown.items(), key=lambda x: x[1])
        if top_earning_cat[1] > 0:
            reasons.append(
                f"Highest reward driver: **{top_earning_cat[0].replace('_', ' ').title()}** earning ₹{top_earning_cat[1]:,.0f} cashback/rewards."
            )
            
    # Check lounge perks
    lounge_val = card_result.get('lounge_benefit_value_inr', 0.0)
    if lounge_val > 0:
        reasons.append(
            f"Complimentary lounge access provides an estimated **₹{lounge_val:,.0f}** annual travel comfort value."
        )
        
    # Check fee waiver
    if card_result.get('is_waived', False):
        reasons.append(
            f"Annual fee of ₹{card_result['annual_fee']:,.0f} is **100% waived** because your spend exceeds the waiver threshold."
        )
    elif card_result.get('annual_fee', 0) == 0:
        reasons.append("**Lifetime Free (LTF)** card with zero maintenance cost.")
        
    # Welcome bonus
    welcome = card_result.get('welcome_bonus_value', 0.0)
    if welcome > 0:
        reasons.append(f"Includes joining welcome vouchers/bonus worth **₹{welcome:,.0f}**.")
        
    return reasons
