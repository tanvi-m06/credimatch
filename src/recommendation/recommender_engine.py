"""
CrediMatch Recommendation Engine.
Integrates Risk Gating (Home Credit XGBoost), Persona Tailoring (K-Means),
and Deterministic Expected Annual Value (EAV) Card Ranking (311-Card Catalogue).
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import (
    DOMESTIC_LOUNGE_VALUE_INR,
    INTERNATIONAL_LOUNGE_VALUE_INR
)
from src.data_processing.credit_data_loader import load_card_catalogue

class CrediMatchRecommender:
    def __init__(self):
        self.catalogue = load_card_catalogue()
        
    def recommend(
        self,
        risk_profile: Dict[str, Any],
        persona_profile: Dict[str, Any],
        spending_profile: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates Top-N card recommendations with full mathematical valuation.
        """
        df = self.catalogue.copy()
        
        # User spend parameters
        annual_spend = float(spending_profile.get('annual_spend', 300000.0))
        dining_spend = float(spending_profile.get('dining_spend', 50000.0))
        travel_spend = float(spending_profile.get('travel_spend', 60000.0))
        grocery_spend = float(spending_profile.get('grocery_spend', 60000.0))
        intl_spend = float(spending_profile.get('intl_spend', 30000.0))
        
        user_dom_lounge = int(spending_profile.get('domestic_lounge_visits', 4))
        user_intl_lounge = int(spending_profile.get('intl_lounge_visits', 1))
        
        risk_tier = risk_profile.get('risk_tier', 'Moderate Risk (Near-Prime)')
        max_limit = float(risk_profile.get('max_recommended_limit', 150000.0))
        approval_status = risk_profile.get('approval_status', 'Approved')
        
        # 1. RISK ELIGIBILITY GATING
        # If user is high risk / declined for prime, restrict to lifetime free or low fee cards
        if "High Risk" in risk_tier or "Declined" in approval_status:
            # Filter strictly for entry-level / lifetime free / low maintenance cards
            df = df[(df['is_lifetime_free'] == True) | (df['renewal_fee_inr'] <= 500.0)]
        elif "Moderate Risk" in risk_tier:
            # Filter out ultra-luxury cards with high annual fee (> ₹5,000)
            df = df[df['renewal_fee_inr'] <= 5000.0]
        # Low risk (Prime) has access to the full 311-card universe
        
        results = []
        for _, card in df.iterrows():
            card_name = str(card.get('Card Name', 'Credit Card'))
            bank_name = str(card.get('Bank Name', 'Bank'))
            network = str(card.get('Network', 'Visa/Mastercard'))
            
            joining_fee = float(card.get('joining_fee_inr', 0.0))
            renewal_fee = float(card.get('renewal_fee_inr', 0.0))
            waiver_threshold = card.get('waiver_threshold_inr', np.nan)
            is_ltf = bool(card.get('is_lifetime_free', False))
            
            base_reward_pct = float(card.get('base_reward_pct', 1.0))
            max_reward_pct = float(card.get('max_reward_pct', 2.0))
            card_dom_lounge = float(card.get('domestic_lounge_annual', 0.0))
            card_intl_lounge = float(card.get('intl_lounge_annual', 0.0))
            forex_pct = float(card.get('forex_markup_pct', 3.5))
            
            # --- CALCULATE EXPECTED ANNUAL VALUE (EAV) ---
            # 1. Base Rewards
            base_reward_inr = annual_spend * (base_reward_pct / 100.0)
            
            # 2. Category Bonus (Accelerated reward on user's highest spending areas)
            # Cards with higher max_reward_pct earn accelerated points on dining/travel/shopping
            accelerated_spend = dining_spend + travel_spend
            category_bonus_pct = max(0.0, (max_reward_pct - base_reward_pct) * 0.60)
            category_bonus_inr = accelerated_spend * (category_bonus_pct / 100.0)
            
            total_rewards_earned = base_reward_inr + category_bonus_inr
            
            # 3. Lounge Benefit Value
            effective_dom_lounge = min(user_dom_lounge, int(card_dom_lounge))
            effective_intl_lounge = min(user_intl_lounge, int(card_intl_lounge))
            lounge_benefit_inr = (effective_dom_lounge * DOMESTIC_LOUNGE_VALUE_INR) + (effective_intl_lounge * INTERNATIONAL_LOUNGE_VALUE_INR)
            
            # 4. Fee & Spend-Based Waiver
            is_waived = False
            if is_ltf:
                net_renewal_fee = 0.0
                fee_status = "Lifetime Free (LTF)"
            elif pd.notna(waiver_threshold) and waiver_threshold > 0 and annual_spend >= waiver_threshold:
                net_renewal_fee = 0.0
                is_waived = True
                fee_status = f"Waived (Spend ₹{annual_spend:,.0f} >= ₹{waiver_threshold:,.0f})"
            else:
                net_renewal_fee = renewal_fee
                fee_status = f"₹{renewal_fee:,.0f} / year"
                
            # 5. Forex Markup Cost (vs standard 3.5% benchmark savings)
            forex_cost_inr = intl_spend * (forex_pct / 100.0)
            # Forex savings if card has lower markup than standard 3.5%
            forex_savings_inr = max(0.0, intl_spend * ((3.5 - forex_pct) / 100.0))
            
            # 6. NET ANNUAL BENEFIT (INR)
            net_benefit = (total_rewards_earned + lounge_benefit_inr + forex_savings_inr) - net_renewal_fee
            
            # Persona Alignment Bonus
            persona_name = persona_profile.get('persona_name', '')
            alignment_score = 0.0
            if "Power Spender" in persona_name and max_reward_pct >= 3.0:
                alignment_score += 1200.0
            elif "Affluent" in persona_name and (card_intl_lounge > 0 or forex_pct <= 2.0):
                alignment_score += 1500.0
            elif "Budget Conscious" in persona_name and (is_ltf or net_renewal_fee == 0):
                alignment_score += 1000.0
            elif "Revolver" in persona_name and float(card.get('monthly_interest_pct', 3.5)) < 3.0:
                alignment_score += 1000.0
                
            composite_utility = net_benefit + alignment_score
            
            # Determine Badge
            if is_ltf or net_renewal_fee == 0:
                badge = "Zero Annual Fee / Waived"
            elif card_intl_lounge > 0 or card_dom_lounge >= 8:
                badge = "Top Lounge & Travel Card"
            elif max_reward_pct >= 4.0:
                badge = "Maximum Reward Multiplier"
            elif net_benefit > 10000:
                badge = "High Yield Value Card"
            else:
                badge = "Everyday Cashback"
                
            results.append({
                'card_name': card_name,
                'bank_name': bank_name,
                'network': network,
                'joining_fee': joining_fee,
                'renewal_fee': renewal_fee,
                'is_lifetime_free': is_ltf,
                'fee_status': fee_status,
                'base_reward_pct': base_reward_pct,
                'max_reward_pct': max_reward_pct,
                'domestic_lounge': int(card_dom_lounge),
                'intl_lounge': int(card_intl_lounge),
                'forex_markup_pct': forex_pct,
                'rewards_earned_inr': round(total_rewards_earned, 1),
                'lounge_value_inr': round(lounge_benefit_inr, 1),
                'forex_savings_inr': round(forex_savings_inr, 1),
                'net_renewal_fee_inr': round(net_renewal_fee, 1),
                'net_annual_benefit_inr': round(net_benefit, 1),
                'composite_score': round(composite_utility, 1),
                'badge': badge,
                'welcome_benefits': str(card.get('Welcome Benefits', 'Standard welcome perks'))[:120]
            })
            
        # Sort by Composite Utility / Net Annual Benefit descending
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        return results[:top_n]
