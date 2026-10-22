"""
================================================================================
CrediMatch AI - Personalized Card Recommendation & Ranking Engine
================================================================================
Author: Senior ML Engineer
Target Audience: Third-Year IT Project / Viva Examination
Description:
    Implements a hybrid recommendation engine combining:
    1. Supervised Credit Risk suitability signal (XGBoost).
    2. Unsupervised Spending Persona match (K-Means).
    3. Mathematical Expected Net Annual Value (NAV) calculation from credit_cards.csv.
================================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

from explainability import explain_card_recommendation

BASE_DIR = Path(__file__).resolve().parent
CARDS_FILE = BASE_DIR / "data" / "credit_cards.csv"

DOMESTIC_LOUNGE_VAL_INR = 800.0
INTERNATIONAL_LOUNGE_VAL_INR = 2000.0

def load_card_catalogue():
    """Loads credit cards dataset from data/credit_cards.csv."""
    if not CARDS_FILE.exists():
        raise FileNotFoundError(f"Credit card catalogue not found at {CARDS_FILE}")
    df = pd.read_csv(CARDS_FILE)
    return df

def recommend_top_cards(user_financial, user_spending, risk_profile, cluster_profile, top_n=3):
    """
    Ranks credit cards and returns the top N recommendations with full financial transparency.
    """
    cards_df = load_card_catalogue()
    
    # Financial profile
    income = float(user_financial.get('annual_income', 600000.0))
    risk_tier = risk_profile.get('risk_tier', 'Moderate Risk (Near-Prime)')
    prob_default = float(risk_profile.get('default_probability', 0.15))
    
    # Spending profile (annualized)
    groceries = float(user_spending.get('groceries', 60000.0))
    dining = float(user_spending.get('dining', 50000.0))
    travel = float(user_spending.get('travel', 40000.0))
    online = float(user_spending.get('online_shopping', 70000.0))
    fuel = float(user_spending.get('fuel', 30000.0))
    entertainment = float(user_spending.get('entertainment', 20000.0))
    
    total_annual_spend = groceries + dining + travel + online + fuel + entertainment
    
    user_dom_lounge = int(user_spending.get('domestic_lounge_visits', 4))
    user_intl_lounge = int(user_spending.get('intl_lounge_visits', 1))
    
    persona_priority_cats = cluster_profile.get('priority_categories', ['online_shopping', 'groceries'])
    
    ranked_cards = []
    
    for _, card in cards_df.iterrows():
        card_id = int(card['card_id'])
        name = str(card['card_name'])
        bank = str(card['issuing_bank'])
        annual_fee = float(card['annual_fee'])
        waiver_threshold = float(card['fee_waiver_spend'])
        min_income = float(card['min_annual_income'])
        suitable_tier = str(card['suitable_risk_tier'])
        
        # 1. Eligibility & Risk Check
        income_eligible = income >= min_income
        
        # Risk Gating & Suitability:
        # High-risk applicants (subprime / near default) cannot qualify for luxury Prime cards
        if "High Risk" in risk_tier:
            if suitable_tier == "Prime" or annual_fee > 1500:
                continue  # Disqualify from recommendations
        elif "Moderate Risk" in risk_tier:
            if annual_fee > 5000:
                continue  # Disqualify ultra-premium metal cards

                
        # 2. Category-Wise Rewards Estimation
        r_groc = groceries * (float(card['reward_groceries_pct']) / 100.0)
        r_din = dining * (float(card['reward_dining_pct']) / 100.0)
        r_trav = travel * (float(card['reward_travel_pct']) / 100.0)
        r_onl = online * (float(card['reward_online_pct']) / 100.0)
        r_fuel = fuel * (float(card['reward_fuel_pct']) / 100.0)
        r_ent = entertainment * (float(card['reward_entertainment_pct']) / 100.0)
        
        raw_rewards = r_groc + r_din + r_trav + r_onl + r_fuel + r_ent
        
        # 3. Known Cap Deductions
        caps_note = str(card.get('reward_caps_notes', ''))
        capped_rewards = raw_rewards
        if "Max Rs. 1000 cashback per month" in caps_note:
            capped_rewards = min(raw_rewards, 12000.0)
        elif "Max Rs. 5000 cashback per billing cycle" in caps_note:
            capped_rewards = min(raw_rewards, 60000.0)
            
        # 4. Lounge Benefits Valuation
        card_dom = int(card['lounge_visits_domestic'])
        card_intl = int(card['lounge_visits_international'])
        realized_dom = min(user_dom_lounge, card_dom)
        realized_intl = min(user_intl_lounge, card_intl)
        lounge_val = (realized_dom * DOMESTIC_LOUNGE_VAL_INR) + (realized_intl * INTERNATIONAL_LOUNGE_VAL_INR)
        
        # 5. Annual Fee & Spend-Based Waiver
        is_waived = False
        if annual_fee == 0:
            net_fee = 0.0
            fee_display = "Lifetime Free (LTF)"
        elif waiver_threshold > 0 and total_annual_spend >= waiver_threshold:
            net_fee = 0.0
            is_waived = True
            fee_display = f"Waived (Spend ₹{total_annual_spend:,.0f} >= ₹{waiver_threshold:,.0f})"
        else:
            net_fee = annual_fee
            fee_display = f"₹{annual_fee:,.0f} / year"
            
        welcome_bonus = float(card.get('welcome_bonus_value', 0.0))
        
        # 6. NET ANNUAL VALUE (NAV) FORMULA:
        # NAV = Estimated Rewards + Lounge Value + Welcome Bonus - Net Fee
        net_annual_value = (capped_rewards + lounge_val + welcome_bonus) - net_fee
        
        # 7. Spending Profile Fit Score (Boost for cards aligning with K-Means persona)
        category_rewards_map = {
            'groceries': r_groc,
            'dining': r_din,
            'travel': r_trav,
            'online_shopping': r_onl,
            'fuel': r_fuel,
            'entertainment': r_ent
        }
        
        persona_alignment_bonus = 0.0
        for p_cat in persona_priority_cats:
            # Check if card offers above-average rewards in this persona priority category
            col_name = f"reward_{p_cat}_pct"
            if col_name in card and float(card[col_name]) >= 3.0:
                persona_alignment_bonus += 1000.0
                
        # Final Documented Scoring Formula:
        # Ranking Score = Net Annual Value + Persona Alignment Bonus
        ranking_score = net_annual_value + persona_alignment_bonus

        
        card_obj = {
            'card_id': card_id,
            'card_name': name,
            'issuing_bank': bank,
            'annual_fee': annual_fee,
            'fee_display': fee_display,
            'is_waived': is_waived,
            'is_ltf': annual_fee == 0,
            'welcome_bonus_value': welcome_bonus,
            'capped_rewards_inr': round(capped_rewards, 1),
            'lounge_benefit_value_inr': round(lounge_val, 1),
            'net_fee_inr': round(net_fee, 1),
            'net_annual_value_inr': round(net_annual_value, 1),
            'ranking_score': round(ranking_score, 1),
            'income_eligible': income_eligible,
            'category_rewards_inr': {k: round(v, 1) for k, v in category_rewards_map.items()},
            'key_perks': str(card['key_perks']),
            'caps_notes': caps_note
        }
        
        card_obj['reasons'] = explain_card_recommendation(card_obj, user_spending)
        ranked_cards.append(card_obj)
        
    # Sort descending by Ranking Score
    ranked_cards.sort(key=lambda x: x['ranking_score'], reverse=True)
    return ranked_cards[:top_n]
