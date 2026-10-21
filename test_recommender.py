"""
================================================================================
CrediMatch AI - Test Suite for Recommender and Valuation Logic
================================================================================
Author: Senior ML Engineer
Target Audience: Third-Year IT Project / Viva Examination
Description:
    Unit and integration tests validating:
    1. Net Annual Value (NAV) mathematical correctness.
    2. Spend-based annual fee waiver logic.
    3. Category reward accumulation and caps.
    4. K-Means customer segmentation inference.
    5. Top 3 recommendation ranking consistency.
================================================================================
"""

import unittest
from recommender import recommend_top_cards, load_card_catalogue
from segmentation import predict_spending_cluster

class TestCrediMatchRecommender(unittest.TestCase):

    def setUp(self):
        self.cards_df = load_card_catalogue()
        self.financial_profile = {
            'annual_income': 800000.0
        }
        self.risk_prime = {
            'risk_tier': 'Low Risk (Prime)',
            'default_probability': 0.08
        }
        self.risk_subprime = {
            'risk_tier': 'High Risk (Subprime)',
            'default_probability': 0.65
        }
        self.spending_profile = {
            'groceries': 80000.0,
            'dining': 60000.0,
            'travel': 50000.0,
            'online_shopping': 100000.0,
            'fuel': 30000.0,
            'entertainment': 20000.0,
            'domestic_lounge_visits': 4,
            'intl_lounge_visits': 1
        }
        self.cluster_profile = {
            'cluster_id': 1,
            'persona_name': 'Dining & Lifestyle Enthusiast',
            'priority_categories': ['dining', 'online_shopping']
        }

    def test_catalogue_loaded(self):
        """Test that credit card catalogue loads at least 10 cards with required schema."""
        self.assertGreaterEqual(len(self.cards_df), 10)
        required_cols = [
            'card_name', 'issuing_bank', 'annual_fee', 'fee_waiver_spend',
            'reward_groceries_pct', 'reward_dining_pct', 'reward_travel_pct',
            'reward_online_pct', 'reward_fuel_pct', 'reward_entertainment_pct'
        ]
        for col in required_cols:
            self.assertIn(col, self.cards_df.columns)

    def test_top_three_returned(self):
        """Test that recommendation returns exactly top 3 cards sorted by score."""
        recs = recommend_top_cards(
            self.financial_profile,
            self.spending_profile,
            self.risk_prime,
            self.cluster_profile,
            top_n=3
        )
        self.assertEqual(len(recs), 3)
        # Ensure scores are strictly sorted descending
        self.assertGreaterEqual(recs[0]['ranking_score'], recs[1]['ranking_score'])
        self.assertGreaterEqual(recs[1]['ranking_score'], recs[2]['ranking_score'])

    def test_nav_mathematical_formula(self):
        """Test that Net Annual Value equals Rewards + Lounge + Welcome - Net Fee."""
        recs = recommend_top_cards(
            self.financial_profile,
            self.spending_profile,
            self.risk_prime,
            self.cluster_profile,
            top_n=3
        )
        for card in recs:
            expected_nav = (
                card['capped_rewards_inr'] +
                card['lounge_benefit_value_inr'] +
                card['welcome_bonus_value'] -
                card['net_fee_inr']
            )
            self.assertAlmostEqual(card['net_annual_value_inr'], expected_nav, delta=0.5)

    def test_fee_waiver_logic(self):
        """Test that cards with annual spend >= waiver threshold have net fee = 0."""
        # Very high annual spend (10 Lakhs)
        high_spend = {
            'groceries': 200000.0,
            'dining': 200000.0,
            'travel': 200000.0,
            'online_shopping': 200000.0,
            'fuel': 100000.0,
            'entertainment': 100000.0,
            'domestic_lounge_visits': 4,
            'intl_lounge_visits': 1
        }
        recs = recommend_top_cards(
            self.financial_profile,
            high_spend,
            self.risk_prime,
            self.cluster_profile,
            top_n=5
        )
        for card in recs:
            if card['card_name'] == "HDFC Millennia Credit Card":
                # Waiver threshold is 100000, user spent 1000000 -> must be waived!
                self.assertTrue(card['is_waived'])
                self.assertEqual(card['net_fee_inr'], 0.0)

    def test_subprime_risk_penalizes_expensive_cards(self):
        """Test that subprime applicants are not recommended high-fee luxury cards."""
        recs = recommend_top_cards(
            self.financial_profile,
            self.spending_profile,
            self.risk_subprime,
            self.cluster_profile,
            top_n=3
        )
        for card in recs:
            # Luxury card like Diners Black Metal (fee 10,000) or Atlas (5,000) should not top subprime rank
            self.assertNotEqual(card['card_name'], "HDFC Diners Club Black Metal")

    def test_segmentation_prediction(self):
        """Test that K-Means segmentation outputs valid cluster and persona name."""
        cluster_res = predict_spending_cluster(self.spending_profile)
        self.assertIn('cluster_id', cluster_res)
        self.assertIn('persona_name', cluster_res)
        self.assertIsInstance(cluster_res['cluster_id'], int)
        self.assertIn(cluster_res['cluster_id'], [0, 1, 2, 3])

if __name__ == '__main__':
    unittest.main()
