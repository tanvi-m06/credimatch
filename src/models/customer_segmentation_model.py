"""
Customer Segmentation Model for CrediMatch AI.
Unsupervised K-Means clustering trained on Credit Card Customer Segmentation data.
Discovers behavioral spending archetypes and maps customers to tailored card personas.
"""

from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import (
    SEGMENTATION_MODEL_PATH,
    KMEANS_CLUSTERS,
    KMEANS_RANDOM_STATE
)
from src.data_processing.credit_data_loader import load_segmentation_data, SEGMENTATION_FEATURES

CLUSTERING_FEATURES = [
    'total_trans_amount',
    'total_trans_count',
    'avg_utilization_ratio',
    'credit_limit',
    'estimated_income',
    'avg_ticket_size'
]

class CustomerSegmentationPipeline:
    def __init__(self, n_clusters: int = KMEANS_CLUSTERS):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE, n_init=10)
        self.feature_names = CLUSTERING_FEATURES
        self.cluster_profiles = {}
        self.persona_map = {}
        self.metrics = {}
        
    def fit(self, df: pd.DataFrame):
        X = df[self.feature_names].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        clusters = self.kmeans.fit_predict(X_scaled)
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # Calculate silhouette score on sample for speed
        sample_size = min(3000, len(X))
        sil = silhouette_score(X_scaled[:sample_size], clusters[:sample_size])
        
        self.metrics = {
            'silhouette_score': round(float(sil), 4),
            'total_customers': len(df),
            'n_clusters': self.n_clusters
        }
        
        # Compute cluster centroids in original space
        profile_df = df_clustered.groupby('cluster')[self.feature_names].mean().round(2)
        self.cluster_profiles = profile_df.to_dict(orient='index')
        
        # Dynamically assign meaningful human personas based on centroid attributes
        for c_id, stats in self.cluster_profiles.items():
            spend = stats['total_trans_amount']
            util = stats['avg_utilization_ratio']
            income = stats['estimated_income']
            
            if spend > 8000:
                name = "Power Spender / High Velocity"
                desc = "High transaction frequency & high annual volume. Maximizes reward points and travel benefits."
                focus = ["Travel", "Dining", "Accelerated Points", "Milestone Rewards"]
                icon = "✈️"
            elif util > 0.45:
                name = "Revolver / High Utilization"
                desc = "Carries high balance and high credit utilization. Benefits from lower interest and balance protection."
                focus = ["Low APR", "Cashback", "Balance Transfer", "Utility Bill Payments"]
                icon = "💳"
            elif income > 80000 and util < 0.25:
                name = "Affluent / High Limit Low Utilization"
                desc = "High income with conservative card usage. Seeks luxury lifestyle perks and lounge privileges."
                focus = ["Airport Lounge Access", "Golf Privileges", "Zero Forex Markup", "Premium Concierge"]
                icon = "💎"
            else:
                name = "Budget Conscious / Low Spend"
                desc = "Careful spender prioritizing zero maintenance fees, high base cashback, and essential groceries."
                focus = ["Zero Annual Fee", "Lifetime Free", "Groceries Cashback", "Fuel Surcharge Waiver"]
                icon = "🛒"
                
            self.persona_map[c_id] = {
                'name': name,
                'desc': desc,
                'focus': focus,
                'icon': icon,
                'benchmarks': stats
            }
            
    def predict_segment(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies an applicant into a behavioral spending segment.
        """
        annual_spend = float(user_profile.get('annual_spend', 250000.0))
        # convert to approximate dataset scale or normalized monthly
        trans_count = float(user_profile.get('annual_trans_count', 65.0))
        utilization = float(user_profile.get('utilization_ratio', 0.25))
        credit_limit = float(user_profile.get('credit_limit', 150000.0))
        income = float(user_profile.get('income', 800000.0))
        
        # Scaling adjustment: customer_segmentation.csv spending is annual (mean ~4400 USD / units)
        # Normalize to the dataset distribution:
        # If user inputs INR, divide by approx benchmark ratio to map smoothly into space
        spend_scaled = annual_spend if annual_spend < 50000 else annual_spend / 60.0
        income_scaled = income if income < 300000 else income / 12.0
        limit_scaled = credit_limit if credit_limit < 50000 else credit_limit / 20.0
        
        ticket_size = spend_scaled / max(1.0, trans_count)
        
        row = [spend_scaled, trans_count, utilization, limit_scaled, income_scaled, ticket_size]
        row_scaled = self.scaler.transform([row])
        
        cluster_id = int(self.kmeans.predict(row_scaled)[0])
        persona = self.persona_map.get(cluster_id, {
            'name': 'Everyday Value Maximizer',
            'desc': 'Balanced spender looking for solid rewards across dining, shopping, and travel.',
            'focus': ['Dining', 'Shopping', 'Cashback'],
            'icon': '🌟',
            'benchmarks': {}
        })
        
        return {
            'cluster_id': cluster_id,
            'persona_name': persona['name'],
            'description': persona['desc'],
            'recommended_focus': persona['focus'],
            'icon': persona['icon'],
            'benchmarks': persona.get('benchmarks', {})
        }

def train_or_load_segmentation_model(force_retrain: bool = False) -> CustomerSegmentationPipeline:
    """Loads existing trained K-Means pipeline or fits and persists a new one."""
    if SEGMENTATION_MODEL_PATH.exists() and not force_retrain:
        try:
            pipeline = joblib.load(SEGMENTATION_MODEL_PATH)
            return pipeline
        except Exception:
            pass
            
    print("Training Unsupervised K-Means Customer Segmentation Model...")
    df = load_segmentation_data()
    pipeline = CustomerSegmentationPipeline()
    pipeline.fit(df)
    
    SEGMENTATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, SEGMENTATION_MODEL_PATH)
    print(f"K-Means Model persisted to {SEGMENTATION_MODEL_PATH}. Silhouette: {pipeline.metrics.get('silhouette_score')}")
    return pipeline
