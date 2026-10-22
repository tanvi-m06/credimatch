"""
================================================================================
CrediMatch AI - Spending Behaviour Segmentation Module (K-Means)
================================================================================
Author: Senior ML Engineer
Target Audience: Third-Year IT Project / Viva Examination
Description:
    Implements unsupervised K-Means clustering on multi-category customer spending
    (Groceries, Dining, Travel, Online shopping, Fuel, Entertainment).
    
Key Steps:
    1. Generates / processes category spending behavior data.
    2. Scales features using StandardScaler before distance-based clustering.
    3. Fits K-Means with fixed random seed (k=4).
    4. Evaluates cluster quality using Silhouette Score.
    5. Discovers clear, interpretable spending archetypes.
    6. Persists artifacts to models/ for inference.
================================================================================
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "customer_segmentation_kmeans.joblib"
METRICS_FILE = MODELS_DIR / "segmentation_metrics.json"

CATEGORIES = ['groceries', 'dining', 'travel', 'online_shopping', 'fuel', 'entertainment']

def generate_spending_training_data(n_samples=5000, random_state=42):
    """
    Generates realistic, representative customer category spending distributions
    spanning 4 distinct natural behavioral personas.
    """
    np.random.seed(random_state)
    data = []
    
    # Archetype 1: Everyday Essentials / Family Groceries (40% groceries, 25% online, 15% dining, 10% fuel, 5% entertainment, 5% travel)
    for _ in range(int(n_samples * 0.35)):
        total = np.random.normal(250000, 50000)
        data.append({
            'groceries': total * np.random.uniform(0.35, 0.45),
            'dining': total * np.random.uniform(0.10, 0.18),
            'travel': total * np.random.uniform(0.02, 0.08),
            'online_shopping': total * np.random.uniform(0.20, 0.30),
            'fuel': total * np.random.uniform(0.08, 0.15),
            'entertainment': total * np.random.uniform(0.03, 0.08),
        })
        
    # Archetype 2: Dining & Urban Lifestyle / Millennial (35% dining, 30% online, 15% entertainment, 10% groceries, 5% fuel, 5% travel)
    for _ in range(int(n_samples * 0.25)):
        total = np.random.normal(320000, 60000)
        data.append({
            'groceries': total * np.random.uniform(0.08, 0.15),
            'dining': total * np.random.uniform(0.30, 0.42),
            'travel': total * np.random.uniform(0.05, 0.12),
            'online_shopping': total * np.random.uniform(0.25, 0.35),
            'fuel': total * np.random.uniform(0.04, 0.08),
            'entertainment': total * np.random.uniform(0.12, 0.20),
        })
        
    # Archetype 3: High-Flyer / Luxury Travel (45% travel, 20% dining, 15% online, 10% entertainment, 5% groceries, 5% fuel)
    for _ in range(int(n_samples * 0.20)):
        total = np.random.normal(650000, 120000)
        data.append({
            'groceries': total * np.random.uniform(0.03, 0.08),
            'dining': total * np.random.uniform(0.15, 0.25),
            'travel': total * np.random.uniform(0.40, 0.55),
            'online_shopping': total * np.random.uniform(0.12, 0.20),
            'fuel': total * np.random.uniform(0.03, 0.07),
            'entertainment': total * np.random.uniform(0.08, 0.15),
        })
        
    # Archetype 4: Commuter & Fuel / Offline Value (35% fuel, 25% groceries, 15% dining, 12% online, 8% entertainment, 5% travel)
    for _ in range(int(n_samples * 0.20)):
        total = np.random.normal(200000, 40000)
        data.append({
            'groceries': total * np.random.uniform(0.20, 0.30),
            'dining': total * np.random.uniform(0.10, 0.18),
            'travel': total * np.random.uniform(0.02, 0.06),
            'online_shopping': total * np.random.uniform(0.08, 0.15),
            'fuel': total * np.random.uniform(0.30, 0.45),
            'entertainment': total * np.random.uniform(0.05, 0.10),
        })
        
    df = pd.DataFrame(data).clip(lower=1000)
    return df

def train_segmentation_model(n_clusters=4):
    """
    Fits StandardScaler and K-Means, calculates Silhouette score, and saves model.
    """
    print(f"[*] Generating category spending dataset ({n_clusters} clusters)...")
    df = generate_spending_training_data()
    X = df[CATEGORIES]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KMeans with fixed seed
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters
    
    # Evaluate with silhouette score
    sil = silhouette_score(X_scaled[:2500], clusters[:2500])
    print(f"[*] K-Means Silhouette Score: {sil:.4f}")
    
    # Analyze Cluster Centroids in Original Space
    centroids = df.groupby('cluster')[CATEGORIES].mean().round(0)
    print("\n" + "=" * 60)
    print("CLUSTER CENTROIDS (Mean Category Spending in INR):")
    print("=" * 60)
    print(centroids.to_string())
    print("=" * 60)
    
    # Determine human-friendly labels dynamically from centroids
    persona_definitions = {}
    for c_id, row in centroids.iterrows():
        top_cat = row.idxmax()
        if top_cat == 'travel':
            name = "High-Flyer / Luxury Travel"
            desc = "Heavy travel & international spender. Maximizes airline miles, hotel points, and lounge perks."
            icon = "✈️"
            priority_cats = ['travel', 'dining']
        elif top_cat == 'dining':
            name = "Dining & Lifestyle Enthusiast"
            desc = "Frequent restaurant diner and food-delivery user with high e-commerce & weekend entertainment activity."
            icon = "🍽️"
            priority_cats = ['dining', 'entertainment', 'online_shopping']
        elif top_cat == 'fuel':
            name = "Daily Commuter & Fuel Maximizer"
            desc = "Prioritizes vehicle fuel expenses, highway tolls, and day-to-day offline convenience."
            icon = "⛽"
            priority_cats = ['fuel', 'groceries']
        else:
            name = "Everyday Essentials & Groceries"
            desc = "Prudent household budget manager focusing on supermarkets, groceries, utilities, and online shopping."
            icon = "🛒"
            priority_cats = ['groceries', 'online_shopping']
            
        persona_definitions[int(c_id)] = {
            'name': name,
            'description': desc,
            'icon': icon,
            'priority_categories': priority_cats,
            'centroid': row.to_dict()
        }
        
    pipeline = {
        'scaler': scaler,
        'kmeans': kmeans,
        'categories': CATEGORIES,
        'personas': persona_definitions,
        'silhouette_score': round(float(sil), 4)
    }
    
    joblib.dump(pipeline, MODEL_FILE)
    with open(METRICS_FILE, 'w') as f:
        json.dump({
            'silhouette_score': round(float(sil), 4),
            'n_clusters': n_clusters,
            'personas': {str(k): v['name'] for k, v in persona_definitions.items()}
        }, f, indent=4)
        
    print(f"[+] Saved segmentation pipeline to: {MODEL_FILE}")
    return pipeline

def predict_spending_cluster(user_spending):
    """
    Predicts the customer archetype cluster for a user's spending profile.
    """
    if not MODEL_FILE.exists():
        pipeline = train_segmentation_model()
    else:
        pipeline = joblib.load(MODEL_FILE)
        
    scaler = pipeline['scaler']
    kmeans = pipeline['kmeans']
    personas = pipeline['personas']
    
    # Prepare input vector as DataFrame to match feature names
    vec_df = pd.DataFrame([[float(user_spending.get(cat, 0.0)) for cat in CATEGORIES]], columns=CATEGORIES)
    vec_scaled = scaler.transform(vec_df)

    
    cluster_id = int(kmeans.predict(vec_scaled)[0])
    persona_info = personas.get(cluster_id, {
        'name': 'Balanced Value Spender',
        'description': 'Balanced spending profile across multiple categories.',
        'icon': '💳',
        'priority_categories': ['online_shopping', 'groceries'],
        'centroid': {}
    })
    
    return {
        'cluster_id': cluster_id,
        'persona_name': persona_info['name'],
        'description': persona_info['description'],
        'icon': persona_info['icon'],
        'priority_categories': persona_info['priority_categories'],
        'silhouette_score': pipeline.get('silhouette_score', 0.42)
    }

if __name__ == "__main__":
    train_segmentation_model()
