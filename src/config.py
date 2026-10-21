"""
CrediMatch AI - Configuration Module
Defines project paths, model hyperparameters, and business rules.
"""

from pathlib import Path
import os

# Base directory resolution
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

def find_data_dir() -> Path:
    """Locate the data directory whether running from workspace root or credimatch/."""
    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
        PROJECT_ROOT / "credimatch" / "data",
        Path.cwd() / "data",
        Path.cwd() / "credimatch" / "data"
    ]
    for p in candidates:
        if p.exists() and (p / "home_credit").exists():
            return p.resolve()
    # Default fallback
    return (PROJECT_ROOT / "data").resolve()

DATA_DIR = find_data_dir()

# Dataset Paths
HOME_CREDIT_PATH = DATA_DIR / "home_credit" / "application_train.csv"
SEGMENTATION_PATH = DATA_DIR / "segmentation" / "customer_segmentation.csv"
CATALOGUE_XLSX_PATH = DATA_DIR / "catalogue" / "card_catalogue.xlsx"
CATALOGUE_CSV_PATH = DATA_DIR / "catalogue" / "card_catalogue_parsed.csv"

# Saved Model Paths
SAVED_MODELS_DIR = SRC_DIR / "models" / "saved"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

RISK_MODEL_PATH = SAVED_MODELS_DIR / "credit_risk_xgb.joblib"
SEGMENTATION_MODEL_PATH = SAVED_MODELS_DIR / "customer_segmentation_kmeans.joblib"

# Training Configuration
# Chunked / Sampled loading for Home Credit to preserve memory and run fast
HOME_CREDIT_SAMPLE_SIZE = 75000
HOME_CREDIT_RANDOM_STATE = 42

# K-Means Clusters
KMEANS_CLUSTERS = 4
KMEANS_RANDOM_STATE = 42

# Business Rules for Card Valuation
DOMESTIC_LOUNGE_VALUE_INR = 800.0       # Per lounge visit estimated perk value
INTERNATIONAL_LOUNGE_VALUE_INR = 2000.0 # Per intl lounge visit value
DEFAULT_FOREX_BENCHMARK = 3.5            # Standard Indian credit card forex fee %

# Risk Tiers & Gating Thresholds
RISK_THRESHOLD_LOW = 0.20       # P(default) < 0.20 -> Low Risk (Prime)
RISK_THRESHOLD_MEDIUM = 0.45    # 0.20 <= P(default) < 0.45 -> Moderate Risk (Near Prime)
# P(default) >= 0.45 -> High Risk (Subprime)
