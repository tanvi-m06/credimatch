# CrediMatch AI: Personalized Credit Card Recommendation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![K-Means](https://img.shields.io/badge/Clustering-K--Means-brightgreen.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)

A full-stack, genuine Machine Learning major project combining **supervised credit risk classification (XGBoost)**, **unsupervised customer spending segmentation (K-Means)**, **explainable AI (SHAP)**, and a **deterministic Net Annual Value (NAV) card ranking engine**.

---

## 1. Problem Statement
Credit card product selection in consumer finance is plagued by misalignment:
- Consumers apply for cards that mismatch their spending patterns, resulting in sub-optimal reward yields and unrecovered annual fees.
- Aggregators often promote cards based on commercial affiliate kickbacks rather than individual user utility.
- Applicants frequently experience credit inquiry rejections because they apply for prime products exceeding their credit-risk profile.

**CrediMatch AI** addresses this challenge with an academic, transparent, and multi-stage decision pipeline that pairs institutional risk gating with personalized spending optimization.

---

## 2. System Architecture & Data Flow

```
                      +------------------------------------------+
                      |         User Input Interface (UI)        |
                      | - Financial Demographics (Income, Age)   |
                      | - Category Monthly Spends (6 Categories) |
                      | - Airport Lounge Visits & Bureau Signal  |
                      +--------------------+---------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
     +------------------------------+             +-------------------------------+
     |  Stage 1: Supervised Model   |             |  Stage 2: Unsupervised Model  |
     |  XGBoost Credit-Risk Engine  |             |  K-Means Spending Clustering  |
     |  (Home Credit Dataset)       |             |  (Multi-Category Spend Mix)   |
     +--------------+---------------+             +---------------+---------------+
                    |                                             |
                    | PD & Risk Tier                              | Cluster Persona
                    | (Prime / Near-Prime / Subprime)             | (e.g., Dining Enthusiast)
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |    Stage 3: Hybrid Recommendation Engine |
                      | - Filter cards by Risk Approval Gate     |
                      | - Category Rewards Calculation           |
                      | - Spend-based Annual Fee Waiver check    |
                      | - Airport Lounge & Welcome Bonus credit  |
                      | - Net Annual Value (NAV) Optimization    |
                      +--------------------+---------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |      Top 3 Personalized Credit Cards     |
                      | - Exact Rupee Net Annual Value (NAV)     |
                      | - Category-wise Reward Attribution       |
                      | - Local SHAP Risk Explanation Waterfall  |
                      | - Interactive What-If Scenario Simulator |
                      +------------------------------------------+
```

---

## 3. Dataset Sources & Limitations

1. **Home Credit Default Risk (`data/home_credit/application_train.csv`):**
   - **Size:** 158.4 MB (307,511 rows × 122 columns).
   - **Target Variable:** `TARGET` (Binary: `0` non-default [91.9%], `1` default [8.1%]).
   - **Source:** Kaggle Home Credit Default Risk Competition.
   - **Academic Note & Limitation:** This dataset records loan repayment defaults, not direct credit card approval decisions. It is used as a proxy for institutional default risk ($PD$).
2. **Category Spending Distributions:**
   - **Dimensions:** Groceries, Dining, Travel, Online shopping, Fuel, Entertainment.
   - **Scale:** Multi-variate normalized Indian rupee distributions capturing distinct household spending habits.
3. **Credit Card Product Catalogue (`data/credit_cards.csv`):**
   - **Cards:** 15 retail credit cards across HDFC, SBI, ICICI, Axis, Amex, Standard Chartered, Kotak, AU Small Finance, etc.
   - **Features:** Annual fees, spend waiver thresholds, category-wise reward %, welcome perks, lounge quotas, and reward caps.
   - **Separation:** Kept in a standalone CSV file so product terms can be updated without touching Python code.

---

## 4. Feature Engineering

| Feature Name | Source / Formula | Rationale |
| :--- | :--- | :--- |
| `AGE_YEARS` | `-DAYS_BIRTH / 365.25` | Standardizes applicant age into continuous years. |
| `YEARS_EMPLOYED` | `DAYS_EMPLOYED` (negative clipped) `/ 365.25` | Normalizes employment stability; clips anomalous positive values (pensioners). |
| `DEBT_TO_INCOME` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Fundamental solvency ratio measuring debt burden relative to earning capacity. |
| `ANNUITY_TO_INCOME` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Monthly debt serviceability indicator. |
| `EXT_SOURCE_1..3` | Normalized credit bureau scores ($0.0$ to $1.0$) | Captures external credit history and past repayment track record. |

---

## 5. Machine Learning Methodology & Model Selection

### Why XGBoost for Credit Risk?
1. **Handling Tabular Asymmetry & Missing Values:** XGBoost natively handles missing numerical values via optimal default split directions.
2. **Class Imbalance Support:** The Home Credit dataset has an 11.4:1 non-default to default ratio. XGBoost's `scale_pos_weight` parameter directly reweights positive errors in the loss function without needing risky synthetic oversampling (SMOTE).
3. **Non-linear Decision Boundaries:** Captures complex financial interactions between income, credit limit, age, and bureau scores.

### Why K-Means for Spending Behaviour?
1. **Geometric Centroid Interpretability:** K-Means produces clear geometric cluster centers, making it straightforward to analyze what a "Dining Enthusiast" or "Everyday Essentials" spender actually spends on average.
2. **Deterministic Distance Mapping:** An incoming user's spending vector can be mapped into cluster space in $O(k)$ time using Euclidean distance to centroids.

---

## 6. Model Evaluation Results

### Supervised XGBoost Credit-Risk Model (Test Set, 15,000 samples)
* **ROC-AUC Score:** **`0.7349`**
* **Recall (Default Detection):** **`66.22%`**
* **Precision:** `15.91%` (balanced for conservative default detection)
* **F1-Score:** `0.2566`
* **Confusion Matrix:**
  - True Negatives (TN): `9,592`
  - False Positives (FP): `4,206`
  - False Negatives (FN): `406`
  - True Positives (TP): `796`

### Unsupervised K-Means Spending Segmentation
* **Number of Clusters ($k$):** `4`
* **Silhouette Score:** **`0.6540`** (Demonstrating strong, well-separated cluster cohesion)
* **Discovered Personas:**
  1. *Cluster 0 — High-Flyer / Luxury Travel* (Mean Travel: ₹3,20,732 / yr)
  2. *Cluster 1 — Everyday Essentials & Groceries* (Mean Groceries: ₹1,00,823 / yr)
  3. *Cluster 2 — Dining & Lifestyle Enthusiast* (Mean Dining: ₹1,12,495 / yr)
  4. *Cluster 3 — Daily Commuter & Fuel Maximizer* (Mean Fuel: ₹74,878 / yr)

---

## 7. Recommendation & Net Annual Value (NAV) Formulas

For every candidate credit card in the catalogue:

### 1. Estimated Annual Category Rewards
$$\text{Annual Rewards (₹)} = \sum_{c \in \{\text{Groceries, Dining, Travel, Online, Fuel, Entertainment}\}} \left(\text{User Annual Spend}_c \times \frac{\text{Card Reward Rate}_c}{100}\right)$$
*(Adjusted for documented monthly/billing cycle cashback caps)*.

### 2. Airport Lounge Benefit Valuation
$$\text{Lounge Value (₹)} = \min(\text{User Dom Visits}, \text{Card Dom}) \times 800 + \min(\text{User Intl Visits}, \text{Card Intl}) \times 2,000$$

### 3. Net Annual Fee (with Spend-based Waiver)
$$\text{Net Fee (₹)} = \begin{cases} 0 & \text{if } \text{Annual Fee} = 0 \text{ (LTF) or Total Annual Spend} \ge \text{Waiver Threshold} \\ \text{Annual Fee} & \text{otherwise} \end{cases}$$

### 4. Net Annual Value (NAV)
$$\text{NAV (₹)} = \text{Annual Rewards} + \text{Lounge Value} + \text{Welcome Bonus} - \text{Net Fee}$$

### 5. Final Composite Ranking Score
$$\text{Score} = \text{NAV} + \text{Persona Alignment Bonus} - \text{Risk Ineligibility Penalty}$$
*Top 3 highest scoring eligible cards are recommended.*

---

## 8. Explainability (SHAP & Financial Attribution)

* **SHAP TreeExplainer:** Computes exact game-theoretic Shapley values for the XGBoost model. Tells the user whether their debt-to-income, credit limit request, or age *increased* or *decreased* their estimated default risk.
* **Transparent Reward Attribution:** Explains card recommendations using verified reward breakdowns (e.g. `Highest reward driver: Dining earning ₹3,000 cashback`). Recommendation reasons are strictly separated from SHAP risk values.

---

## 9. Project Directory Structure

```
CrediMatch/
├── app.py                     # Main Streamlit interactive dashboard
├── train_model.py             # Supervised XGBoost training pipeline
├── segmentation.py            # Unsupervised K-Means clustering module
├── recommender.py             # NAV card ranking and evaluation logic
├── explainability.py          # SHAP TreeExplainer & recommendation insights
├── test_recommender.py        # Unit and integration test suite
├── requirements.txt           # Verified dependencies
├── README.md                  # Comprehensive project documentation
├── data/
│   ├── credit_cards.csv       # 15-card product catalogue (fees, rewards, caps)
│   ├── home_credit/           # Home credit default risk dataset
│   │   └── application_train.csv
│   └── segmentation/          # Customer segmentation dataset
│       └── customer_segmentation.csv
└── models/
    ├── credit_risk_xgb.joblib # Serialized XGBoost model binary
    ├── risk_imputer.joblib    # Median imputer artifact
    ├── risk_metrics.json      # Saved evaluation metrics & confusion matrix
    ├── customer_segmentation_kmeans.joblib # K-Means & StandardScaler
    └── segmentation_metrics.json # Saved silhouette score & cluster metadata
```

---

## 10. Installation & Execution Commands

### Prerequisites
* Python 3.10 to 3.13
* Virtual environment (recommended)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```powershell
python test_recommender.py
```
*(All 6 unit tests should report `OK`)*.

### Step 3: (Optional) Retrain Models
```powershell
python train_model.py
python segmentation.py
```

### Step 4: Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 11. Project Limitations & Future Scope

1. **Proxy Risk Target:** The Home Credit dataset tracks personal loan repayment rather than revolving card lines. Future iterations could train on open banking transaction records.
2. **Merchant Category Codes (MCC):** Real bank transactions are identified by MCC codes; our engine uses category totals rather than individual merchant feeds.
3. **Dynamic APR & Revolving Balances:** Incorporating month-to-month revolving balance finance charges for applicants who carry balances across cycles.
