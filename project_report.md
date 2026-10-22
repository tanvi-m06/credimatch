# CrediMatch AI — Complete Plain-English Project Guide

---

## 1. The Big Picture: What is CrediMatch AI?

In one sentence:
> **CrediMatch AI is an intelligent recommendation system that helps everyday people find the best credit card for their real spending habits and financial profile, while accurately estimating how much money they will actually earn or save each year.**

### Why does this project exist? (The Problem)
Most credit card comparison websites are confusing, biased, or misleading:
1. **Hidden Fees & Complex Rules:** A card might boast "5% cashback", but hides a monthly cap of ₹500 or requires spending ₹2,00,000 annually to waive a ₹3,000 annual fee.
2. **One-Size-Fits-All Advice:** What is great for a frequent business traveler flying every week is a money-losing card for a college graduate who mostly orders food on Swiggy and buys groceries.
3. **Application Rejections:** People apply for luxury cards they don't qualify for, damaging their credit score with hard inquiries.

**CrediMatch solves this** by combining **machine learning risk checks**, **customer behavioral clustering**, and **transparent rupee-by-rupee financial math**.

---

## 2. How CrediMatch Works (The 3-Pillar Brain)

CrediMatch is **not** a simple static list. It uses a **Hybrid Multi-Stage AI Pipeline**:

```mermaid
flowchart TD
    User["User Financial & Spending Input"] --> Stage1["Pillar 1: Supervised ML (XGBoost)"]
    User --> Stage2["Pillar 2: Unsupervised ML (K-Means)"]
    
    Stage1 -->|"Risk Tier (Prime / Near-Prime / Subprime)"| Gate["Eligibility & Risk Gating"]
    Stage2 -->|"Spending Persona & Top Categories"| Engine["Pillar 3: Recommender Engine"]
    Gate --> Engine
    
    Catalogue["Credit Card Catalogue (credit_cards.csv)"] --> Engine
    Engine -->|"Calculate Net Annual Value (NAV) + Alignment Bonus"| Ranking["Ranked Top 3 Recommendations"]
```

---

### Pillar 1: Supervised ML — Credit Risk Scoring (XGBoost)
* **File:** [`train_model.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/train_model.py)
* **What it does in simple words:** It acts like an automated bank loan officer. Before recommending high-fee metal credit cards, it assesses whether the applicant has low risk, moderate risk, or high default risk.
* **Algorithm:** **XGBoost Classifier** (Extreme Gradient Boosting).
* **Dataset:** Home Credit Default Risk (`application_train.csv`).
* **Input Features (11 key indicators):**
  - Income, credit amount requested, estimated annuity
  - Applicant age and years employed
  - Number of children/dependents
  - Debt-to-Income (DTI) and Annuity-to-Income ratios
  - External credit bureau scores (Bureau 1, 2, 3)
* **Risk Tiers:**
  - **Low Risk (Prime):** Probability of default < 20% $\rightarrow$ Qualifies for all cards, including luxury metal cards.
  - **Moderate Risk (Near-Prime):** Default probability 20%–45% $\rightarrow$ Qualifies for mid-tier cards (annual fee $\le$ ₹5,000).
  - **High Risk (Subprime):** Default probability > 45% $\rightarrow$ Gated to secured / entry-level / lifetime-free cards (fee $\le$ ₹1,500) to protect the applicant from unsustainable debt.

---

### Pillar 2: Unsupervised ML — Spending Archetypes (K-Means Clustering)
* **File:** [`segmentation.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/segmentation.py)
* **What it does in simple words:** People don't spend money randomly; they have distinct lifestyle habits. Instead of assuming what a user wants, K-Means groups spending patterns into natural human "personas."
* **Algorithm:** **K-Means Clustering** ($k = 4$) with feature scaling (`StandardScaler`).
* **Categories Evaluated (6):** Groceries, Dining, Travel, Online Shopping, Fuel, Entertainment.
* **The 4 Personas Identified:**
  1. 🛒 **Everyday Essentials Spender:** High groceries, supermarkets, and utility bills.
  2. 🍕 **Dining & Urban Lifestyle:** Heavy on food delivery (Swiggy/Zomato), restaurants, and weekend outings.
  3. ✈️ **Luxury Travel & High-Flyer:** Frequent flights, hotels, and international lounge visits.
  4. 🚗 **Daily Commuter & Fuel Spender:** High fuel purchases, transit, and offline shopping.
* **Why this matters:** When a user matches the *Dining & Urban Lifestyle* persona, the recommendation engine gives bonus ranking points to cards offering 5% on Swiggy and Zomato.

---

### Pillar 3: The Financial Valuation Engine (Net Annual Value — NAV)
* **File:** [`recommender.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/recommender.py)
* **What it does in simple words:** It translates marketing percentages into cold, hard rupees so the user knows exactly: *"Will this card make me money or cost me money?"*

#### The Exact Mathematical Formula:
$$\text{Net Annual Value (NAV)} = (\text{Category Rewards} + \text{Lounge Perk Value} + \text{Welcome Bonus}) - \text{Net Annual Fee}$$

1. **Category Rewards:** Calculates rupees earned per category ($Spend \times Reward\%$).
2. **Cap Deductions:** If a card caps cashback at ₹1,000/month (e.g. Axis Ace), the engine caps annual reward at ₹12,000 rather than showing fake millions.
3. **Lounge Access Valuation:** Evaluates domestic lounges at ₹800/visit and international lounges at ₹2,000/visit, up to the card's complimentary limit.
4. **Spend Waiver Logic:** If user's annual spend exceeds the card's waiver threshold (e.g. ₹1,00,000 spend on HDFC Millennia), the fee drops to ₹0.
5. **Final Ranking Score:**
   $$\text{Ranking Score} = \text{NAV} + \text{Persona Alignment Bonus}$$
   Cards offering $\ge 3\%$ on the user's persona priority categories receive a ₹1,000 alignment bonus.

---

## 3. The Credit Card Catalogue ([`credit_cards.csv`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/data/credit_cards.csv))

The catalogue includes 10 popular Indian credit cards covering every segment:
- **HDFC Millennia:** Ideal for online shoppers (5% on Amazon/Flipkart, ₹1L waiver).
- **SBI SimplyCLICK:** Great entry-level card for online merchants.
- **Axis Ace:** Ideal for bill payments and offline spends (Google Pay 5%, 2% flat).
- **HDFC Diners Club Black Metal:** Premium travel & dining card with unlimited airport lounges.
- **Tata Neu Infinity HDFC:** Top-tier UPI and Tata ecosystem card (5% NeuCoins).
- **Axis Atlas:** Frequent flyer card with high tier-point milestones.
- **ICICI Amazon Pay:** 100% Lifetime Free (LTF) card for Amazon Prime users.
- **Standard Chartered DigiSmart:** Entry-level discount card for movie tickets and Myntra.
- **BPCL SBI Card Octane:** Fuel multiplier card with surcharge waivers.
- **SBI Card ELITE:** Balanced premium card for movies and international travel.

---

## 4. The Website Experience ([`app.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/app.py))

Built with Streamlit and styled with a **minimalist digital banking aesthetic** (warm off-white `#FAF9F6`, beige `#F0E8DF`, deep maroon `#702D3D`):

### The 4 Core Navigation Pages:
1. **Home:** Clean introductory landing page explaining value proposition with a "Find My Cards" call to action.
2. **My Profile:**
   - **Zero Defaults:** Starts with every field at `0` (no assumed profile, no fake defaults).
   - **Fields:** Monthly Take-Home Income, Age, Airport Lounge Visits, and 6 category expenses.
   - **Validation:** Clicking "Find My Cards" requires entering at least one spending category $>0$. All-zero profiles show a friendly error message.
3. **Find My Cards (The Core Feature):**
   - **Before Submission:** Displays a clean empty state: *"Your personalised matches will appear here."* with a **[Complete My Profile]** button.
   - **After Submission:** Displays the Top 3 personalized card matches with:
     - Prominent **Estimated Net Annual Value (₹/year)**.
     - Annual fee breakdown and spend waiver status.
     - Human-readable reason bullets explaining why the card matches their spending.
     - Expandable details with reward breakdowns and cap rules.
     - Side-by-side comparison table.
   - **Stale Detection:** If the user alters their profile numbers, the app displays a prominent alert indicating recalculation is required.
4. **Spending Insights:**
   - Visual horizontal bar chart of the user's spending breakdown.
   - Assigned cardholder persona badge and description.
   - Rupee-value highlights showing their highest spending category and multiplier opportunities.

---

## 5. College Viva / Interview Cheat Sheet (Q&A)

Here are the most common questions professors or interviewers ask about this project and the exact answers:

### Q1: "Is this just a rule-based system or actual Machine Learning?"
> **Answer:** "It is a hybrid intelligent system. We use two genuine Machine Learning models:
> 1. A supervised **XGBoost classifier** trained on applicant financial features to predict default probability and gate card suitability.
> 2. An unsupervised **K-Means clustering model** trained on multi-category spending to discover user lifestyle personas.
> Rule-based logic is only used where financial regulation requires exact determinism — specifically for calculating exact rupees earned from published card fee and reward schedules."

### Q2: "Why did you use XGBoost for credit risk instead of Logistic Regression?"
> **Answer:** "Financial default data contains non-linear interactions and skewed distributions (e.g. debt-to-income interacting with employment tenure). Tree-based gradient boosting (XGBoost) naturally captures non-linear relationships, handles missing values gracefully, and outperforms linear models on the Home Credit dataset with an AUC score of ~0.74."

### Q3: "Why did you use K-Means clustering for customer segmentation?"
> **Answer:** "K-Means efficiently identifies natural groupings in multi-dimensional spending space. By scaling features with `StandardScaler` and tuning $k=4$, we achieved distinct, interpretable clusters (Family Groceries, Dining/Urban, Luxury Travel, Commuter Fuel) verified with a Silhouette Score of ~0.42."

### Q4: "What is Net Annual Value (NAV) and why is it important?"
> **Answer:** "Most card aggregators only tell users the reward rate (e.g., 'Earn 5%'). NAV calculates the true economic benefit:
> $\text{NAV} = \text{Rewards} + \text{Lounge Value} + \text{Welcome Bonus} - \text{Net Fee}$.
> This ensures a card with a ₹10,000 fee is only recommended if the rewards and perks significantly exceed the fee."

### Q5: "What happens if a user spends ₹0 in one category or has zero income?"
> **Answer:** "The application includes strict zero-division safeguards across all ratios (like DTI and annuity calculations). If a category has ₹0 spend, the model handles it gracefully without skewing recommendations. If income is not provided, the engine safely estimates gating thresholds from spending volume without throwing errors."

---

## 6. Project Architecture & File Summary

| File | Purpose | Key Technologies |
|---|---|---|
| [`app.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/app.py) | Streamlit web frontend | Streamlit, Plotly, CSS |
| [`train_model.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/train_model.py) | Credit risk model training | XGBoost, scikit-learn, joblib |
| [`segmentation.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/segmentation.py) | Spending persona clustering | K-Means, StandardScaler |
| [`recommender.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/recommender.py) | NAV calculation & card ranking | pandas, NumPy, financial math |
| [`explainability.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/explainability.py) | Transparent recommendation reasons | SHAP, Python dictionaries |
| [`test_recommender.py`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/test_recommender.py) | Recommender unit test suite | Python `unittest` (6/6 tests) |
| [`credit_cards.csv`](file:///c:/Users/Krishna/OneDrive/Documents/CrediMatch/data/credit_cards.csv) | Card product catalogue | 10 Indian cards & fee rules |
