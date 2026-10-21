"""
================================================================================
CrediMatch AI - Consumer Credit Card Recommendation Website
================================================================================
A clean, premium, minimalist consumer fintech platform that matches everyday
users to optimal credit cards based on spending habits, lifestyle, and rewards.

Visual Palette:
  - Main Background: Warm off-white (#FAF9F6)
  - Cards & Content: Pure white (#FFFFFF)
  - Secondary Panels: Soft beige (#F0E8DF)
  - Primary Accent: Deep maroon (#702D3D)
  - Hover Accent: Muted burgundy (#8B4556)
  - Text: Dark charcoal (#292524)
  - Secondary Text: Warm grey (#78716C)
  - Borders: Subtle beige-grey (#E7DED5)
================================================================================
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Import modular ML backend components (Intact and preserved)
from recommender import recommend_top_cards, load_card_catalogue
from segmentation import predict_spending_cluster, CATEGORIES

# Page configuration
st.set_page_config(
    page_title="CrediMatch — Find the Card That Fits Your Life",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# FINTECH STYLING (Scoped CSS, Warm White & Maroon Theme)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #FAF9F6;
        color: #292524;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    h1, h2, h3, h4 {
        color: #292524 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, span, label, div {
        color: #292524;
    }
    
    /* Slim Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FAF9F6;
        border-right: 1px solid #E7DED5;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 8px 12px;
        border-radius: 8px;
        color: #44403C;
    }
    
    /* Primary Maroon Buttons */
    div.stButton > button {
        background-color: #702D3D !important;
        color: #FFFFFF !important;
        border: 1px solid #702D3D !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 1px 2px rgba(112, 45, 61, 0.1) !important;
    }
    div.stButton > button:hover {
        background-color: #8B4556 !important;
        border-color: #8B4556 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(112, 45, 61, 0.16) !important;
    }
    
    /* Pure White Cards */
    .product-card {
        background-color: #FFFFFF;
        border: 1px solid #E7DED5;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(41, 37, 36, 0.04);
    }
    
    /* Top #1 Match Card with subtle maroon accent */
    .top-match-card {
        background-color: #FFFFFF;
        border: 1px solid #E7DED5;
        border-left: 5px solid #702D3D !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(112, 45, 61, 0.05);
    }
    
    /* Soft Beige Panel for Monthly Spending */
    .beige-panel {
        background-color: #F0E8DF;
        border: 1px solid #E7DED5;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
    }
    
    /* Badges & Pills */
    .maroon-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 16px;
        background-color: #F0E8DF;
        color: #702D3D;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .persona-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background-color: #F0E8DF;
        color: #702D3D;
        font-size: 0.92rem;
        font-weight: 700;
        border: 1px solid #E7DED5;
    }
    
    /* Net Annual Value Display */
    .nav-figure {
        font-size: 1.85rem;
        font-weight: 800;
        color: #702D3D;
        line-height: 1.1;
    }
    
    /* Hero Header */
    .hero-container {
        padding: 2.5rem 0 2rem 0;
        max-width: 780px;
    }
    .hero-heading {
        font-size: 2.7rem;
        font-weight: 800;
        color: #292524;
        line-height: 1.18;
        margin-bottom: 0.9rem;
        letter-spacing: -0.03em;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #78716C;
        line-height: 1.6;
        margin-bottom: 1.8rem;
    }
    
    /* Clean feature highlight */
    .feature-tile {
        background-color: #FFFFFF;
        border: 1px solid #E7DED5;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(41, 37, 36, 0.03);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MODEL ARTIFACT LOADING (Cached)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "credit_risk_xgb.joblib"
IMPUTER_FILE = MODELS_DIR / "risk_imputer.joblib"

@st.cache_resource
def load_models_quietly():
    """Loads backend ML models quietly for inference without exposing technical metrics."""
    risk_m = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    risk_imp = joblib.load(IMPUTER_FILE) if IMPUTER_FILE.exists() else None
    cards_data = load_card_catalogue()
    return risk_m, risk_imp, cards_data

risk_model, risk_imputer, cards_df = load_models_quietly()

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Profile defaults (simple, consumer-relevant)
if "monthly_income" not in st.session_state:
    st.session_state.monthly_income = 65000  # ₹65,000 / month
if "age" not in st.session_state:
    st.session_state.age = 30

# Monthly spending categories (₹)
if "m_groceries" not in st.session_state:
    st.session_state.m_groceries = 6000
if "m_dining" not in st.session_state:
    st.session_state.m_dining = 5000
if "m_travel" not in st.session_state:
    st.session_state.m_travel = 4000
if "m_online" not in st.session_state:
    st.session_state.m_online = 7000
if "m_fuel" not in st.session_state:
    st.session_state.m_fuel = 2500
if "m_entertainment" not in st.session_state:
    st.session_state.m_entertainment = 1500
if "dom_lounge" not in st.session_state:
    st.session_state.dom_lounge = 4
if "intl_lounge" not in st.session_state:
    st.session_state.intl_lounge = 1

# ------------------------------------------------------------------------------
# SIMPLIFIED NAVIGATION (Only 4 Items)
# ------------------------------------------------------------------------------
NAV_ITEMS = [
    "Home",
    "My Profile",
    "Find My Cards",
    "Spending Insights"
]

st.sidebar.markdown("""
<div style="padding: 10px 0 18px 0; border-bottom: 1px solid #E7DED5; margin-bottom: 14px;">
    <div style="font-size: 1.25rem; font-weight: 800; color: #702D3D; letter-spacing: -0.02em;">
        💳 CrediMatch
    </div>
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #78716C; margin-top: 2px;">
        Smart Card Advisory
    </div>
</div>
""", unsafe_allow_html=True)

def on_nav_select():
    st.session_state.current_page = st.session_state.nav_radio

curr_page_idx = NAV_ITEMS.index(st.session_state.current_page) if st.session_state.current_page in NAV_ITEMS else 0
st.sidebar.radio(
    "Navigation Menu",
    NAV_ITEMS,
    index=curr_page_idx,
    key="nav_radio",
    on_change=on_nav_select,
    label_visibility="collapsed"
)

# ------------------------------------------------------------------------------
# INTERNAL REVENUE & CARD MATCHING HELPER
# ------------------------------------------------------------------------------
def compute_card_matches():
    """Computes personalized recommendations based on profile and spending."""
    annual_inc = float(st.session_state.monthly_income * 12)
    req_lim = min(annual_inc * 0.35, 300000.0)
    
    # Internal risk scoring quietly assigns suitability tier without showing raw probability to user
    feature_cols = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'YEARS_EMPLOYED',
        'CNT_CHILDREN', 'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    row_dict = {
        'AMT_INCOME_TOTAL': annual_inc,
        'AMT_CREDIT': req_lim,
        'AMT_ANNUITY': req_lim * 0.12,
        'AGE_YEARS': float(st.session_state.age),
        'YEARS_EMPLOYED': 4.0,
        'CNT_CHILDREN': 0,
        'DEBT_TO_INCOME': round(req_lim / max(1000.0, annual_inc), 3),
        'ANNUITY_TO_INCOME': round((req_lim * 0.12) / max(1000.0, annual_inc), 3),
        'EXT_SOURCE_1': 0.68,
        'EXT_SOURCE_2': 0.72,
        'EXT_SOURCE_3': 0.70
    }
    df_app = pd.DataFrame([row_dict], columns=feature_cols)
    df_imp = pd.DataFrame(risk_imputer.transform(df_app), columns=feature_cols)
    pd_prob = float(risk_model.predict_proba(df_imp)[0, 1])
    
    if pd_prob < 0.20:
        r_tier = "Low Risk (Prime)"
    elif pd_prob < 0.45:
        r_tier = "Moderate Risk (Near-Prime)"
    else:
        r_tier = "High Risk (Subprime)"
        
    risk_info = {
        'default_probability': pd_prob,
        'risk_tier': r_tier
    }
    
    spend_profile = {
        'groceries': float(st.session_state.m_groceries * 12),
        'dining': float(st.session_state.m_dining * 12),
        'travel': float(st.session_state.m_travel * 12),
        'online_shopping': float(st.session_state.m_online * 12),
        'fuel': float(st.session_state.m_fuel * 12),
        'entertainment': float(st.session_state.m_entertainment * 12),
        'domestic_lounge_visits': int(st.session_state.dom_lounge),
        'intl_lounge_visits': int(st.session_state.intl_lounge)
    }
    
    cluster_info = predict_spending_cluster(spend_profile)
    top_cards = recommend_top_cards(
        {'annual_income': annual_inc},
        spend_profile,
        risk_info,
        cluster_info,
        top_n=3
    )
    return cluster_info, spend_profile, top_cards

current_persona, current_spend, top_recommendations = compute_card_matches()

# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if st.session_state.current_page == "Home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-heading">Find the credit card that fits your life.</div>
        <div class="hero-sub">
            Discover personalised card recommendations based on your spending habits,
            lifestyle, and potential rewards.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Find My Cards", type="primary"):
        st.session_state.current_page = "Find My Cards"
        st.rerun()
        
    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
    
    # Exactly three compact feature highlights
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""
        <div class="feature-tile">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">🎯</div>
            <h4 style="margin: 0 0 6px 0; color: #292524;">Personalised card matches</h4>
            <p style="margin: 0; color: #78716C; font-size: 0.92rem; line-height: 1.5;">
                Cards matched to your real category spending across groceries, dining, online shopping, and travel.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-tile">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">₹</div>
            <h4 style="margin: 0 0 6px 0; color: #292524;">Estimated rewards and savings</h4>
            <p style="margin: 0; color: #78716C; font-size: 0.92rem; line-height: 1.5;">
                See clear rupee forecasts of rewards earned, annual fees, spend waivers, and airport lounge perks.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-tile">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">📊</div>
            <h4 style="margin: 0 0 6px 0; color: #292524;">Spending insights</h4>
            <p style="margin: 0; color: #78716C; font-size: 0.92rem; line-height: 1.5;">
                Understand your unique spending persona and discover where your budget unlocks maximum value.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 2. MY PROFILE
# ==============================================================================
elif st.session_state.current_page == "My Profile":
    st.markdown("## My Profile")
    st.markdown("<p style='color: #78716C; margin-bottom: 1.5rem;'>Tell us a little about your income and typical monthly spending to tailor your card recommendations.</p>", unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns([1, 1], gap="large")
    
    with col_p1:
        st.markdown("### About You")
        new_inc = st.number_input(
            "Monthly Take-Home Income (₹)",
            min_value=15000,
            max_value=1500000,
            value=st.session_state.monthly_income,
            step=5000
        )
        st.session_state.monthly_income = new_inc
        st.caption(f"Estimated Annual Income: **₹{new_inc * 12:,.0f}**")
        
        st.session_state.age = st.slider(
            "Age (Years)",
            min_value=21,
            max_value=65,
            value=st.session_state.age
        )
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Travel & Airport Lounges")
        st.caption("How many times do you visit airport lounges in a year?")
        cl1, cl2 = st.columns(2)
        with cl1:
            st.session_state.dom_lounge = st.number_input("Domestic Lounges / Year", 0, 24, st.session_state.dom_lounge)
        with cl2:
            st.session_state.intl_lounge = st.number_input("Intl Lounges / Year", 0, 12, st.session_state.intl_lounge)

    with col_p2:
        st.markdown("### Monthly Spending")
        st.markdown("<div class='beige-panel'>", unsafe_allow_html=True)
        st.caption("Enter your average monthly spending in ₹ across these categories:")
        
        st.session_state.m_groceries = st.number_input("Groceries & Supermarkets (₹)", 0, 100000, st.session_state.m_groceries, step=1000)
        st.session_state.m_dining = st.number_input("Dining & Food Delivery (₹)", 0, 100000, st.session_state.m_dining, step=1000)
        st.session_state.m_travel = st.number_input("Travel, Hotels & Flights (₹)", 0, 200000, st.session_state.m_travel, step=1000)
        st.session_state.m_online = st.number_input("Online Shopping (₹)", 0, 150000, st.session_state.m_online, step=1000)
        st.session_state.m_fuel = st.number_input("Fuel & Commuting (₹)", 0, 50000, st.session_state.m_fuel, step=500)
        st.session_state.m_entertainment = st.number_input("Entertainment & Movies (₹)", 0, 50000, st.session_state.m_entertainment, step=500)
        
        m_tot = (st.session_state.m_groceries + st.session_state.m_dining + st.session_state.m_travel + 
                 st.session_state.m_online + st.session_state.m_fuel + st.session_state.m_entertainment)
        st.markdown(f"<div style='margin-top: 12px; font-weight: 600; color: #702D3D;'>Total Monthly Card Spend: ₹{m_tot:,.0f} (₹{m_tot*12:,.0f} / year)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("Find My Cards", type="primary"):
        st.session_state.current_page = "Find My Cards"
        st.rerun()

# ==============================================================================
# 3. FIND MY CARDS (The Primary Feature)
# ==============================================================================
elif st.session_state.current_page == "Find My Cards":
    st.markdown("## Your Best Card Matches")
    st.markdown("<p style='color: #78716C; margin-bottom: 0.5rem;'>Here are the top three credit cards ranked for your spending patterns and lifestyle preferences.</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <span class="persona-badge">{current_persona['icon']} Lifestyle Persona: {current_persona['persona_name']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 3 Card Recommendations
    for rank, card in enumerate(top_recommendations, 1):
        nav_val = card['net_annual_value_inr']
        card_class = "top-match-card" if rank == 1 else "product-card"
        badge_label = "Top Match" if rank == 1 else f"Match #{rank}"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span class="maroon-pill">{badge_label}</span>
                    <h3 style="margin: 4px 0 2px 0; color: #292524;">{card['card_name']}</h3>
                    <p style="margin: 0; color: #78716C; font-size: 0.9rem;">Issued by <strong>{card['issuing_bank']}</strong></p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #78716C; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Net Value</span>
                    <div class="nav-figure">₹{nav_val:,.0f} <span style="font-size: 0.95rem; font-weight: 500; color: #78716C;">/ year</span></div>
                    <span style="font-size: 0.85rem; color: #78716C;">Annual Fee: <strong>{card['fee_display']}</strong></span>
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #E7DED5;">
                <p style="margin: 0 0 6px 0; font-weight: 600; font-size: 0.88rem; color: #44403C;">Why this card fits your spending:</p>
                <ul style="margin: 0; padding-left: 20px; font-size: 0.88rem; color: #57534E; line-height: 1.5;">
                    {''.join([f'<li>{reason}</li>' for reason in card['reasons']])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"View Benefits & Details for {card['card_name']}"):
            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown("**Estimated Category Rewards**")
                for cat, amt in card['category_rewards_inr'].items():
                    st.write(f"- {cat.replace('_', ' ').title()}: `₹{amt:,.0f}`")
                st.write(f"**Total Annual Rewards:** `₹{card['capped_rewards_inr']:,.0f}`")
            with b2:
                st.markdown("**Perks & Welcome Benefits**")
                st.write(f"- Lounge Benefit Value: `₹{card['lounge_benefit_value_inr']:,.0f}`")
                st.write(f"- Welcome Bonus: `₹{card['welcome_bonus_value']:,.0f}`")
                st.write(f"- Key Perks: _{card['key_perks']}_")
            with b3:
                st.markdown("**Annual Fee & Net Calculation**")
                st.write(f"- Standard Annual Fee: `₹{card['annual_fee']:,.0f}`")
                st.write(f"- Net Payable Fee: `₹{card['net_fee_inr']:,.0f}`")
                st.write(f"- **Final Net Annual Value:** `₹{card['net_annual_value_inr']:,.0f}`")
                st.caption(f"Cap policy: {card['caps_notes']}")

    # Simple, compact comparison table
    st.markdown("### Quick Comparison")
    comp_df = pd.DataFrame([{
        'Rank': f"#{i+1}",
        'Card Name': c['card_name'],
        'Bank': c['issuing_bank'],
        'Net Annual Value': f"₹{c['net_annual_value_inr']:,.0f}",
        'Annual Fee': c['fee_display'],
        'Estimated Rewards': f"₹{c['capped_rewards_inr']:,.0f}",
        'Lounge Benefit': f"₹{c['lounge_benefit_value_inr']:,.0f}"
    } for i, c in enumerate(top_recommendations)])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.caption("Disclaimer: Estimated rewards and net values are calculated based on your input spending patterns and standard published card terms. Issuer approval terms, merchant categorization, and eligibility criteria apply.")

# ==============================================================================
# 4. SPENDING INSIGHTS
# ==============================================================================
elif st.session_state.current_page == "Spending Insights":
    st.markdown("## Spending Insights")
    st.markdown("<p style='color: #78716C; margin-bottom: 1.5rem;'>A clear breakdown of your monthly spending and your personalized cardholder persona.</p>", unsafe_allow_html=True)
    
    # 1. Persona Summary Card
    st.markdown(f"""
    <div class="product-card" style="border-left: 4px solid #702D3D;">
        <span class="maroon-pill">Cardholder Persona</span>
        <h3 style="margin: 4px 0 6px 0; color: #292524;">{current_persona['icon']} {current_persona['persona_name']}</h3>
        <p style="margin: 0 0 10px 0; color: #57534E; font-size: 0.95rem; line-height: 1.5;">{current_persona['description']}</p>
        <span style="font-size: 0.85rem; color: #78716C;">Best Reward Categories for You: <strong>{", ".join([c.replace('_', ' ').title() for c in current_persona['priority_categories']])}</strong></span>
    </div>
    """, unsafe_allow_html=True)
    
    col_chart, col_insights = st.columns([1.2, 1], gap="large")
    
    with col_chart:
        spend_df = pd.DataFrame([
            {'Category': c.replace('_', ' ').title(), 'Monthly Spend (₹)': current_spend[c] / 12}
            for c in CATEGORIES
        ]).sort_values('Monthly Spend (₹)', ascending=True)
        
        fig = px.bar(
            spend_df,
            x='Monthly Spend (₹)',
            y='Category',
            orientation='h',
            title="Monthly Spending by Category",
            color_discrete_sequence=['#702D3D']
        )
        fig.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(family="sans-serif", color="#292524"),
            xaxis=dict(showgrid=True, gridcolor='#E7DED5', zeroline=False),
            yaxis=dict(showgrid=False),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_insights:
        st.markdown("### Spending Highlights")
        sorted_cats = sorted(
            [(c.replace('_', ' ').title(), current_spend[c] / 12) for c in CATEGORIES],
            key=lambda x: x[1],
            reverse=True
        )
        top_cat, top_amt = sorted_cats[0]
        second_cat, sec_amt = sorted_cats[1]
        m_total = sum([current_spend[c] / 12 for c in CATEGORIES])
        top_pct = (top_amt / max(1.0, m_total)) * 100
        
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E7DED5; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
            <strong>1. Primary Spending Category</strong>
            <p style="margin: 4px 0 0 0; color: #78716C; font-size: 0.9rem;">
                <strong>{top_cat}</strong> is your largest card expense, taking up <strong>{top_pct:.1f}%</strong> (₹{top_amt:,.0f}/month) of your total monthly budget.
            </p>
        </div>
        <div style="background-color: #FFFFFF; border: 1px solid #E7DED5; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
            <strong>2. High-Yield Multiplier Opportunity</strong>
            <p style="margin: 4px 0 0 0; color: #78716C; font-size: 0.9rem;">
                Your spending in <strong>{second_cat}</strong> (₹{sec_amt:,.0f}/month) makes you an ideal match for cards with accelerated category cashback.
            </p>
        </div>
        <div style="background-color: #FFFFFF; border: 1px solid #E7DED5; border-radius: 10px; padding: 16px;">
            <strong>3. Travel & Airport Comfort</strong>
            <p style="margin: 4px 0 0 0; color: #78716C; font-size: 0.9rem;">
                With {st.session_state.dom_lounge} domestic and {st.session_state.intl_lounge} international planned lounge visits, complimentary lounge access yields substantial annual savings.
            </p>
        </div>
        """, unsafe_allow_html=True)
