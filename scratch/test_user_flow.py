"""
Test automated flow of CrediMatch Streamlit AppTest
"""
import sys
from streamlit.testing.v1 import AppTest

def test_full_user_flow():
    print("[1] Initializing AppTest...")
    at = AppTest.from_file("app.py", default_timeout=25).run()
    assert not at.exception, f"App threw exception on launch: {at.exception}"
    print("[*] Initial load clean.")

    # 1. Confirm session state initially has all zeros and no recommendations
    print("[2] Verifying initial session state zero-defaults...")
    assert at.session_state.recommendations_generated == False, "Recommendations should NOT be generated initially"
    assert at.session_state.submitted_profile is None, "Submitted profile should be None initially"
    assert at.session_state.monthly_income == 0, "Monthly income should start at 0"
    assert at.session_state.age == 0, "Age should start at 0"
    assert at.session_state.m_groceries == 0, "Groceries should start at 0"
    assert at.session_state.m_dining == 0, "Dining should start at 0"
    assert at.session_state.m_travel == 0, "Travel should start at 0"
    assert at.session_state.m_online == 0, "Online should start at 0"
    assert at.session_state.m_fuel == 0, "Fuel should start at 0"
    assert at.session_state.m_entertainment == 0, "Entertainment should start at 0"
    assert at.session_state.dom_lounge == 0, "Domestic lounges should start at 0"
    assert at.session_state.intl_lounge == 0, "Intl lounges should start at 0"
    print("[+] All numeric profile fields initially show 0. Zero defaults verified.")

    # 2. Check Find My Cards page before submission
    print("[3] Checking Find My Cards empty state before submission...")
    at.session_state.current_page = "Find My Cards"
    at.session_state.nav_radio = "Find My Cards"
    at.run()
    assert not at.exception
    # Check that empty state message is present
    page_text = " ".join([m.value for m in at.markdown])
    assert "Your personalised matches will appear here." in page_text, "Empty state header missing"
    assert "Enter your profile and monthly spending details to discover which cards may suit your lifestyle." in page_text, "Empty state subtitle missing"
    # Verify Complete My Profile button is present
    comp_btns = [b for b in at.button if "Complete My Profile" in b.label]
    assert len(comp_btns) > 0, "'Complete My Profile' button must be present on empty Find My Cards"
    print("[+] Find My Cards empty state and Complete My Profile button verified.")

    # 3. Check Spending Insights page before submission
    print("[4] Checking Spending Insights empty state before submission...")
    at.session_state.current_page = "Spending Insights"
    at.session_state.nav_radio = "Spending Insights"
    at.run()
    assert not at.exception
    page_text = " ".join([m.value for m in at.markdown])
    assert "Your spending insights will appear here." in page_text, "Spending Insights empty state missing"
    print("[+] Spending Insights empty state verified.")

    # 4. Navigate to My Profile and attempt submission with all zeros
    print("[5] Testing submission on My Profile with all zeroes...")
    at.session_state.current_page = "My Profile"
    at.session_state.nav_radio = "My Profile"
    at.run()
    assert not at.exception
    
    # Confirm numeric inputs on page are 0
    for num_input in at.number_input:
        assert num_input.value == 0, f"Number input {num_input.label} value was {num_input.value}, expected 0"
    print("[+] My Profile number inputs verified as 0.")

    # Click 'Find My Cards' button while spending is 0
    find_cards_btn = [b for b in at.button if b.label == "Find My Cards"][0]
    find_cards_btn.click().run()
    assert not at.exception
    # Should show validation error
    assert len(at.error) > 0, "Validation error should appear for 0 spending"
    err_text = " ".join([e.value for e in at.error])
    assert "Please enter your estimated monthly spending in at least one category" in err_text
    assert at.session_state.recommendations_generated == False, "Recommendations must NOT be generated on 0 spend"
    print("[+] Validation error for all-zero profile verified successfully.")

    # 5. Enter a valid profile with zero in some categories (e.g., fuel=0, travel=0)
    print("[6] Entering valid spending profile with individual zero categories...")
    # Fill in inputs
    for ni in at.number_input:
        if "Income" in ni.label:
            ni.set_value(65000)
        elif "Age" in ni.label:
            ni.set_value(29)
        elif "Groceries" in ni.label:
            ni.set_value(16000)
        elif "Dining" in ni.label:
            ni.set_value(8000)
        elif "Online" in ni.label:
            ni.set_value(14000)
        elif "Domestic" in ni.label:
            ni.set_value(2)
        # fuel, travel, entertainment, intl_lounge remain 0
    
    at.run()
    assert not at.exception
    
    # Click 'Find My Cards'
    find_cards_btn = [b for b in at.button if b.label == "Find My Cards"][0]
    find_cards_btn.click().run()
    assert not at.exception

    # Verify recommendations are generated and page navigated to Find My Cards
    print("[7] Verifying recommendations generated...")
    assert at.session_state.recommendations_generated == True, "Recommendations should be generated now"
    assert at.session_state.current_page == "Find My Cards", f"Expected Find My Cards, got {at.session_state.current_page}"
    assert len(at.session_state.stored_recommendations) == 3, "Expected exactly 3 recommended cards"
    
    recs = at.session_state.stored_recommendations
    print(f"[*] Top 3 Cards Recommended: {[c['card_name'] for c in recs]}")
    for c in recs:
        assert c['net_annual_value_inr'] is not None
        assert 'capped_rewards_inr' in c
        assert len(c['reasons']) > 0
    print("[+] Top 3 personalised recommendations successfully generated and rendered.")

    # 6. Check Spending Insights page now shows results
    print("[8] Verifying Spending Insights after submission...")
    at.session_state.current_page = "Spending Insights"
    at.session_state.nav_radio = "Spending Insights"
    at.run()
    assert not at.exception
    si_text = " ".join([m.value for m in at.markdown])
    assert "Cardholder Persona" in si_text
    assert "Primary Spending Category" in si_text
    print("[+] Spending Insights successfully populated.")

    # 7. Modify profile spending and resubmit
    print("[9] Modifying spending values and resubmitting...")
    at.session_state.current_page = "My Profile"
    at.session_state.nav_radio = "My Profile"
    at.run()
    assert not at.exception

    # Change groceries and add travel
    for ni in at.number_input:
        if "Groceries" in ni.label:
            ni.set_value(28000)
        elif "Travel" in ni.label:
            ni.set_value(20000)
            
    at.run()
    assert not at.exception

    # Resubmit
    find_cards_btn = [b for b in at.button if b.label == "Find My Cards"][0]
    find_cards_btn.click().run()
    assert not at.exception

    updated_recs = at.session_state.stored_recommendations
    print(f"[*] Updated Top 3 Cards: {[c['card_name'] for c in updated_recs]}")
    assert updated_recs[0]['net_annual_value_inr'] != recs[0]['net_annual_value_inr'] or updated_recs[0]['card_name'] != recs[0]['card_name'], "Recommendations should update upon new spending inputs"
    print("[+] Resubmission with updated values successfully refreshed recommendations.")

    print("\n=======================================================")
    print("ALL 8 USER FLOW REQUIREMENTS SUCCESSFULLY VALIDATED!")
    print("=======================================================")

if __name__ == "__main__":
    test_full_user_flow()
