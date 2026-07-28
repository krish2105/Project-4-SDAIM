import pandas as pd
import numpy as np

def generate_counterfactual_scenarios(model, input_df, target_threshold=0.45):
    """
    Generates prescriptive counterfactual "What-If" recourse scenarios for E-Commerce Churn.
    """
    base_prob = float(model.predict_proba(input_df)[0, 1])
    scenarios = []
    
    # -------------------------------------------------------------
    # Scenario 1: CashBack Loyalty & Coupon Incentive
    # -------------------------------------------------------------
    s1_df = input_df.copy()
    current_cashback = float(s1_df['CashBackAmount'].iloc[0])
    s1_df['CashBackAmount'] = current_cashback + 50.0
    s1_df['Complain'] = 0
    prob1 = float(model.predict_proba(s1_df)[0, 1])
    
    scenarios.append({
        'Scenario_Name': '🎁 CashBack Incentive & Complaint Resolution',
        'Actions_Required': [
            'Offer $50 CashBack bonus on next checkout',
            'Provide dedicated VIP customer support to resolve active complaints'
        ],
        'Original_Risk_%': round(base_prob * 100, 1),
        'New_Risk_%': round(prob1 * 100, 1),
        'Risk_Reduction_%': round((base_prob - prob1) * 100, 1),
        'Achieves_Safe_Status': prob1 < target_threshold
    })
    
    # -------------------------------------------------------------
    # Scenario 2: Priority Free Shipping & Logistics Upgrade
    # -------------------------------------------------------------
    s2_df = input_df.copy()
    s2_df['Tenure'] = s2_df['Tenure'] + 3
    s2_df['Complain'] = 0
    prob2 = float(model.predict_proba(s2_df)[0, 1])
    
    scenarios.append({
        'Scenario_Name': '🚚 Express Shipping & Subscription Upgrade',
        'Actions_Required': [
            'Enroll in 3-Month Free Prime Express Delivery tier',
            'Waive shipping fees for orders near warehouse hubs'
        ],
        'Original_Risk_%': round(base_prob * 100, 1),
        'New_Risk_%': round(prob2 * 100, 1),
        'Risk_Reduction_%': round((base_prob - prob2) * 100, 1),
        'Achieves_Safe_Status': prob2 < target_threshold
    })
    
    return scenarios
