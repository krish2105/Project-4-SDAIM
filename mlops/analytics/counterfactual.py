import pandas as pd
import numpy as np

def generate_counterfactual_scenarios(model, input_df, target_threshold=0.45):
    """
    Generates prescriptive counterfactual "What-If" recourse scenarios.
    
    Args:
        model: Sklearn/XGBoost Pipeline model
        input_df: DataFrame containing a single customer's features
        target_threshold: Churn probability cutoff for safe retention
        
    Returns:
        List of actionable recourse scenarios with new risk probability and net risk reduction.
    """
    base_prob = float(model.predict_proba(input_df)[0, 1])
    customer = input_df.iloc[0].to_dict()
    
    scenarios = []
    
    # -------------------------------------------------------------
    # Action Scenario 1: Digital Engagement & Active Membership
    # -------------------------------------------------------------
    s1_df = input_df.copy()
    s1_df['IsActiveMember'] = 1
    if s1_df['NumOfProducts'].iloc[0] == 1:
        s1_df['NumOfProducts'] = 2
    prob1 = float(model.predict_proba(s1_df)[0, 1])
    
    scenarios.append({
        'Scenario_Name': '📱 Engagement & Product Activation',
        'Actions_Required': [
            'Activate Customer Digital Banking & Active Membership status',
            'Cross-sell 2nd core bank product (e.g., High-Yield Savings Account)'
        ],
        'Original_Risk_%': round(base_prob * 100, 1),
        'New_Risk_%': round(prob1 * 100, 1),
        'Risk_Reduction_%': round((base_prob - prob1) * 100, 1),
        'Achieves_Safe_Status': prob1 < target_threshold
    })
    
    # -------------------------------------------------------------
    # Action Scenario 2: Deposit Incentive & Loyalty Balance Boost
    # -------------------------------------------------------------
    s2_df = input_df.copy()
    current_bal = float(s2_df['Balance'].iloc[0])
    s2_df['Balance'] = current_bal + 15000.0
    s2_df['IsActiveMember'] = 1
    prob2 = float(model.predict_proba(s2_df)[0, 1])
    
    scenarios.append({
        'Scenario_Name': '💰 Deposit Growth & Rate Bonus',
        'Actions_Required': [
            'Offer promotional 4.25% APY deposit rate bonus',
            'Encourage +$15,000 balance increase from primary salary account'
        ],
        'Original_Risk_%': round(base_prob * 100, 1),
        'New_Risk_%': round(prob2 * 100, 1),
        'Risk_Reduction_%': round((base_prob - prob2) * 100, 1),
        'Achieves_Safe_Status': prob2 < target_threshold
    })
    
    # -------------------------------------------------------------
    # Action Scenario 3: Comprehensive Executive Relationship Offer
    # -------------------------------------------------------------
    s3_df = input_df.copy()
    s3_df['IsActiveMember'] = 1
    s3_df['NumOfProducts'] = 2
    s3_df['HasCrCard'] = 1
    s3_df['Balance'] = float(s3_df['Balance'].iloc[0]) + 10000.0
    prob3 = float(model.predict_proba(s3_df)[0, 1])
    
    scenarios.append({
        'Scenario_Name': '🌟 Premium Executive VIP Relationship Package',
        'Actions_Required': [
            'Assign Dedicated Personal Relationship Manager',
            'Waive Annual Credit Card & Account Maintenance Fees',
            'Activate 2nd Product + $10,000 Deposit Incentive'
        ],
        'Original_Risk_%': round(base_prob * 100, 1),
        'New_Risk_%': round(prob3 * 100, 1),
        'Risk_Reduction_%': round((base_prob - prob3) * 100, 1),
        'Achieves_Safe_Status': prob3 < target_threshold
    })
    
    return scenarios
