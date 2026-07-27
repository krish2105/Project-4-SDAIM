import pandas as pd
import numpy as np

def calculate_clv(balance, salary, num_products, tenure):
    """
    Estimates annual Customer Lifetime Value (CLV) for a bank customer.
    """
    annual_margin_balance = float(balance) * 0.035  # Net interest margin ~3.5%
    annual_margin_salary = float(salary) * 0.025   # Fee income from transactions
    product_multiplier = float(num_products) * 150.0
    tenure_loyalty_factor = min(1.5, 1.0 + (float(tenure) * 0.05))
    
    clv = (annual_margin_balance + annual_margin_salary + product_multiplier) * tenure_loyalty_factor
    return max(300.0, float(clv))

def calculate_expected_retention_roi(churn_prob, clv, campaign_cost=150.0, retention_success_rate=0.40):
    """
    Calculates expected net profit from intervening with a retention campaign offer.
    """
    # Expected Loss without intervention
    expected_loss = churn_prob * clv
    
    # Expected Value saved with campaign intervention
    saved_value = churn_prob * retention_success_rate * clv
    net_campaign_profit = saved_value - campaign_cost
    
    roi_percent = (net_campaign_profit / campaign_cost) * 100 if campaign_cost > 0 else 0.0
    
    return {
        'Customer_CLV': round(clv, 2),
        'Expected_Unmitigated_Loss': round(expected_loss, 2),
        'Expected_Saved_Value': round(saved_value, 2),
        'Retention_Campaign_Cost': round(campaign_cost, 2),
        'Net_Financial_Impact': round(net_campaign_profit, 2),
        'Campaign_ROI_%': round(roi_percent, 1),
        'Recommend_Campaign': net_campaign_profit > 0 and churn_prob >= 0.35
    }

def optimize_decision_threshold(df_predictions, campaign_cost=150.0, success_rate=0.40):
    """
    Finds the optimal classification threshold that maximizes total bank profit.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    profits = []
    
    for t in thresholds:
        targeted = df_predictions[df_predictions['Churn_Probability'] >= t]
        if len(targeted) == 0:
            profits.append(0.0)
            continue
            
        # Total CLV of targeted at-risk customers
        total_clv = targeted['CLV'].sum()
        total_cost = len(targeted) * campaign_cost
        total_saved = (targeted['Churn_Probability'] * success_rate * targeted['CLV']).sum()
        net_profit = total_saved - total_cost
        profits.append(net_profit)
        
    best_idx = np.argmax(profits)
    return {
        'Optimal_Threshold': round(thresholds[best_idx], 2),
        'Max_Net_Profit': round(profits[best_idx], 2),
        'Threshold_Curve': pd.DataFrame({'Threshold': thresholds, 'Net_Profit': profits})
    }
