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
    df = df_predictions.copy()
    
    # Resolve Churn Probability Column
    if 'Churn_Probability' in df.columns:
        prob_series = df['Churn_Probability']
    elif 'Churn_Probability_%' in df.columns:
        prob_series = df['Churn_Probability_%'] / 100.0
        df['Churn_Probability'] = prob_series
    else:
        prob_series = pd.Series(0.5, index=df.index)
        df['Churn_Probability'] = prob_series

    # Resolve CLV Column
    if 'CLV' in df.columns:
        clv_series = df['CLV']
    elif 'CLV_$' in df.columns:
        clv_series = df['CLV_$']
        df['CLV'] = clv_series
    else:
        clv_series = pd.Series([
            calculate_clv(r.get('Balance', 50000), r.get('EstimatedSalary', 75000), r.get('NumOfProducts', 1), r.get('Tenure', 5))
            for _, r in df.iterrows()
        ], index=df.index)
        df['CLV'] = clv_series

    thresholds = np.linspace(0.10, 0.90, 81)
    profits = []
    
    for t in thresholds:
        targeted = df[df['Churn_Probability'] >= t]
        if len(targeted) == 0:
            profits.append(0.0)
            continue
            
        # Total CLV of targeted at-risk customers
        total_cost = len(targeted) * campaign_cost
        total_saved = (targeted['Churn_Probability'] * success_rate * targeted['CLV']).sum()
        net_profit = total_saved - total_cost
        profits.append(float(net_profit))
        
    best_idx = int(np.argmax(profits))
    return {
        'Optimal_Threshold': round(float(thresholds[best_idx]), 2),
        'Max_Net_Profit': round(float(profits[best_idx]), 2),
        'Threshold_Curve': pd.DataFrame({'Threshold': thresholds, 'Net_Profit': profits})
    }
