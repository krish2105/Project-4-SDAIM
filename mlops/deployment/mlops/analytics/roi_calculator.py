import pandas as pd
import numpy as np

def calculate_clv(cashback, order_hike, tenure, satisfaction_score):
    """
    Estimates annual Customer Lifetime Value (CLV) for an E-Commerce shopper.
    """
    base_annual_spend = 800.0 + (float(cashback) * 3.5)
    growth_multiplier = 1.0 + (float(order_hike) / 100.0)
    tenure_loyalty_factor = min(1.6, 1.0 + (float(tenure) * 0.04))
    satisfaction_factor = 0.8 + (float(satisfaction_score) * 0.1)
    
    clv = base_annual_spend * growth_multiplier * tenure_loyalty_factor * satisfaction_factor
    return max(250.0, float(clv))

def calculate_expected_retention_roi(churn_prob, clv, campaign_cost=50.0, retention_success_rate=0.45):
    """
    Calculates expected net profit from intervening with a coupon/discount retention campaign offer.
    """
    expected_loss = churn_prob * clv
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

def optimize_decision_threshold(df_predictions, campaign_cost=50.0, success_rate=0.45):
    """
    Finds the optimal classification threshold that maximizes total e-commerce profit.
    """
    df = df_predictions.copy()
    
    if 'Churn_Probability' in df.columns:
        prob_series = df['Churn_Probability']
    elif 'Churn_Probability_%' in df.columns:
        prob_series = df['Churn_Probability_%'] / 100.0
        df['Churn_Probability'] = prob_series
    else:
        prob_series = pd.Series(0.5, index=df.index)
        df['Churn_Probability'] = prob_series

    if 'CLV' in df.columns:
        clv_series = df['CLV']
    elif 'CLV_$' in df.columns:
        clv_series = df['CLV_$']
        df['CLV'] = clv_series
    else:
        clv_series = df.apply(
            lambda r: calculate_clv(
                r.get('CashBackAmount', 150),
                r.get('OrderAmountHikeFromlastYear', 15),
                r.get('Tenure', 12),
                r.get('SatisfactionScore', 3)
            ), axis=1
        )
        df['CLV'] = clv_series

    thresholds = np.linspace(0.10, 0.90, 81)
    results = []
    
    for th in thresholds:
        target_indices = df['Churn_Probability'] >= th
        n_targeted = np.sum(target_indices)
        
        targeted_probs = df.loc[target_indices, 'Churn_Probability']
        targeted_clvs = df.loc[target_indices, 'CLV']
        
        total_cost = n_targeted * campaign_cost
        gross_saved = np.sum(targeted_probs * success_rate * targeted_clvs)
        net_profit = gross_saved - total_cost
        
        results.append({
            'Threshold': round(th, 2),
            'Targeted_Customers': int(n_targeted),
            'Campaign_Cost_$': round(total_cost, 2),
            'Gross_Saved_CLV_$': round(gross_saved, 2),
            'Net_Profit_$': round(net_profit, 2)
        })
        
    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['Net_Profit_$'].idxmax()]
    
    return {
        'Optimal_Threshold': float(best_row['Threshold']),
        'Max_Net_Profit_$': float(best_row['Net_Profit_$']),
        'Targeted_Count': int(best_row['Targeted_Customers']),
        'Threshold_Curve_DF': res_df
    }
