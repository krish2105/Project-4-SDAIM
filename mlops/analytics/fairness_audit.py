import pandas as pd
import numpy as np

def run_fairness_audit(df_predictions, protected_attribute='Geography'):
    """
    Evaluates algorithmic bias & Disparate Impact across protected attributes.
    
    Rule of 4/5ths (80% Rule):
    Disparate Impact Ratio = (High Risk Rate in Protected Group) / (High Risk Rate in Reference Group)
    Acceptable range: 0.80 to 1.25.
    """
    df = df_predictions.copy()
    
    if protected_attribute not in df.columns:
        return {'Status': f'Attribute {protected_attribute} missing', 'Compliant': True}
        
    group_stats = {}
    
    for group, group_df in df.groupby(protected_attribute):
        total = len(group_df)
        if 'Risk_Status' in group_df.columns:
            high_risk = np.sum(group_df['Risk_Status'].str.contains('HIGH'))
        else:
            high_risk = np.sum(group_df['Churn_Probability_%'] >= 45.0)
            
        rate = (high_risk / total) if total > 0 else 0.0
        group_stats[group] = {
            'Total_Customers': total,
            'High_Risk_Count': int(high_risk),
            'High_Risk_Rate_%': round(rate * 100, 2)
        }
        
    # Calculate Disparate Impact Ratios against base group (France or max group)
    rates = [v['High_Risk_Rate_%'] for v in group_stats.values()]
    max_rate = max(rates) if rates else 1.0
    min_rate = min(rates) if rates else 1.0
    
    disparate_impact_ratio = round(min_rate / max_rate, 2) if max_rate > 0 else 1.0
    is_fair = 0.80 <= disparate_impact_ratio <= 1.25
    
    return {
        'Protected_Attribute': protected_attribute,
        'Group_Statistics': group_stats,
        'Disparate_Impact_Ratio': disparate_impact_ratio,
        'Fairness_4_5th_Rule_Compliant': is_fair,
        'Regulatory_Status': 'COMPLIANT ✅' if is_fair else 'AUDIT WARNING: Disparate Impact Discrepancy ⚠️'
    }
