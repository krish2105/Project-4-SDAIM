import pandas as pd
import numpy as np

def segment_causal_uplift(df_predictions):
    """
    Categorizes customers into Causal ML Uplift Segments for Campaign Profit Optimization.
    
    Segments:
    - Persuadables: Moderate/High Risk + High Balance/Salary -> High Campaign Uplift (Target Here!)
    - Sure Things: Low Churn Risk -> Do Not Spend Budget
    - Lost Causes: Very High Risk (>85%) + Inactive -> Low Uplift
    - Sleeping Dogs: Low Risk + Inactive -> Risk of disturbance if contacted
    """
    df = df_predictions.copy()
    
    causal_segments = []
    uplift_scores = []
    
    for _, row in df.iterrows():
        p = float(row.get('Churn_Probability', row.get('Churn_Probability_%', 0) / 100.0 if 'Churn_Probability_%' in row else 0.5))
        is_active = row.get('IsActiveMember', 1)
        balance = float(row.get('Balance', 50000))
        
        # Causal Classification Heuristics
        if 0.35 <= p <= 0.80 and balance > 10000:
            segment = "🎯 Persuadable (High Campaign ROI)"
            uplift = round(p * 0.55, 2)
        elif p < 0.35:
            segment = "🔒 Sure Thing (Low Churn Risk)"
            uplift = round(p * 0.05, 2)
        elif p > 0.80 and is_active == 0:
            segment = "❌ Lost Cause (Ineffective Target)"
            uplift = round(p * 0.10, 2)
        else:
            segment = "⚠️ Sleeping Dog (Do Not Disturb)"
            uplift = round(-0.15, 2)
            
        causal_segments.append(segment)
        uplift_scores.append(uplift)
        
    df['Causal_Segment'] = causal_segments
    df['Uplift_Score'] = uplift_scores
    
    return df
