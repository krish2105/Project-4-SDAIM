import pandas as pd
import numpy as np

def segment_causal_uplift(df_predictions):
    """
    Categorizes shoppers into Causal ML Uplift Segments for E-Commerce Campaign Profit Optimization.
    """
    df = df_predictions.copy()
    
    causal_segments = []
    uplift_scores = []
    
    for _, row in df.iterrows():
        p = float(row.get('Churn_Probability', row.get('Churn_Probability_%', 0) / 100.0 if 'Churn_Probability_%' in row else 0.5))
        complain = int(row.get('Complain', 0))
        cashback = float(row.get('CashBackAmount', 150))
        
        # Causal Classification Heuristics
        if 0.35 <= p <= 0.80 and cashback > 50:
            segment = "🎯 Persuadable (Target for $50 Coupon)"
            uplift = round(p * 0.55, 2)
        elif p < 0.35:
            segment = "🔒 Sure Thing (Organic Repeat Shopper)"
            uplift = round(p * 0.05, 2)
        elif p > 0.80 and complain == 1:
            segment = "❌ Lost Cause (Severe Complaint Unresolved)"
            uplift = round(p * 0.10, 2)
        else:
            segment = "⚠️ Sleeping Dog (Low Active Engagement)"
            uplift = round(-0.15, 2)
            
        causal_segments.append(segment)
        uplift_scores.append(uplift)
        
    df['Causal_Segment'] = causal_segments
    df['Uplift_Score'] = uplift_scores
    
    return df
