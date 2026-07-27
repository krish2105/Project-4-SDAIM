import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp

def run_drift_analysis(current_df, reference_csv="mlops/data/bank_customer_churn.csv"):
    """
    Analyzes statistical Data Drift between reference baseline and production traffic.
    
    Args:
        current_df: DataFrame of recent inference inputs / batch payload
        reference_csv: Path to baseline dataset
        
    Returns:
        dict containing Overall Drift Index, drifted feature list, and per-feature p-values.
    """
    if not os.path.exists(reference_csv):
        return {
            'Status': 'Baseline Dataset Missing',
            'Drift_Detected': False,
            'Drift_Share': 0.0,
            'Drifted_Features': []
        }
        
    ref_df = pd.read_csv(reference_csv)
    
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    
    drifted_features = []
    feature_metrics = {}
    
    for feat in numeric_features:
        if feat in current_df.columns and feat in ref_df.columns:
            ref_data = ref_df[feat].dropna()
            curr_data = current_df[feat].dropna()
            
            if len(curr_data) > 1:
                # Kolmogorov-Smirnov 2-sample statistical test
                stat, p_val = ks_2samp(ref_data, curr_data)
                is_drifted = p_val < 0.05
                
                if is_drifted:
                    drifted_features.append(feat)
                    
                feature_metrics[feat] = {
                    'p_value': round(float(p_val), 4),
                    'statistic': round(float(stat), 4),
                    'drifted': is_drifted
                }
                
    drift_share = len(drifted_features) / len(numeric_features) if numeric_features else 0.0
    overall_drift = drift_share >= 0.33  # Significant if > 33% features drifted
    
    return {
        'Status': 'Analysis Complete',
        'Drift_Detected': overall_drift,
        'Drift_Share_%': round(drift_share * 100, 1),
        'Drifted_Features': drifted_features,
        'Feature_Metrics': feature_metrics
    }
