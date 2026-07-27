import pandas as pd
import numpy as np

def predict_survival_timeline(churn_probability, tenure_years=3):
    """
    Computes a 24-month survival probability curve and expected customer lifespan.
    
    Args:
        churn_probability: Model predicted 1-year churn probability (0.0 to 1.0)
        tenure_years: Current customer tenure with the bank
        
    Returns:
        dict containing Expected_Months_To_Churn, Survival_Curve DataFrame, and Hazard Risk Category.
    """
    # Monthly baseline hazard rate derived from churn probability
    monthly_hazard = -np.log(1.0 - min(0.99, max(0.01, churn_probability))) / 12.0
    
    months = np.arange(1, 25)
    survival_probs = np.exp(-monthly_hazard * months)
    
    # Estimate median survival months
    if churn_probability > 0.50:
        expected_months = int(max(1, round(6.0 / (churn_probability * 1.5))))
    else:
        expected_months = int(round(24.0 / max(0.2, churn_probability)))
        
    timeline_df = pd.DataFrame({
        'Month': months,
        'Survival_Probability_%': np.round(survival_probs * 100, 2),
        'Cumulative_Hazard_%': np.round((1.0 - survival_probs) * 100, 2)
    })
    
    if churn_probability >= 0.65:
        hazard_cat = "CRITICAL (0 - 3 Months Risk) 🚨"
    elif churn_probability >= 0.45:
        hazard_cat = "ELEVATED (3 - 6 Months Risk) ⚠️"
    else:
        hazard_cat = "LOW HAZARD (24+ Months Lifespan) ✅"
        
    return {
        'Expected_Months_Until_Churn': expected_months,
        'Hazard_Risk_Category': hazard_cat,
        'Survival_Curve_DF': timeline_df,
        'Prob_Survival_6M_%': round(float(survival_probs[5]) * 100, 1),
        'Prob_Survival_12M_%': round(float(survival_probs[11]) * 100, 1),
        'Prob_Survival_24M_%': round(float(survival_probs[23]) * 100, 1)
    }
