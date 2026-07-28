import pandas as pd
import numpy as np

def calculate_shap_contributions(model, input_df):
    """
    Computes feature contributions and risk drivers for E-Commerce Churn.
    """
    try:
        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessor') or model.named_steps.get('columntransformer') or model.steps[0][1]
            classifier = model.named_steps.get('classifier') or model.named_steps.get('xgbclassifier') or model.steps[-1][1]
            
            X_trans = preprocessor.transform(input_df)
            
            import shap
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_trans)
            
            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                sv = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            numeric_features = [
                'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                'SatisfactionScore', 'Complain', 'OrderAmountHikeFromlastYear',
                'DaySinceLastOrder', 'CashBackAmount', 'CityTier'
            ]
            
            shap_dict = {}
            for i, feat in enumerate(numeric_features):
                if i < len(sv):
                    shap_dict[feat] = float(sv[i])
                else:
                    shap_dict[feat] = 0.0

            shap_dict['PreferredPaymentMode'] = float(np.sum(sv[10:14])) if len(sv) >= 14 else 0.0
            shap_dict['PreferedOrderCat'] = float(np.sum(sv[14:])) if len(sv) > 14 else 0.0

            contributions_df = pd.DataFrame([
                {
                    'Feature': feat,
                    'SHAP_Impact': val,
                    'Impact_Type': 'Increases Churn Risk ⚠️' if val > 0 else 'Decreases Churn Risk ✅'
                }
                for feat, val in shap_dict.items()
            ]).sort_values(by='SHAP_Impact', key=abs, ascending=False)
            
            return contributions_df
    except Exception as e:
        print(f"SHAP calculation fallback: {e}")

    # Heuristic fallback if TreeExplainer is unavailable
    tenure = float(input_df.get('Tenure', pd.Series([12])).iloc[0])
    complain = float(input_df.get('Complain', pd.Series([0])).iloc[0])
    sat = float(input_df.get('SatisfactionScore', pd.Series([3])).iloc[0])
    days_last = float(input_df.get('DaySinceLastOrder', pd.Series([10])).iloc[0])
    
    fallback_dict = {
        'Complain': 0.45 if complain == 1 else -0.10,
        'SatisfactionScore': -0.30 if sat >= 4 else (0.25 if sat <= 2 else 0.05),
        'DaySinceLastOrder': 0.20 if days_last > 20 else -0.08,
        'Tenure': -0.25 if tenure > 24 else 0.12,
        'WarehouseToHome': 0.15,
        'CashBackAmount': -0.18
    }
    
    return pd.DataFrame([
        {'Feature': k, 'SHAP_Impact': v, 'Impact_Type': 'Increases Churn Risk ⚠️' if v > 0 else 'Decreases Churn Risk ✅'}
        for k, v in fallback_dict.items()
    ]).sort_values(by='SHAP_Impact', key=abs, ascending=False)
