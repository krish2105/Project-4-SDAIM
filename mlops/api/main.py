from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download

# Scikit-Learn 1.6+ compatibility patch for pickled XGBClassifier models
try:
    from sklearn.base import ClassifierMixin
    from sklearn.utils._tags import Tags, TargetTags, ClassifierTags
    ClassifierMixin.__sklearn_tags__ = lambda self: Tags(
        estimator_type='classifier',
        target_tags=TargetTags(required=False),
        transformer_tags=None,
        regressor_tags=None,
        classifier_tags=ClassifierTags()
    )
except Exception:
    pass

from mlops.analytics.shap_explainer import calculate_shap_contributions
from mlops.analytics.roi_calculator import calculate_clv, calculate_expected_retention_roi
from mlops.monitoring.drift_monitor import run_drift_analysis

app = FastAPI(
    title="E-Commerce Customer Churn Intelligence Microservice",
    description="Enterprise REST API for Shopper Churn Analytics, SHAP Attribution & Promo ROI Optimization",
    version="1.0.0"
)

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        if os.path.exists("best_churn_model.joblib"):
            MODEL = joblib.load("best_churn_model.joblib")
        else:
            try:
                model_path = hf_hub_download(
                    repo_id="krish21may/Bank-Customer-Churn-4", 
                    filename="best_churn_model.joblib"
                )
                MODEL = joblib.load(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
    return MODEL

class CustomerFeatures(BaseModel):
    Tenure: int = Field(..., ge=0, le=72, example=12)
    WarehouseToHome: int = Field(..., ge=1, le=150, example=15)
    HourSpendOnApp: int = Field(..., ge=1, le=8, example=3)
    NumberOfDeviceRegistered: int = Field(..., ge=1, le=8, example=3)
    SatisfactionScore: int = Field(..., ge=1, le=5, example=3)
    Complain: int = Field(..., ge=0, le=1, example=0)
    OrderAmountHikeFromlastYear: int = Field(..., ge=0, le=50, example=15)
    DaySinceLastOrder: int = Field(..., ge=0, le=60, example=8)
    CashBackAmount: float = Field(..., ge=0.0, example=160.0)
    CityTier: int = Field(..., ge=1, le=3, example=1)
    PreferredPaymentMode: str = Field(..., example="Debit Card")
    Gender: str = Field(..., example="Female")
    PreferedOrderCat: str = Field(..., example="Laptop & Accessory")
    MaritalStatus: str = Field(..., example="Married")

@app.get("/v1/health")
def health_check():
    model = get_model()
    return {
        "status": "online",
        "model_loaded": model is not None,
        "domain": "E-Commerce Customer Churn Intelligence"
    }

@app.post("/v1/predict")
def predict_single_customer(data: CustomerFeatures):
    model = get_model()
    input_df = pd.DataFrame([data.dict()])
    
    try:
        prob = float(model.predict_proba(input_df)[0, 1])
        is_churn = prob >= 0.45
        
        shap_df = calculate_shap_contributions(model, input_df)
        shap_records = shap_df.to_dict(orient="records")
        
        clv = calculate_clv(data.CashBackAmount, data.OrderAmountHikeFromlastYear, data.Tenure, data.SatisfactionScore)
        roi_info = calculate_expected_retention_roi(prob, clv)
        
        return {
            "Churn_Probability": round(prob, 4),
            "Churn_Probability_%": round(prob * 100, 2),
            "Predicted_Status": "HIGH CHURN RISK ⚠️" if is_churn else "LOW CHURN RISK ✅",
            "SHAP_Feature_Attributions": shap_records,
            "Financial_Metrics": roi_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/predict-batch")
def predict_batch_customers(customers: list[CustomerFeatures]):
    model = get_model()
    input_df = pd.DataFrame([c.dict() for c in customers])
    
    try:
        probs = model.predict_proba(input_df)[:, 1]
        input_df['Churn_Probability'] = np.round(probs, 4)
        input_df['Churn_Probability_%'] = np.round(probs * 100, 2)
        input_df['Predicted_Status'] = np.where(probs >= 0.45, "HIGH RISK ⚠️", "SAFE ✅")
        
        return {
            "Total_Processed": len(input_df),
            "High_Risk_Count": int(np.sum(probs >= 0.45)),
            "Average_Churn_Risk_%": round(float(np.mean(probs * 100)), 2),
            "Predictions": input_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/drift-check")
def check_data_drift(customers: list[CustomerFeatures]):
    input_df = pd.DataFrame([c.dict() for c in customers])
    result = run_drift_analysis(input_df)
    return result
