from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download

from mlops.analytics.shap_explainer import calculate_shap_contributions
from mlops.analytics.roi_calculator import calculate_clv, calculate_expected_retention_roi
from mlops.monitoring.drift_monitor import run_drift_analysis

app = FastAPI(
    title="Bank Customer Churn Intelligence Microservice",
    description="Enterprise REST API for Churn Risk Analytics, SHAP Attribution & Financial ROI Optimization",
    version="1.0.0"
)

# Global Model Cache
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        try:
            model_path = hf_hub_download(
                repo_id="krish21may/Bank-Customer-Churn-4", 
                filename="best_churn_model.joblib"
            )
            MODEL = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from Hugging Face: {e}")
    return MODEL

# Input Schema Validation
class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, example=650)
    Geography: str = Field(..., example="France")
    Age: int = Field(..., ge=18, le=100, example=42)
    Tenure: int = Field(..., ge=0, le=20, example=5)
    Balance: float = Field(..., ge=0.0, example=75000.0)
    NumOfProducts: int = Field(..., ge=1, le=4, example=1)
    HasCrCard: int = Field(..., ge=0, le=1, example=1)
    IsActiveMember: int = Field(..., ge=0, le=1, example=0)
    EstimatedSalary: float = Field(..., ge=0.0, example=60000.0)

class BatchCustomerPayload(BaseModel):
    customers: list[CustomerFeatures]

@app.get("/")
def read_root():
    return {
        "service": "Bank Customer Churn Intelligence REST API",
        "status": "Online",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/v1/health")
def health_check():
    model = get_model()
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/v1/predict")
def predict_single(customer: CustomerFeatures):
    model = get_model()
    input_df = pd.DataFrame([customer.dict()])
    
    # Inference
    prob = float(model.predict_proba(input_df)[0, 1])
    threshold = 0.45
    is_churn = prob >= threshold
    
    # Financial CLV & ROI Calculation
    clv = calculate_clv(customer.Balance, customer.EstimatedSalary, customer.NumOfProducts, customer.Tenure)
    roi_metrics = calculate_expected_retention_roi(prob, clv)
    
    # SHAP Feature Drivers
    shap_df = calculate_shap_contributions(model, input_df)
    top_risk_drivers = shap_df.head(3).to_dict(orient='records')
    
    return {
        "churn_probability": round(prob, 4),
        "churn_probability_percent": round(prob * 100, 2),
        "threshold": threshold,
        "risk_status": "HIGH CHURN RISK" if is_churn else "LOW CHURN RISK",
        "clv_estimate_usd": clv,
        "financial_roi_analysis": roi_metrics,
        "top_shap_risk_drivers": top_risk_drivers
    }

@app.post("/v1/predict-batch")
def predict_batch(payload: BatchCustomerPayload):
    model = get_model()
    input_df = pd.DataFrame([c.dict() for c in payload.customers])
    
    probs = model.predict_proba(input_df)[:, 1]
    threshold = 0.45
    preds = (probs >= threshold).astype(int)
    
    results = []
    for i, row in input_df.iterrows():
        p = float(probs[i])
        clv = calculate_clv(row['Balance'], row['EstimatedSalary'], row['NumOfProducts'], row['Tenure'])
        results.append({
            "record_id": i + 1,
            "churn_probability_%": round(p * 100, 2),
            "risk_status": "HIGH RISK" if preds[i] == 1 else "LOW RISK",
            "clv_usd": round(clv, 2)
        })
        
    return {
        "total_records": len(results),
        "high_risk_count": int(np.sum(preds == 1)),
        "avg_churn_probability_%": round(float(np.mean(probs)) * 100, 2),
        "predictions": results
    }

@app.post("/v1/drift-check")
def drift_check(payload: BatchCustomerPayload):
    input_df = pd.DataFrame([c.dict() for c in payload.customers])
    analysis = run_drift_analysis(input_df)
    return analysis
