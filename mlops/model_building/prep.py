import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi, hf_hub_download

token = os.getenv("HF_TOKEN")
if token:
    try:
        login(token=token)
    except Exception as login_err:
        print(f"Login warning: {login_err}")

api = HfApi(token=token)

local_csv = "mlops/data/ecommerce_customer_churn.csv"
fallback_csv = "mlops/data/bank_customer_churn.csv"

if os.path.exists(local_csv):
    print(f"Loading dataset from local file: '{local_csv}'")
    df = pd.read_csv(local_csv)
elif os.path.exists(fallback_csv):
    print(f"Loading dataset from fallback file: '{fallback_csv}'")
    df = pd.read_csv(fallback_csv)
else:
    raise FileNotFoundError("Dataset CSV file not found.")

print("Dataset loaded successfully. Shape:", df.shape)

target = 'Churn'

numeric_features = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'SatisfactionScore', 'Complain', 'OrderAmountHikeFromlastYear',
    'DaySinceLastOrder', 'CashBackAmount', 'CityTier'
]

categorical_features = ['PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']

# Standardize columns
X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# Split dataset into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete. Preprocessed files saved successfully.")
