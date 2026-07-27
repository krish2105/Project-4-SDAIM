# for data manipulation
import pandas as pd
import sklearn
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

# Prefer local dataset file if present to avoid rate limits (429)
local_csv = "mlops/data/bank_customer_churn.csv"
if os.path.exists(local_csv):
    print(f"Loading dataset from local file: '{local_csv}'")
    bank_dataset = pd.read_csv(local_csv)
else:
    print("Local dataset not found. Downloading via authenticated Hugging Face API...")
    try:
        csv_file = hf_hub_download(
            repo_id="krish21may/Bank-Customer-Churn-4",
            filename="bank_customer_churn.csv",
            repo_type="dataset",
            token=token
        )
        bank_dataset = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Authenticated download failed: {e}. Attempting fallback...")
        bank_dataset = pd.read_csv("hf://datasets/jpaggarwal/bank-customer-churn/bank_customer_churn.csv")

print("Dataset loaded successfully. Shape:", bank_dataset.shape)

# Define target variable for classification task
target = 'Exited'

# List of numerical features
numeric_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# List of categorical features
categorical_features = ['Geography']

X = bank_dataset[numeric_features + categorical_features]
y = bank_dataset[target]

# Split dataset into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

repo_id = "krish21may/Bank-Customer-Churn-4"
for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded '{file_path}' to dataset repo '{repo_id}'.")
    except Exception as upload_err:
        print(f"Failed to upload '{file_path}': {upload_err}")
