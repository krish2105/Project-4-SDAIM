from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
import pandas as pd

repo_id = "krish21may/Bank-Customer-Churn-4"
repo_type = "dataset"
token = os.getenv("HF_TOKEN")

# Initialize API client
api = HfApi(token=token)

# Step 1: Ensure local data directory exists and contains data
os.makedirs("mlops/data", exist_ok=True)
csv_file_path = os.path.join("mlops/data", "bank_customer_churn.csv")

if not os.path.exists(csv_file_path):
    print(f"File '{csv_file_path}' not found locally. Fetching base dataset...")
    df = pd.read_csv("hf://datasets/jpaggarwal/bank-customer-churn/bank_customer_churn.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"Saved dataset to '{csv_file_path}'.")

# Step 2: Check if the dataset repo exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except Exception as e:
    print(f"Dataset '{repo_id}' not found or error: {e}. Creating new dataset repo...")
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
        print(f"Dataset '{repo_id}' created.")
    except Exception as create_err:
        print(f"Repo creation info: {create_err}")

# Step 3: Upload folder to Hugging Face
api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Successfully uploaded 'mlops/data' to '{repo_id}'.")
