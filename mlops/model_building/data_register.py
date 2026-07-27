from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "krish21may/Bank-Customer-Churn-4"
repo_type = "dataset"
token = os.getenv("HF_TOKEN")

# Initialize API client
api = HfApi(token=token)

# Step 1: Check if the dataset repo exists
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

api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
