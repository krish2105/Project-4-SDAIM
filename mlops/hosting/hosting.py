from huggingface_hub import HfApi, create_repo, login
import os
import shutil
import time

token = os.getenv("HF_TOKEN")
if token:
    try:
        login(token=token)
    except Exception as login_err:
        print(f"Login notice: {login_err}")

api = HfApi(token=token)
repo_id = "krish21may/Bank-Customer-Churn-4"
repo_type = "space"

# Step 1: Sync required modules & files into deployment directory for Hugging Face Space
deployment_dir = "mlops/deployment"
deployment_mlops = os.path.join(deployment_dir, "mlops")
os.makedirs(deployment_mlops, exist_ok=True)

with open(os.path.join(deployment_mlops, "__init__.py"), "w") as f:
    f.write("# Package init\n")

for pkg in ["analytics", "monitoring", "reports", "api", "data"]:
    src = os.path.join("mlops", pkg)
    dst = os.path.join(deployment_mlops, pkg)
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied '{src}' -> '{dst}' for Space packaging.")

# Copy Xtest.csv, best_churn_model.joblib, and requirements.txt directly into deployment root
for fname in ["Xtest.csv", "best_churn_model.joblib", "Xtrain.csv", "ytrain.csv", "ytest.csv"]:
    if os.path.exists(fname):
        shutil.copy(fname, os.path.join(deployment_dir, fname))
        print(f"Copied '{fname}' -> '{deployment_dir}/{fname}'")

# Step 2: Ensure Space exists on Hugging Face
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except Exception as e:
    print(f"Space '{repo_id}' not found or error: {e}. Creating new Streamlit Space...")
    try:
        create_repo(
            repo_id=repo_id, 
            repo_type=repo_type, 
            space_sdk="streamlit", 
            private=False, 
            token=token
        )
        print(f"Space '{repo_id}' created successfully.")
    except Exception as create_err:
        print(f"Space creation info: {create_err}")

# Step 3: Upload folder with automatic retry & backoff for rate limits
max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        print(f"Uploading 'mlops/deployment' to Space '{repo_id}' (Attempt {attempt}/{max_retries})...")
        api.upload_folder(
            folder_path="mlops/deployment",
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo="",
        )
        print("Successfully deployed app to Hugging Face Space!")
        break
    except Exception as err:
        print(f"Upload error (attempt {attempt}/{max_retries}): {err}")
        if attempt < max_retries:
            time.sleep(5)
