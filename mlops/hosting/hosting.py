from huggingface_hub import HfApi, create_repo, login
import os
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

# Step 1: Ensure Space exists on Hugging Face
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

# Step 2: Upload folder with automatic retry & backoff for rate limits
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
        print("Successfully uploaded files to Hugging Face Space!")
        break
    except Exception as upload_err:
        print(f"Upload attempt {attempt} failed: {upload_err}")
        if attempt < max_retries:
            sleep_time = attempt * 5
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            raise upload_err
