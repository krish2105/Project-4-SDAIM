import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

REPO_ID = "krish2105/bank-customer-churn"
REPO_TYPE = "dataset"
DATA_FOLDER = "mlops/data"


def require_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print(
            "HF_TOKEN is missing. Add a GitHub Actions repository secret named "
            "HF_TOKEN containing a Hugging Face token with write access.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return token


def main() -> None:
    token = require_token()
    api = HfApi(token=token)

    try:
        account = api.whoami()
        username = account.get("name") or account.get("fullname") or "unknown"
        print(f"Authenticated to Hugging Face as {username}.")
    except HfHubHTTPError as exc:
        print(
            "HF_TOKEN is invalid, expired, or does not have access to the "
            "Hugging Face account.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"Dataset repository '{REPO_ID}' already exists.")
    except RepositoryNotFoundError:
        print(f"Dataset repository '{REPO_ID}' not found. Creating it...")
        api.create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            exist_ok=True,
        )

    api.upload_folder(
        folder_path=DATA_FOLDER,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"Uploaded '{DATA_FOLDER}' to https://huggingface.co/datasets/{REPO_ID}.")


if __name__ == "__main__":
    main()
