import os
from huggingface_hub import HfApi

HF_USERNAME = os.environ.get("HF_USERNAME", "AaaaaAa5657")
SPACE_NAME = "streamlit-tourism-app"
SPACE_REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
HF_TOKEN = os.environ.get("HF_TOKEN")

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN environment variable is not set.")

api = HfApi(token=HF_TOKEN)

for fname in ["app.py", "requirements.txt", "best_model.pkl.gz"]:
    fpath = os.path.join(DEPLOY_DIR, fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
        )
        print(f"Uploaded {fname} -> {SPACE_REPO_ID}")

print(f"\nSpace URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")
