"""The Hub side: the token (the environment, or a Colab secret), a file of the project repo, and a folder shipped to it."""
from __future__ import annotations

import os
import time

from . import REPO


def token() -> str | None:
    """HF_TOKEN from the environment, or from the Colab secrets when the notebook has access to them."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from google.colab import userdata  # type: ignore
        tok = userdata.get("HF_TOKEN")
    except Exception:
        tok = None
    if tok:
        os.environ["HF_TOKEN"] = tok
    return tok


def fetch(path_in_repo: str, repo: str = REPO) -> str:
    """A file of the project repo, downloaded (cached) and returned as a local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path_in_repo, repo_type="model", token=token())


def ship(folder: str, path_in_repo: str, message: str, repo: str = REPO, retries: int = 4) -> str | None:
    """Upload a folder under a path of the project repo, with backoff retries; returns the commit url or None."""
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    for attempt in range(retries):
        try:
            info = api.upload_folder(folder_path=folder, path_in_repo=path_in_repo, repo_id=repo, repo_type="model",
                                     commit_message=message)
            return getattr(info, "commit_url", None) or str(info)
        except Exception as e:  # the Hub's faults are retried; the local results stay on disk either way
            wait = 15 * (2 ** attempt)
            print(f"  ship failed ({type(e).__name__}: {str(e)[:120]}); retry in {wait} s", flush=True)
            time.sleep(wait)
    print("  ship gave up; the results stay in", folder, flush=True)
    return None
