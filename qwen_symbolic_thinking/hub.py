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


def fetch_folder(path_in_repo: str, repo: str = REPO) -> str:
    """A folder of the project repo (a full fine-tune under full/<model>/<stamp>_<tag>/final), downloaded (cached) and
    returned as a local path."""
    from huggingface_hub import snapshot_download
    path = path_in_repo.strip("/")
    root = snapshot_download(repo, repo_type="model", allow_patterns=[f"{path}/*"], token=token())
    return os.path.join(root, *path.split("/"))


def resolve_model(model_id: str, repo: str = REPO) -> str:
    """A model id as the loaders take it: a Hub repo id or a local folder as given; 'hub:<path>' (a full fine-tune shipped
    under the project repo) downloaded to its folder."""
    return fetch_folder(model_id[4:], repo) if model_id.startswith("hub:") else model_id


def model_short(model_id: str) -> str:
    """The model's short name for paths: Qwen/Qwen3.5-4B -> Qwen3.5-4B; hub:full/Qwen3.5-4B/<stamp>/final -> Qwen3.5-4B."""
    parts = model_id[4:].strip("/").split("/") if model_id.startswith("hub:") else model_id.rstrip("/").split("/")
    return parts[1] if model_id.startswith("hub:") and len(parts) > 1 else parts[-1]


def status_of(e: BaseException) -> int | None:
    """The HTTP status of a Hub fault, when it carries a response."""
    return getattr(getattr(e, "response", None), "status_code", None)


def transient(e: BaseException) -> bool:
    """A fault worth waiting out: the Hub's rate limit (429), a server fault (5xx), or a dropped or timed-out connection
    (an exception of the HTTP client with no response). A missing file, a refused token or a bad request is not."""
    code = status_of(e)
    if code is not None:
        return code == 429 or code >= 500
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    return type(e).__module__.split(".")[0] in ("httpx", "httpcore", "requests", "urllib3")


def retry_after(e: BaseException) -> float | None:
    """The seconds the server asks a client to wait (its Retry-After header), when it says."""
    try:
        value = e.response.headers.get("Retry-After")     # type: ignore[attr-defined]
        return float(value) if value is not None else None
    except Exception:
        return None


def wait_for(attempt: int, e: BaseException | None = None, first: float = 15.0, cap: float = 120.0) -> float:
    """The wait before try attempt + 2: first * 2**attempt, at most cap, and never less than the server's Retry-After."""
    return max(min(cap, first * 2 ** attempt), (retry_after(e) or 0.0) if e is not None else 0.0)


def with_retries(call, what: str, tries: int = 8, first: float = 15.0, cap: float = 120.0, sleep=None):
    """call() with the Hub's passing faults waited out: on a transient fault it waits (wait_for) and tries again, up to tries
    times; with the defaults the waits add up to about ten minutes, past the Hub's 300 s rate-limit window. Any other fault,
    and the last transient one, is raised. sleep: the wait (time.sleep, looked up at the call)."""
    for attempt in range(tries):
        try:
            return call()
        except Exception as e:
            if not transient(e) or attempt == tries - 1:
                raise
            wait = wait_for(attempt, e, first, cap)
            print(f"  {what}: {type(e).__name__} (status {status_of(e)}); try {attempt + 2} of {tries} in {wait:.0f} s", flush=True)
            (sleep or time.sleep)(wait)


def ship(folder: str, path_in_repo: str, message: str, repo: str = REPO, retries: int = 8,
         repo_type: str = "model", sleep=None) -> str | None:
    """Upload a folder under a path of the project repo (or another repo of repo_type), retrying every fault with growing
    waits (wait_for: about ten minutes over the default 8 tries, past the Hub's 300 s rate-limit window); returns the commit
    url, or None once it gives up (the results stay on disk)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    for attempt in range(retries):
        try:
            info = api.upload_folder(folder_path=folder, path_in_repo=path_in_repo, repo_id=repo, repo_type=repo_type,
                                     commit_message=message)
            return getattr(info, "commit_url", None) or str(info)
        except Exception as e:  # the Hub's faults are retried; the local results stay on disk either way
            if attempt == retries - 1:
                break
            wait = wait_for(attempt, e)
            print(f"  ship failed ({type(e).__name__}: {str(e)[:120]}); retry in {wait:.0f} s", flush=True)
            (sleep or time.sleep)(wait)
    print("  ship gave up; the results stay in", folder, flush=True)
    return None
