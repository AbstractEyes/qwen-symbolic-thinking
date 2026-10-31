"""Where the readers keep files and how they fetch them from the Hub when missing."""
import json
import os

from huggingface_hub import hf_hub_download

TRAIN_REPO = "AbstractPhil/qwen-symbolic-thinking-image-training"
DATA_REPO = "AbstractPhil/qwen-thinking-images"
MANIFEST = "manifest_v1.5.jsonl"
MUSEUMS = ("aic", "nga", "cleveland", "met")


def root(arg: str | None = None) -> str:
    """The data root: the argument, else QST_DATA, else ./data."""
    return os.path.abspath(arg or os.environ.get("QST_DATA") or "data")


def add_common(parser, manifest: bool = True):
    parser.add_argument("--data", default=None, help="the data root (default: $QST_DATA, else ./data)")
    parser.add_argument("--offline", action="store_true", help="read the files already under the root; fetch nothing")
    if manifest:
        parser.add_argument("--manifest", default=None, help=f"a manifest path (default: <root>/{MANIFEST}, fetched when missing)")


def marker(run: str, root_dir: str, offline: bool = False) -> dict:
    """A run's marker (`runs/done/<run>.json`), fetched fresh unless offline: it changes while the run goes."""
    p = os.path.join(root_dir, "runs", "done", f"{run}.json")
    if not offline:
        hf_hub_download(TRAIN_REPO, f"runs/done/{run}.json", local_dir=root_dir, force_download=True)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def rows_of(run: str, root_dir: str, offline: bool = False):
    """An extraction run's marker and its rows (None until the run is done)."""
    m = marker(run, root_dir, offline)
    path = (m.get("extract") or {}).get("path_in_repo")
    if m.get("status") != "done" or not path:
        return m, None
    p = os.path.join(root_dir, *path.split("/"), "rows.jsonl")
    if not os.path.exists(p):
        if offline:
            raise FileNotFoundError(p)
        hf_hub_download(DATA_REPO, f"{path}/rows.jsonl", repo_type="dataset", local_dir=root_dir)
    with open(p, encoding="utf-8") as fh:
        return m, [json.loads(line) for line in fh if line.strip()]


def manifest_rows(root_dir: str, path: str | None = None, offline: bool = False):
    """The manifest's rows, one dict at a time."""
    p = path or os.path.join(root_dir, MANIFEST)
    if not os.path.exists(p):
        if offline:
            raise FileNotFoundError(p)
        hf_hub_download(DATA_REPO, os.path.basename(p), repo_type="dataset", local_dir=root_dir)
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def caption_text(row: dict) -> str:
    """A manifest row's captions as one string (a dict of fields, a JSON-ish string, or plain text)."""
    caps = row.get("captions") or {}
    if isinstance(caps, str):
        try:
            caps = json.loads(caps.replace("'", '"'))
        except ValueError:
            caps = {"text": caps}
    return " ".join(str(v) for v in caps.values()) if isinstance(caps, dict) else str(caps)


def museum_captions(root_dir: str, path: str | None = None, offline: bool = False, sources=MUSEUMS) -> dict:
    """sample_id -> the catalogue line, for the museum sources."""
    return {m["sample_id"]: caption_text(m) for m in manifest_rows(root_dir, path, offline) if m.get("source") in sources}


def screen_dir(stamp: str, root_dir: str, fetch: bool = False) -> str:
    """A screen's folder (`screens/<stamp>/`), its results and summary fetched when missing or asked for."""
    d = os.path.join(root_dir, "screens", stamp)
    if fetch or not os.path.exists(os.path.join(d, "results.jsonl")):
        for f in ("results.jsonl", "summary.json"):
            hf_hub_download(TRAIN_REPO, f"screens/{stamp}/{f}", local_dir=root_dir, force_download=fetch)
    return d
