"""A lane works through a queue of runs kept on the Hub, so a card runs unattended and new runs can be added while it works.

    from qwen_symbolic_thinking.queue import run_queue
    run_queue("lane1")

The queue is runs/queue_<lane>.json in the project repo: a list of specs, each {"name", "kind": "train" | "synth", ...}. A
train spec trains (its "train" keys go to train.run) and then screens the adapter (its "screen" keys go to screen.run, with the
trained model's forms by default). A synth spec generates synthetic items (synth.generate). Each run leaves runs/done/<name>.json
on the Hub (started, then done or failed, with the summaries); a spec whose marker exists is skipped unless it says "retry".
When the queue is empty the lane waits for new specs, polling every minute, and leaves after idle_minutes without any."""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone

from . import REPO
from .hub import token

DEFAULT_SCREEN = {"inputs": "hub:screens/inputs/screen_v1.json", "forms": ("full", "short", "spec_full"), "batch": 4,
                  "max_new_tokens": 700}


def _fresh(path_in_repo: str, repo: str):
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(repo, path_in_repo, repo_type="model", token=token(), force_download=True)
    except Exception:
        return None
    with open(local, encoding="utf-8") as fh:
        return json.load(fh)


def _markers(repo: str) -> dict:
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    out = {}
    for f in api.list_repo_files(repo):
        if f.startswith("runs/done/") and f.endswith(".json"):
            out[f[len("runs/done/"):-5]] = f
    return out


def _mark(name: str, payload: dict, repo: str, out_dir: str) -> None:
    from huggingface_hub import HfApi
    os.makedirs(out_dir, exist_ok=True)
    local = os.path.join(out_dir, f"{name}.json")
    with open(local, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    for attempt in range(3):
        try:
            HfApi(token=token()).upload_file(path_or_fileobj=local, path_in_repo=f"runs/done/{name}.json", repo_id=repo, repo_type="model",
                                             commit_message=f"run {name}: {payload.get('status')}")
            return
        except Exception as e:
            print(f"  marker upload failed ({type(e).__name__}); retry", flush=True)
            time.sleep(10 * (attempt + 1))


def run_spec(spec: dict, lane: str, repo: str, out_dir: str) -> dict:
    name = spec["name"]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _mark(name, {"status": "started", "lane": lane, "started": started, "spec": spec}, repo, out_dir)
    result = {"status": "done", "lane": lane, "started": started, "spec": spec}
    if spec["kind"] == "train":
        from . import screen, train
        t = train.run(**spec["train"], tag=name, repo=repo)
        result["train"] = {k: v for k, v in t.items() if k not in ("log", "env")}
        s = dict(DEFAULT_SCREEN, **spec.get("screen", {}))
        s["forms"] = tuple(s["forms"])
        result["screen"] = screen.run(models=[spec["train"].get("model_id", "Qwen/Qwen3.5-9B")], adapter=t["adapter_dir"], tag=name, repo=repo, **s)
    elif spec["kind"] == "synth":
        from . import synth
        s = synth.generate(**spec["synth"], tag=name, repo=repo)
        result["synth"] = {k: v for k, v in s.items() if k != "env"}
    else:
        raise ValueError(f"unknown kind {spec['kind']!r}")
    result["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


def run_queue(lane: str, repo: str = REPO, out_dir: str = "/content/runs", idle_minutes: int = 45, poll_s: int = 60) -> list:
    """Work through runs/queue_<lane>.json until it is empty for idle_minutes; returns the names run."""
    ran, idle_since = [], None
    while True:
        specs = _fresh(f"runs/queue_{lane}.json", repo) or []
        markers = _markers(repo)
        todo = [s for s in specs if s["name"] not in markers or s.get("retry")]
        if not todo:
            if idle_since is None:
                idle_since = time.time()
                print(f"[{lane}] queue empty; waiting up to {idle_minutes} min for new runs", flush=True)
            if time.time() - idle_since > idle_minutes * 60:
                print(f"[{lane}] nothing new for {idle_minutes} min; leaving", flush=True)
                return ran
            time.sleep(poll_s)
            continue
        idle_since = None
        spec = todo[0]
        print(f"\n[{lane}] == {spec['name']} ({spec['kind']}) at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC", flush=True)
        try:
            result = run_spec(spec, lane, repo, out_dir)
        except Exception as e:
            result = {"status": "failed", "lane": lane, "spec": spec, "error": f"{type(e).__name__}: {str(e)[:500]}",
                      "traceback": traceback.format_exc()[-3000:]}
            print(f"[{lane}] FAILED {spec['name']}: {result['error']}", flush=True)
        _mark(spec["name"], result, repo, out_dir)
        ran.append(spec["name"])
