"""A training's progress file: a small JSON the training writes to its project repo, runs/progress/<run>.json, so a heartbeat
reads where every run stands without opening the machine's page; and its reader.

    python -m qwen_symbolic_thinking.tools.progress <project repo> [<run> ...]      (every run's line; HF_TOKEN for a private repo)

The file (written at the training's first log step, then every `every_min` minutes at a log step, and once when the training
ends):

| key | what it holds |
|---|---|
| run, state | the run's name (its queue spec's name), "training" or "trained" |
| step, of | the optimizer steps done and the run's total |
| loss, general_kl | the last log line's training loss and keeping term (the mean since the line before) |
| s_per_step, steady_s_per_step | this machine's seconds a step since its training began; the seconds a step since the previous write (None at a machine's first write) |
| left_s, eta | the seconds left at this machine's pace, and the time that makes (UTC) |
| peak_gb | the most GPU memory the run's tensors held so far |
| saved_step, shipped_step | the step of the last saved training state (a resume continues from it; None without resume) and of the last adapter shipped during training |
| time, package | when the file was written (UTC) and the package version |

Writing is best effort and never holds the training: the file is uploaded on a thread of its own, one try; a write that fails
is printed and skipped (the next comes `every_min` minutes later), and while one upload is still running the next write is
skipped. The training's own log, saves and ships do not depend on it. train.run(progress_every_min=N) turns it on (a queue train
spec: "progress_every_min": N among its train keys); without it nothing is written."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

from .. import REPO, __version__

FOLDER = "runs/progress"


def path_of(run: str) -> str:
    return f"{FOLDER}/{run}.json"


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def upload_json(record: dict, repo: str, path_in_repo: str) -> None:
    """The default writer: one upload of the record to the project repo (no retries: the next write comes soon)."""
    from huggingface_hub import HfApi

    from ..hub import token
    data = json.dumps(record, indent=1, default=str).encode("utf-8")
    HfApi(token=token()).upload_file(path_or_fileobj=data, path_in_repo=path_in_repo, repo_id=repo, repo_type="model",
                                     commit_message=f"{record.get('run')}: progress at step {record.get('step')} of {record.get('of')}")


class ProgressWriter:
    """Writes a run's progress file from its step log (train.progress's records). maybe(log) at every log step writes when one is
    due (the first call, then every every_min minutes); finish(log) writes the last record and waits up to wait_s for it.
    upload(record, repo, path_in_repo): the writer (upload_json by default; tests pass their own); clock: time.time; peak(),
    saved(), shipped(): the run's peak GPU memory in GB, its last saved and last shipped steps (each may return None)."""

    def __init__(self, run: str, total: int, repo: str = REPO, every_min: float = 10.0, upload=None, clock=time.time,
                 local_dir: str | None = None, peak=None, saved=None, shipped=None):
        self.run, self.total, self.repo, self.every_s = run, int(total), repo, float(every_min) * 60.0
        self.upload, self.clock, self.local_dir = upload or upload_json, clock, local_dir
        self.peak, self.saved, self.shipped = peak, saved, shipped
        self.last_t, self.last_rec, self.thread = None, None, None
        self.writes, self.failures, self.skipped = 0, 0, 0

    def _value(self, fn):
        try:
            return fn() if fn is not None else None
        except Exception:       # a reading that fails leaves its field empty, never the training
            return None

    def record(self, log: list, state: str = "training") -> dict:
        rec = dict(log[-1]) if log else {}
        now = self.clock()
        steady = None
        prev = self.last_rec
        if prev is not None and rec.get("step") is not None and rec["step"] > prev.get("step", 0) and \
                rec.get("spent_s") is not None and prev.get("spent_s") is not None:
            steady = round((float(rec["spent_s"]) - float(prev["spent_s"])) / (rec["step"] - prev["step"]), 3)
        left = rec.get("left_s")
        return {"run": self.run, "state": state, "step": rec.get("step"), "of": self.total, "loss": rec.get("loss"),
                "general_kl": rec.get("general_kl"), "s_per_step": rec.get("s_per_step"), "steady_s_per_step": steady,
                "left_s": left, "eta": _utc(now + left) if isinstance(left, (int, float)) else None,
                "peak_gb": self._value(self.peak), "saved_step": self._value(self.saved),
                "shipped_step": self._value(self.shipped), "time": _utc(now), "package": __version__,
                "_spent_s": rec.get("spent_s")}

    def _send(self, rec: dict) -> None:
        try:
            self.upload({k: v for k, v in rec.items() if not k.startswith("_")}, self.repo, path_of(self.run))
            self.writes += 1
        except Exception as e:   # best effort: printed and skipped; the next write comes every_min minutes later
            self.failures += 1
            print(f"  progress file of {self.run} not written ({type(e).__name__}: {str(e)[:160]}); skipped", flush=True)

    def write(self, log: list, state: str = "training", wait_s: float = 0.0) -> bool:
        """Write now (on its own thread). False when the previous upload is still running (this write is skipped)."""
        if self.thread is not None and self.thread.is_alive():
            self.skipped += 1
            return False
        rec = self.record(log, state)
        self.last_t, self.last_rec = self.clock(), {"step": rec["step"], "spent_s": rec["_spent_s"]}
        if self.local_dir:
            try:
                os.makedirs(self.local_dir, exist_ok=True)
                with open(os.path.join(self.local_dir, "progress.json"), "w", encoding="utf-8") as fh:
                    json.dump({k: v for k, v in rec.items() if not k.startswith("_")}, fh, indent=1, default=str)
            except OSError:
                pass
        self.thread = threading.Thread(target=self._send, args=(rec,), name=f"progress-{self.run}", daemon=True)
        self.thread.start()
        if wait_s:
            self.thread.join(timeout=wait_s)
        return True

    def due(self) -> bool:
        return self.last_t is None or self.clock() - self.last_t >= self.every_s

    def maybe(self, log: list) -> bool:
        """At a log step: write when one is due. Returns whether a write started."""
        return self.write(log) if (log and self.due()) else False

    def finish(self, log: list, wait_s: float = 5.0) -> bool:
        """The last record (state "trained"), waited for up to wait_s seconds (a slower upload goes on in the background)."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=wait_s)
        return self.write(log, "trained", wait_s=wait_s)


def progress_callback(writer: ProgressWriter, log: list):
    """The TrainerCallback that drives a ProgressWriter from the training's step log (it runs after train.progress's callback,
    which appends each log step's record to log)."""
    from transformers import TrainerCallback

    class ProgressFile(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None and "loss" not in logs:     # the end-of-training summary line, not a step
                return
            writer.maybe(log)

        def on_train_end(self, args, state, control, **kw):
            writer.finish(log)

    return ProgressFile()


# ------------------------------------------------------------------------------------------------------------- the reader
def _list_files(repo: str) -> list:
    from huggingface_hub import HfApi

    from ..hub import token, with_retries
    return with_retries(lambda: HfApi(token=token()).list_repo_files(repo), "list the repo's files")


def _fetch(repo: str, path_in_repo: str):
    from huggingface_hub import hf_hub_download

    from ..hub import token
    try:
        local = hf_hub_download(repo, path_in_repo, repo_type="model", token=token(), force_download=True)
    except Exception:
        return None
    with open(local, encoding="utf-8") as fh:
        return json.load(fh)


def read(repo: str, runs=None, list_files=None, fetch=None) -> list:
    """The progress records of the named runs in a project repo, or of every run with a progress file. A named run without
    one reads {"run": <name>, "state": None}. list_files(repo) and fetch(repo, path) default to the Hub's."""
    if not runs:
        files = (list_files or _list_files)(repo)
        runs = sorted(f[len(FOLDER) + 1:-5] for f in files if f.startswith(FOLDER + "/") and f.endswith(".json"))
    out = []
    for run in runs:
        rec = (fetch or _fetch)(repo, path_of(run))
        out.append(rec if isinstance(rec, dict) else {"run": run, "state": None})
    return out


def line(rec: dict, now: float | None = None) -> str:
    """One run's progress as one line of text."""
    if not rec.get("state"):
        return f"{rec.get('run')}: no progress file"
    step, of = rec.get("step"), rec.get("of")
    parts = [f"{rec['run']}: {rec['state']}, step {step if step is not None else '-'} of {of}"
             + (f" ({100.0 * step / of:.1f}%)" if step is not None and of else "")]
    if rec.get("loss") is not None:
        parts.append(f"loss {rec['loss']:.4f}")
    if rec.get("general_kl") is not None:
        parts.append(f"general_kl {rec['general_kl']:.5f}")
    if rec.get("s_per_step") is not None:
        parts.append(f"{rec['s_per_step']} s a step" + (f" (steady {rec['steady_s_per_step']})"
                                                       if rec.get("steady_s_per_step") is not None else ""))
    if rec.get("peak_gb") is not None:
        parts.append(f"peak {rec['peak_gb']} GB")
    parts.append(f"saved at step {rec['saved_step'] if rec.get('saved_step') is not None else '-'}, shipped step "
                 f"{rec['shipped_step'] if rec.get('shipped_step') is not None else '-'}")
    if rec.get("left_s") is not None and rec["state"] == "training":
        parts.append(f"about {rec['left_s'] / 3600.0:.1f} h left" + (f" (ends about {rec['eta']})" if rec.get("eta") else ""))
    if rec.get("time"):
        written = rec["time"]
        try:
            age = (now if now is not None else time.time()) - datetime.fromisoformat(written).timestamp()
            written += f" ({int(age // 60)} min ago)"
        except ValueError:
            pass
        parts.append(f"written {written}")
    return "; ".join(parts)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Print each run's progress line from its project repo's runs/progress/ files.")
    ap.add_argument("repo", help="the project repo (owner/name)")
    ap.add_argument("runs", nargs="*", help="run names (default: every run with a progress file)")
    args = ap.parse_args(argv)
    recs = read(args.repo, args.runs)
    if not recs:
        print(f"no progress files in {args.repo}/{FOLDER}/")
    for rec in recs:
        print(line(rec))


if __name__ == "__main__":
    main()
