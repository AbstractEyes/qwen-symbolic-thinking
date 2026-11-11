"""Build the training set from accepted rows inside a queue run: rows.build_api_rows, shipped to the project repo as
sft/<tag>.json, so a later run in the same queue can train on `hub:sft/<tag>.json`.

    {"name": "B_rows", "kind": "tool", "tool": {"module": "build_rows", "args": {"accepted": "hub:owner/dataset",
        "manifest": "hub:owner/images/manifest.jsonl", "writers": ["model-a", "model-b"]}}}

The queue passes the run's name as the tag; the result (the counts, the filters, the file's place) goes into the run's marker."""
from __future__ import annotations

import os
import tempfile

from .. import REPO
from ..hub import ship
from ..rows import build_api_rows


def run(accepted: str, tag: str = "rows", repo: str = REPO, do_ship: bool = True, out_dir: str | None = None, **kwargs) -> dict:
    """build_api_rows(accepted, <out_dir>/<tag>.json, **kwargs), shipped under sft/ of the project repo; returns the build's
    counts and filters without the items. JSON lists in kwargs (writers, licenses, the held splits, the forms) are read as the
    tuples the builder takes."""
    for key in ("licenses", "held_manifest_splits", "forms_picture", "forms_spec"):
        if kwargs.get(key) is not None:
            kwargs[key] = tuple(kwargs[key])
    folder = out_dir or tempfile.mkdtemp(prefix="rows_")
    path = os.path.join(folder, f"{tag}.json")
    built = build_api_rows(accepted, path, **kwargs)
    summary = {"tag": tag, "path_in_repo": f"sft/{tag}.json", "items": len(built["items"]), "counts": built["counts"],
               "filters": built["filters"], "built": built["built"]}
    print(f"rows {tag}: {summary['items']} items; dropped {built['counts']['dropped']}", flush=True)
    if do_ship:
        summary["shipped"] = ship(folder, "sft", f"rows {tag}: {summary['items']} items", repo)
    return summary
