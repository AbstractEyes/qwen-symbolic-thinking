"""Build the training set from accepted rows inside a queue run: rows.build_api_rows, shipped to the project repo as
sft/<tag>.json, so a later run in the same queue can train on `hub:sft/<tag>.json`.

    {"name": "B_rows", "kind": "tool", "tool": {"module": "build_rows", "args": {"accepted": "hub:owner/dataset",
        "manifest": "hub:owner/images/manifest.jsonl", "writers": ["model-a", "model-b"]}}}

The queue passes the run's name as the tag; the result (the counts, the filters, the file's place) goes into the run's marker.
With "hold_out" in the args, the drawn pictures ship beside the training set as sft/<tag>_held.json (see rows.build_api_rows),
the held pictures tools.anchors can take for the held-out read. With "pack" ({"bucket": "owner/name/folder", "dirs": {source:
[dirs]}}), every picture of those sources is first copied from the bucket into the project repo (pack.pack_staged, under
pictures/<tag>/), and its manifest joins the build's last, so the copies the rows were written from win over an older manifest's
picture of the same number (two copies of one dataset can number their pictures differently); the rows are downloaded once, so the
pack and the build read the same snapshot."""
from __future__ import annotations

import os
import tempfile

from .. import REPO, data
from ..hub import ship
from ..pack import LISTS, pack_staged
from ..rows import build_api_rows, local_accepted, seeds_without_locator


def run(accepted: str, tag: str = "rows", repo: str = REPO, do_ship: bool = True, out_dir: str | None = None,
        pack: dict | None = None, **kwargs) -> dict:
    """build_api_rows(accepted, <out_dir>/<tag>.json, **kwargs), shipped under sft/ of the project repo; returns the build's
    counts and filters without the items. JSON lists in kwargs (writers, licenses, the held splits, the forms) are read as the
    tuples the builder takes."""
    for key in ("licenses", "held_manifest_splits", "forms_picture", "forms_spec"):
        if kwargs.get(key) is not None:
            kwargs[key] = tuple(kwargs[key])
    folder = out_dir or tempfile.mkdtemp(prefix="rows_")
    path = os.path.join(folder, f"{tag}.json")
    if kwargs.get("hold_out"):
        kwargs.setdefault("held_path", os.path.join(folder, f"{tag}_held.json"))
    packed, packed_hub = None, None
    if pack:
        accepted = local_accepted(accepted)
        sources = list(pack.get("dirs") or {}) or None
        # Every picture of the named sources comes from the staged copies the rows were written from, even where an older
        # manifest holds a picture under the same number: two copies of one dataset can number their pictures differently.
        # The packed manifest goes last, so its locators win. With no sources named, only what no manifest locates is packed.
        need = seeds_without_locator(accepted, None if sources else kwargs.get("manifest"), sources)
        packed = pack_staged(need, pack["bucket"], dirs=pack.get("dirs"), out_dir=tempfile.mkdtemp(prefix="pack_"), tag=tag,
                             repo=repo, lists=pack.get("lists", LISTS), do_ship=do_ship)
        base = kwargs.get("manifest")
        kwargs["manifest"] = ([base] if isinstance(base, str) else list(base or [])) + [packed["manifest"]]
        packed_hub = f"{repo}:{packed['path_in_repo']}/manifest.jsonl"
    built = build_api_rows(accepted, path, **kwargs)
    if packed:            # the record names the packed manifest by its place in the project repo, not the runtime's folder
        built["filters"]["manifest"] = [m if m != packed["manifest"] else packed_hub for m in built["filters"]["manifest"]]
        built["filters"]["pack"] = {k: v for k, v in packed.items() if k != "manifest"}
        data._write(built, path)
    summary = {"tag": tag, "path_in_repo": f"sft/{tag}.json", "items": len(built["items"]), "counts": built["counts"],
               "filters": built["filters"], "built": built["built"],
               "held_path_in_repo": f"sft/{os.path.basename(kwargs['held_path'])}" if kwargs.get("hold_out") else None,
               "pack": {k: v for k, v in packed.items() if k != "manifest"} if packed else None}
    print(f"rows {tag}: {summary['items']} items; dropped {built['counts']['dropped']}", flush=True)
    if do_ship:
        summary["shipped"] = ship(folder, "sft", f"rows {tag}: {summary['items']} items", repo)
    return summary
