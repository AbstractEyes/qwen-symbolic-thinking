"""General rows held to a picture-license rule: the items of a general-row file (tools.anchors) whose picture passes the rule a
training-set build applies to its own pictures (rows.build_api_rows: own_license_sources, picture_licenses, licenses), shipped
to the project repo as anchors/<tag>/anchors.json with its summary.json. A run whose training set keeps only some pictures of a
source then does not see the others through its general rows either. Text rows, and pictures of sources not named, pass as
they are; the base model's answers are kept unchanged, so the filtered set is a subset of the original, item for item.

    {"name": "g_by", "kind": "tool", "tool": {"module": "filter_anchors", "args": {"anchors": "hub:anchors/g/anchors.json",
        "own_license_sources": ["coco"], "picture_licenses": "hub:licenses/photos.json", "licenses": ["Attribution License"]}}}

With no picture_licenses, every picture of the named sources leaves (its own license not given)."""
from __future__ import annotations

import os
import tempfile
from collections import Counter
from datetime import datetime, timezone

from .. import REPO, __version__, data
from ..hub import ship
from ..rows import load_picture_licenses
from ..screen import load_inputs


def keep_item(item: dict, own_sources: set, own_licenses: dict, allowed: set | None) -> tuple:
    """(keep or not, the reason or the license read): an item whose seed source is named keeps only when its picture's own
    license is given and allowed; every other item keeps."""
    seed = str(item.get("seed_id") or "")
    source = seed.split(":", 1)[0]
    if not seed or source not in own_sources:
        return True, None
    license_ = own_licenses.get(seed)
    if not license_:
        return False, "picture's own license not given"
    if allowed is not None and license_ not in allowed:
        return False, "license"
    return True, license_


def run(anchors: str, tag: str = "anchors_filtered", own_license_sources=(), picture_licenses=None, licenses=None,
        repo: str = REPO, do_ship: bool = True, out_dir: str | None = None) -> dict:
    """Filter the general rows of `anchors` (a hub path or a local file) and ship them as anchors/<tag>/anchors.json; returns
    the summary (kept and dropped by source and reason, the licenses read)."""
    loaded = load_inputs(anchors)
    items = list(loaded["items"])
    own_sources = set(own_license_sources or ())
    own_licenses = load_picture_licenses(picture_licenses)
    allowed = set(licenses) if licenses is not None else None
    kept, dropped, kept_licenses = [], Counter(), Counter()
    for it in items:
        ok, why = keep_item(it, own_sources, own_licenses, allowed)
        if ok:
            kept.append(it)
            if why:
                kept_licenses[why] += 1
        else:
            dropped[f"{it.get('source')}: {why}"] += 1
    folder = out_dir or tempfile.mkdtemp(prefix="anchors_")
    out = {k: v for k, v in loaded.items() if k != "items"}
    out.update(tag=tag, filtered_from=anchors, items=kept,
               filtered={"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "package": __version__,
                         "own_license_sources": sorted(own_sources), "licenses": sorted(allowed) if allowed is not None else None,
                         "picture_licenses": picture_licenses if isinstance(picture_licenses, str) else
                         ({"entries": len(own_licenses)} if picture_licenses else None)})
    data._write(out, os.path.join(folder, "anchors.json"))
    summary = {"tag": tag, "path_in_repo": f"anchors/{tag}/anchors.json", "items": len(items), "kept": len(kept),
               "kept_by_category": dict(Counter(it.get("category") for it in kept)),
               "kept_by_source": dict(Counter(str(it.get("source")) for it in kept)),
               "kept_licenses": dict(kept_licenses), "dropped": dict(dropped), "filtered": out["filtered"]}
    data._write(summary, os.path.join(folder, "summary.json"))
    print(f"anchors {tag}: kept {len(kept)} of {len(items)}; dropped {dict(dropped)}", flush=True)
    if do_ship:
        summary["shipped"] = ship(folder, f"anchors/{tag}", f"anchors {tag}: {len(kept)} of {len(items)} general rows", repo)
    return summary
