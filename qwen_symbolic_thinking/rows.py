"""Training items from rows written by an outside row-writing system.

    from qwen_symbolic_thinking.rows import build_api_rows
    build_api_rows("hub:AbstractPhil/some-dataset", "sft/api_rows.json", manifest="data/manifest.jsonl")

The writer leaves its accepted rows as accepted/<route>/<task>.jsonl, one JSON row a line: the row's thoughts, wordings and
questions, a `source` (a seeded picture or a synthetic specification), the `writer` that wrote it (its model, route, date, ...)
and the `source_label` of the picture (dataset, item, license). This module turns those rows into the items train.run reads
(data.item's keys, plus the writer, the route, the license and the row's own id), after four filters, each counted in the output:
the writers a run accepts, the licenses it accepts, the pictures held out of training (the screen set's, the manifest's held
splits, and the museum pictures a held-out read draws), and the pictures whose locator cannot be found.

A row that names a seed (`source.kind` "seeded" and a `seed_id`) is a picture row; a "synthetic" row is a specification row,
taught from its records with no picture. Several writers may have written about the same picture; every row stays an item of
its own, told apart by `row_id`."""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone

from . import data, extract, screen
from .hub import token

DOCCI_ARCHIVE = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"
DEFAULT_SCREEN = "hub:screens/inputs/screen_v1.json"
DEFAULT_LICENSES = ("CC BY 4.0", "Apache-2.0")


# ------------------------------------------------------------------------------------------------------ reading the rows
def accepted_dir(accepted: str) -> str:
    """The local folder that holds the route folders: the given folder's own accepted/ when it has one, else the folder
    itself; 'hub:<dataset repo id>' downloads accepted/*/*.jsonl first (the dataset repo, a token from the environment)."""
    if accepted.startswith("hub:"):
        from huggingface_hub import snapshot_download
        root = snapshot_download(accepted[4:], repo_type="dataset", allow_patterns=["accepted/*/*.jsonl"], token=token())
    else:
        root = accepted
    inner = os.path.join(root, "accepted")
    return inner if os.path.isdir(inner) else root


def read_accepted(accepted: str, unreadable: Counter | None = None) -> list:
    """[(route, row)] for every line of every accepted/<route>/<task>.jsonl, in file order (routes and tasks sorted). A line
    that does not parse as a JSON object is counted in `unreadable` when one is given."""
    out = []
    for path in sorted(glob.glob(os.path.join(accepted_dir(accepted), "*", "*.jsonl"))):
        route = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                if isinstance(row, dict):
                    out.append((route, row))
                elif unreadable is not None:
                    unreadable["unreadable"] += 1
    return out


# ------------------------------------------------------------------------------------------------------ held-out pictures
def seed_id_of(manifest_row: dict) -> str | None:
    """A manifest row's picture as a seed id, '<source>:<source_id>' (None when the row names no source id)."""
    source, source_id = manifest_row.get("source"), manifest_row.get("source_id")
    return f"{source}:{source_id}" if source and source_id not in (None, "") else None


def held_seed_ids(screen_inputs: str | None, manifest_rows: list, held_splits: tuple, museum: dict | None = None) -> set:
    """The seed ids no training item may show: every picture of the screen set's items and worked examples (screen_inputs, a
    hub path or a local file, read the way screen.load_inputs reads it), every manifest row of the held splits, and, when
    `museum` is given ({"manifest", "split", "sources", "per_source"}), the pictures extract.select_rows picks with those
    settings, which is the held-out museum read's draw."""
    held = set()
    if screen_inputs:
        loaded = screen.load_inputs(screen_inputs)
        pools = list(loaded.get("items") or []) + [ex for exs in (loaded.get("examples") or {}).values() for ex in exs]
        held |= {it["seed_id"] for it in pools if it.get("seed_id")}
    splits = set(held_splits or ())
    held |= {s for s in (seed_id_of(m) for m in manifest_rows if m.get("split") in splits) if s}
    if museum:
        drawn = extract.select_rows(extract.load_manifest(museum["manifest"]), museum.get("split"),
                                    tuple(museum["sources"]) if museum.get("sources") else None, None,
                                    museum.get("per_source"))
        held |= {s for s in (seed_id_of(m) for m in drawn) if s}
    return held


# ------------------------------------------------------------------------------------------------------ picture locators
def manifest_index(manifest_rows: list) -> dict:
    """{(source, source_id): the manifest row's image locator} for the rows that have one."""
    out = {}
    for m in manifest_rows:
        loc = m.get("image")
        if loc and loc.get("kind") not in (None, "pending") and m.get("source") and m.get("source_id") not in (None, ""):
            out[(str(m["source"]), str(m["source_id"]))] = loc
    return out


def locator(seed_id: str, index: dict) -> tuple | None:
    """(image_urls, image locator or None) of a picture, or None when it cannot be found. COCO and Localized Narratives
    pictures are the COCO 2017 images (data.image_urls); a DOCCI picture is a member of the DOCCI archive; any other source is
    looked up in the manifest index by (source, source_id)."""
    source, _, key = seed_id.partition(":")
    if not key:
        return None
    if source in data.PICTURE_SOURCES:
        try:
            return data.image_urls(seed_id), None
        except ValueError:
            return None
    if source == "docci":
        return [], {"kind": "url_archive", "archive_url": DOCCI_ARCHIVE, "member": f"images/{key}.jpg", "example_id": key}
    found = index.get((source, key))
    return ([], found) if found else None


# ------------------------------------------------------------------------------------------------------ the build
NO_PICTURE_LICENSE = "picture in view, its own license not given"


def load_picture_licenses(picture_licenses) -> dict:
    """{seed id: the picture's own license} from a dict or from the path of a JSON file holding one (None: none given)."""
    if not picture_licenses:
        return {}
    if isinstance(picture_licenses, dict):
        return dict(picture_licenses)
    with open(picture_licenses, encoding="utf-8") as fh:
        return dict(json.load(fh))


def make_item(row: dict, route: str, picture: tuple | None, forms_picture: tuple, forms_spec: tuple,
              license_: str | None = None) -> dict:
    """One training item from one accepted row. picture is (the seed id, the locator pair of `locator`) for a picture row,
    None for a specification row. The item is data.item's, with kind, forms and picture locator set from the row's source
    kind (data.item itself knows only the COCO sources), and the row's writer, route, license and own id added. license_, when
    given, is the license the filters used (the picture's own for a row written with its picture in view)."""
    src = dict(row.get("source") or {})
    label = row.get("source_label") or {}
    writer = (row.get("writer") or {}).get("model")
    row_id = row.get("row_id") or f"{row['id']}:{writer}"
    plain = {**row, "source": {**src, "seed_id": None}}          # data.item decides kind from the seed; set it below instead
    it = data.item(plain)
    if picture is not None:
        seed_id, (urls, image) = picture
        it.update(kind="image", seed_id=seed_id, image_urls=list(urls), forms=list(forms_picture),
                  source=seed_id.split(":", 1)[0])
        if image:
            it["image"] = image
    else:
        it.update(kind="spec", seed_id=None, image_urls=[], forms=list(forms_spec), source="synthetic")
    it.update(id=row_id, row_id=row_id, writer=writer, route=route,
              license=license_ or label.get("license") or src.get("license"))
    return it


def build_api_rows(accepted: str, out_path: str, manifest: str | None = None, screen: str | None = DEFAULT_SCREEN,
                   writers=None, licenses=DEFAULT_LICENSES, held_manifest_splits=("probe",), museum: dict | None = None,
                   forms_picture=("full", "short"), forms_spec=("spec_full",), hold_in_view=("coco",),
                   picture_licenses=None) -> dict:
    """Training items from the accepted rows of an outside row-writing system, written to out_path as one input file
    {"version": 3, "built", "accepted", "counts", "items"}; the items come sorted by id, so a build is reproducible.

    accepted: a local folder of accepted/<route>/<task>.jsonl files, or 'hub:<dataset repo id>'. manifest: the picture list
    (a jsonl, local or 'hub:<repo>/<path>'; rows with sample_id, split, source, source_id and an image locator), which finds
    the pictures of sources other than COCO and DOCCI and names the held splits. writers: when given, a row by any other
    writer is dropped. licenses: a row whose license (the picture's, from source_label; a specification row's own when the
    label has none) is not listed is dropped; None keeps every license. Held out of training: the screen set's pictures
    (screen), every manifest row of held_manifest_splits and, when museum is given, the pictures the held-out museum read
    draws (see held_seed_ids). A row written with its picture in view (a non-empty writer.image_size) carries the picture's own
    license as well as the annotations': for such a row whose seed source is in hold_in_view (COCO photos keep each author's
    license, many of them non-commercial), the license the filter reads is picture_licenses' entry for its seed id (a dict, or
    the path of a JSON file of one), and a row with no entry is dropped as "picture in view, its own license not given". A row
    whose image_size is empty or null keeps source_label's license. counts holds the kept items by writer, source, license and kind, and every drop by its reason."""
    dropped = Counter()
    manifest_rows = extract.load_manifest(manifest) if manifest else []
    held = held_seed_ids(screen, manifest_rows, tuple(held_manifest_splits or ()), museum)
    index = manifest_index(manifest_rows)
    allowed_writers = set(writers) if writers is not None else None
    allowed_licenses = set(licenses) if licenses is not None else None
    in_view_sources = set(hold_in_view or ())
    own_licenses = load_picture_licenses(picture_licenses)
    items, seen = [], set()
    for route, row in read_accepted(accepted, dropped):
        writer = (row.get("writer") or {}).get("model")
        if allowed_writers is not None and writer not in allowed_writers:
            dropped["writer"] += 1
            continue
        label = row.get("source_label") or {}
        license_ = label.get("license") or (row.get("source") or {}).get("license")
        seed = str((row.get("source") or {}).get("seed_id") or "")
        if (row.get("writer") or {}).get("image_size") and seed.split(":", 1)[0] in in_view_sources:
            license_ = own_licenses.get(seed)
            if not license_:
                dropped[NO_PICTURE_LICENSE] += 1
                continue
        if allowed_licenses is not None and license_ not in allowed_licenses:
            dropped["license"] += 1
            continue
        src = row.get("source") or {}
        picture = None
        if src.get("kind") == "seeded" and src.get("seed_id"):
            seed_id = str(src["seed_id"])
            if seed_id in held:
                dropped["held_out_picture"] += 1
                continue
            found = locator(seed_id, index)
            if found is None:
                dropped["no_locator"] += 1
                continue
            picture = (seed_id, found)
        elif src.get("kind") != "synthetic":
            dropped["unknown_source_kind"] += 1
            continue
        try:
            item = make_item(row, route, picture, tuple(forms_picture), tuple(forms_spec), license_)
        except KeyError:                       # a row without its thoughts, spec, depth or domain cannot be taught
            dropped["incomplete_row"] += 1
            continue
        if item["id"] in seen:
            dropped["duplicate_row_id"] += 1
            continue
        seen.add(item["id"])
        items.append(item)
    items.sort(key=lambda it: it["id"])
    out = {"version": 3, "built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "accepted": accepted,
           "counts": {"by_writer": dict(Counter(it["writer"] for it in items)),
                      "by_source": dict(Counter(it["source"] for it in items)),
                      "by_license": dict(Counter(str(it["license"]) for it in items)),
                      "by_kind": dict(Counter(it["kind"] for it in items)),
                      "dropped": dict(dropped), "hold_in_view": sorted(in_view_sources),
                      "picture_licenses": (picture_licenses if isinstance(picture_licenses, str)
                                           else {"entries": len(own_licenses)} if picture_licenses else None)},
           "filters": {"writers": sorted(allowed_writers) if allowed_writers is not None else None,
                       "licenses": sorted(allowed_licenses) if allowed_licenses is not None else None,
                       "screen": screen, "held_manifest_splits": list(held_manifest_splits or ()), "museum": museum,
                       "held_pictures": len(held), "manifest": manifest,
                       "forms_picture": list(forms_picture), "forms_spec": list(forms_spec)},
           "items": items}
    data._write(out, out_path)
    return out
