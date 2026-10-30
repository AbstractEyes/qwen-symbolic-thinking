"""The screen set: rows the model is asked to answer, drawn from the accepted rows of the dataset and the seeds they came from.

Rows with a real picture come from the COCO 2017 images (the `coco` and `localized_narratives` seeds name the image by its id;
the image is fetched from the COCO site at screen time, never redistributed). Rows without a picture are drawn rows, whose
specification stands in for the picture. A seed's hint (the captions or the narration the row was written from) travels with
each picture row, to describe the picture in the worked examples and to read the model's chain against."""
from __future__ import annotations

import glob
import json
import os
import random
import re
from collections import defaultdict

from thought_seeds.curate import accepted_files, read_rows

PICTURE_SOURCES = ("coco", "localized_narratives")
COCO_URL = "http://images.cocodataset.org/{split}/{id:012d}.jpg"
COCO_SPLITS = ("val2017", "train2017")


def image_urls(seed_id: str) -> list:
    """The COCO 2017 urls an image id may live at (the seeds walked val2017 first, then train2017)."""
    image_id = int(seed_id.split(":")[1])
    return [COCO_URL.format(split=s, id=image_id) for s in COCO_SPLITS]


def hints(seeds_dir: str, seed_ids: set) -> dict:
    """{seed_id: hint} for the seeds wanted, read from the seed shards of their sources."""
    out = {}
    sources = {s.split(":")[0] for s in seed_ids}
    for src in sorted(sources):
        for path in sorted(glob.glob(os.path.join(seeds_dir, src, "*.jsonl"))):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r'\{"seed_id":\s*"([^"]+)"', line)
                    if m and m.group(1) in seed_ids:
                        out[m.group(1)] = json.loads(line).get("hint", "")
    return out


def item(row: dict, hint: str = "") -> dict:
    src = row.get("source") or {}
    seed_id = src.get("seed_id") or ""
    picture = seed_id.split(":")[0] in PICTURE_SOURCES
    return {"kind": "image" if picture else "spec", "id": row["id"], "depth": row["depth"], "domain": row["domain"],
            "genre": row.get("genre"), "spec": row["spec"], "thoughts": row["thoughts"], "wordings": row.get("wordings") or {},
            "hint": hint, "seed_id": seed_id or None, "image_urls": image_urls(seed_id) if picture else [],
            "license": src.get("license")}


def build_screen(accepted_path: str, seeds_dir: str, out_path: str, n_images: int = 60, n_specs: int = 40, n_examples: int = 2,
                 rng_seed: int = 20261029) -> dict:
    """Draw the screen set and write it as one JSON file: the picture rows (simple and medium, evenly), the specification rows
    (medium and deep, evenly), and per depth the worked examples for the shown form (picture rows outside the set)."""
    rng = random.Random(rng_seed)
    rows = [r for f in accepted_files(accepted_path) for r in read_rows(f)]
    by = defaultdict(list)
    for r in rows:
        src = ((r.get("source") or {}).get("seed_id") or "").split(":")[0]
        kind = "image" if src in PICTURE_SOURCES else ("spec" if not src else None)
        if kind:
            by[(kind, r["depth"])].append(r)
    picks, examples = [], {}
    for depth in ("simple", "medium"):
        pool = by[("image", depth)]
        rng.shuffle(pool)
        picks += pool[:n_images // 2]
        examples[depth] = pool[n_images // 2:n_images // 2 + n_examples]
    for depth in ("medium", "deep"):
        pool = by[("spec", depth)]
        rng.shuffle(pool)
        picks += pool[:n_specs // 2]
    wanted = {(r.get("source") or {}).get("seed_id") for r in picks + sum(examples.values(), [])}
    wanted = {s for s in wanted if s}
    hint = hints(seeds_dir, wanted)
    data = {"version": 1, "rng_seed": rng_seed,
            "items": [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", "")) for r in picks],
            "examples": {d: [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", "")) for r in rs]
                         for d, rs in examples.items()}}
    _write(data, out_path)
    return data


def build_sft(accepted_path: str, seeds_dir: str, screen_path: str, out_path: str, rng_seed: int = 20261030) -> dict:
    """The fine-tune set: every accepted picture row outside the screen set (and its examples), and as many drawn rows, drawn
    simple : medium : deep = 1 : 1 : 2 so the deep depth is taught through the specifications."""
    rng = random.Random(rng_seed)
    with open(screen_path, encoding="utf-8") as fh:
        screen = json.load(fh)
    held = {it["id"] for it in screen["items"]} | {ex["id"] for exs in screen["examples"].values() for ex in exs}
    rows = [r for f in accepted_files(accepted_path) for r in read_rows(f)]
    pictures, drawn = [], defaultdict(list)
    for r in rows:
        if r["id"] in held:
            continue
        src = ((r.get("source") or {}).get("seed_id") or "").split(":")[0]
        if src in PICTURE_SOURCES:
            pictures.append(r)
        elif not src:
            drawn[r["depth"]].append(r)
    n = len(pictures)
    want = {"simple": n // 4, "medium": n // 4, "deep": n - 2 * (n // 4)}
    specs = []
    for depth, k in want.items():
        pool = drawn[depth]
        rng.shuffle(pool)
        specs += pool[:k]
    wanted = {(r.get("source") or {}).get("seed_id") for r in pictures}
    hint = hints(seeds_dir, {s for s in wanted if s})
    items = [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", "")) for r in pictures] + [item(r) for r in specs]
    rng.shuffle(items)
    data = {"version": 1, "rng_seed": rng_seed, "screen": os.path.basename(screen_path), "held_out": sorted(held),
            "items": items}
    _write(data, out_path)
    return data


def _write(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
