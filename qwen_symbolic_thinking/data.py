"""The data sets: rows the model is asked to answer or taught from, drawn from the accepted rows of the dataset and the seeds
they came from.

Rows with a real picture come from the COCO 2017 images (the `coco` and `localized_narratives` seeds name the image by its id;
the image is fetched from the COCO site at run time, never redistributed). Rows without a picture are drawn rows, whose
specification stands in for the picture. A seed's hint (the captions or the narration the row was written from) travels with
each picture row, to describe the picture in the worked examples and to read a chain's grounding against.

An item carries `forms`: the prompt forms it is taught under (short: the chain and a caption; full: the whole row with the
wordings and the questions; spec_short and spec_full for a specification instead of a picture)."""
from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict

from thought_seeds.curate import accepted_files, read_rows

PICTURE_SOURCES = ("coco", "localized_narratives")
COCO_URL = "http://images.cocodataset.org/{split}/{id:012d}.jpg"
COCO_SPLITS = ("val2017", "train2017")


def image_urls(seed_id: str) -> list:
    """The COCO 2017 urls an image id may live at (the seeds walked val2017 first, then train2017)."""
    image_id = int(seed_id.split(":")[1])
    return [COCO_URL.format(split=s, id=image_id) for s in COCO_SPLITS]


def seed_lines(seeds_dir: str, sources: tuple):
    """Every seed of the sources, as parsed records, one shard at a time."""
    for src in sources:
        for path in sorted(glob.glob(os.path.join(seeds_dir, src, "*.jsonl"))):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def hints(seeds_dir: str, seed_ids: set) -> dict:
    """{seed_id: hint} for the seeds wanted, read from the seed shards of their sources."""
    sources = tuple(sorted({s.split(":")[0] for s in seed_ids}))
    return {d["seed_id"]: d.get("hint", "") for d in seed_lines(seeds_dir, sources) if d.get("seed_id") in seed_ids}


def item(row: dict, hint: str = "", forms: tuple | None = None) -> dict:
    src = row.get("source") or {}
    seed_id = src.get("seed_id") or ""
    picture = seed_id.split(":")[0] in PICTURE_SOURCES
    return {"kind": "image" if picture else "spec", "id": row["id"], "depth": row["depth"], "domain": row["domain"],
            "genre": row.get("genre"), "spec": row["spec"], "thoughts": row["thoughts"], "wordings": row.get("wordings") or {},
            "questions": row.get("questions") or [], "hint": hint, "seed_id": seed_id or None,
            "image_urls": image_urls(seed_id) if picture else [], "license": src.get("license"),
            "forms": list(forms) if forms else (["short"] if picture else ["spec_short"])}


def _rows(accepted_path: str) -> list:
    return [r for f in accepted_files(accepted_path) for r in read_rows(f)]


def _kind(row: dict) -> str | None:
    src = ((row.get("source") or {}).get("seed_id") or "").split(":")[0]
    return "image" if src in PICTURE_SOURCES else ("spec" if not src else None)


def _held(screen_path: str) -> set:
    with open(screen_path, encoding="utf-8") as fh:
        screen = json.load(fh)
    return {it["id"] for it in screen["items"]} | {ex["id"] for exs in screen["examples"].values() for ex in exs}


def build_screen(accepted_path: str, seeds_dir: str, out_path: str, n_images: int = 60, n_specs: int = 40, n_examples: int = 2,
                 rng_seed: int = 20261029) -> dict:
    """Draw the screen set and write it as one JSON file: the picture rows (simple and medium, evenly), the specification rows
    (medium and deep, evenly), and per depth the worked examples for the shown form (picture rows outside the set)."""
    rng = random.Random(rng_seed)
    by = defaultdict(list)
    for r in _rows(accepted_path):
        kind = _kind(r)
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
    hint = hints(seeds_dir, {s for s in wanted if s})
    data = {"version": 1, "rng_seed": rng_seed,
            "items": [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", "")) for r in picks],
            "examples": {d: [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", "")) for r in rs]
                         for d, rs in examples.items()}}
    _write(data, out_path)
    return data


def build_sft(accepted_path: str, seeds_dir: str, screen_path: str, out_path: str, rng_seed: int = 20261030) -> dict:
    """The first fine-tune set: every accepted picture row outside the screen set (and its examples), and as many drawn rows,
    drawn simple : medium : deep = 1 : 1 : 2 so the deep depth is taught through the specifications."""
    rng = random.Random(rng_seed)
    held = _held(screen_path)
    pictures, drawn = [], defaultdict(list)
    for r in _rows(accepted_path):
        if r["id"] in held:
            continue
        kind = _kind(r)
        if kind == "image":
            pictures.append(r)
        elif kind == "spec":
            drawn[r["depth"]].append(r)
    n = len(pictures)
    want = {"simple": n // 4, "medium": n // 4, "deep": n - 2 * (n // 4)}
    specs = []
    for depth, k in want.items():
        pool = drawn[depth]
        rng.shuffle(pool)
        specs += pool[:k]
    hint = hints(seeds_dir, {(r.get("source") or {}).get("seed_id") for r in pictures})
    items = [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", "")) for r in pictures] + [item(r) for r in specs]
    rng.shuffle(items)
    data = {"version": 1, "rng_seed": rng_seed, "screen": os.path.basename(screen_path), "held_out": sorted(held),
            "items": items}
    _write(data, out_path)
    return data


def build_pictures(accepted_path: str, seeds_dir: str, screen_path: str, out_path: str, forms: tuple = ("full", "short")) -> dict:
    """Every accepted picture row with a caption outside the screen set, with the whole row and the forms it is taught under
    (a row whose only wording is tags would teach an answer without a caption, so it is left out)."""
    held = _held(screen_path)
    pictures = [r for r in _rows(accepted_path)
                if r["id"] not in held and _kind(r) == "image" and (r.get("wordings") or {}).get("caption")]
    hint = hints(seeds_dir, {(r.get("source") or {}).get("seed_id") for r in pictures})
    items = [item(r, hint.get((r.get("source") or {}).get("seed_id") or "", ""), forms) for r in pictures]
    data = {"version": 2, "screen": os.path.basename(screen_path), "held_out": sorted(held), "forms": list(forms), "items": items}
    _write(data, out_path)
    return data


def build_drawn(accepted_path: str, screen_path: str, out_path: str, depths: tuple = ("deep",), n: int = 500,
                forms: tuple = ("spec_full",), rng_seed: int = 20261030) -> dict:
    """Drawn rows of the depths asked for, outside the screen set, n in all, with the whole row."""
    rng = random.Random(rng_seed)
    held = _held(screen_path)
    pool = [r for r in _rows(accepted_path) if r["id"] not in held and _kind(r) == "spec" and r["depth"] in depths]
    rng.shuffle(pool)
    items = [item(r, "", forms) for r in pool[:n]]
    data = {"version": 2, "screen": os.path.basename(screen_path), "depths": list(depths), "forms": list(forms), "items": items}
    _write(data, out_path)
    return data


def build_synth_seeds(accepted_path: str, seeds_dir: str, screen_path: str, out_path: str, n: int = 3500, source: str = "coco",
                      depth: str = "medium", rng_seed: int = 20261030) -> dict:
    """Pictures no accepted row was written about: seeds of one source, with their captions and specifications, for the base
    model to think through; the chains it writes are audited and become synthetic picture items (synth.generate)."""
    rng = random.Random(rng_seed)
    used = {(r.get("source") or {}).get("seed_id") for r in _rows(accepted_path)}
    held = _held(screen_path)
    pool = [d for d in seed_lines(seeds_dir, (source,)) if d.get("seed_id") not in used and d.get("hint")]
    rng.shuffle(pool)
    items = []
    for d in pool[:n]:
        items.append({"kind": "image", "id": f"syn-{d['seed_id'].replace(':', '-')}", "depth": depth, "domain": d.get("domain", "photograph"),
                      "genre": d.get("genre"), "spec": d.get("spec") or {}, "thoughts": [], "wordings": {}, "questions": [],
                      "hint": d.get("hint", ""), "seed_id": d["seed_id"], "image_urls": image_urls(d["seed_id"]),
                      "license": (d.get("source") or {}).get("license"), "forms": ["short"]})
    data = {"version": 1, "source": source, "depth": depth, "rng_seed": rng_seed, "excluded_rows": len(held), "items": items}
    _write(data, out_path)
    return data


def _write(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
