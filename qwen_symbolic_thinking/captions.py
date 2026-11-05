"""Captions and booru tags for a zip of photos: a stock vision-language model writes one caption per photo from a caption ask
(the photo's own description may follow it as a helper line), and a booru tagger tags the same photo.

The photos are read from a zip archive (the next chunk decoded while the current one generates); the VLM runs a batch of photos
in one greedy generation (a batch that runs out of memory is retried in halves), the tagger the same batch. Rows go to
extractions/<stamp>/rows.jsonl in the dataset repo, shipped as the run goes (a copy of the file, uploaded in a background thread,
with retries), so a lost runtime loses at most one shipment; run(resume="extractions/<stamp>") goes on from a shipped folder.
A row keeps the photo's file name, the caption, the tags with their scores and how much of the caption repeats the helper
line's wording word for word; the helper line itself is not written. With pose_fit, a 3D body fit (pose.py) adds the body's
angles to each row and, with pose_ask, a pose line to the ask; model_id=None runs the fit (and the tagger) without captions."""
from __future__ import annotations

import collections
import gc
import io
import json
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from . import __version__, booru, family
from .hub import token
from .screen import LIBRARY_NOTES, fast_path_note, generate, load_model, preflight

# A caption ask tried by hand on fashion photos: with the photo's own description as the helper line, the stock 4B's captions
# needed few corrections.
FASHION_ASK = ("Describe this picture in one or two sentences: who or what is in it, each garment with its color and any print or "
               "text on it, the pose and the setting. Answer with the visible description ONLY, explaining the specifics of each "
               "garment piece that you recognize. Do not describe print or text if none is present.\n"
               "Helper caption: {original}")
WORD = re.compile(r"[a-z0-9']+")


def one_line(text: str) -> str:
    return " ".join(str(text or "").split())


POSE_LINE = "Helper pose: {pose}"     # added to an ask without {pose} when run(pose_ask=True)


def fill(ask: str, original: str, pose_line: str = "") -> str:
    """The ask for one photo: {original} becomes the photo's own description on one line, {pose} its pose line; a line that
    names one of them while it is empty is left out."""
    values = {"{original}": one_line(original), "{pose}": one_line(pose_line)}
    lines = []
    for line in ask.split("\n"):
        if any(key in line and not value for key, value in values.items()):
            continue
        for key, value in values.items():
            line = line.replace(key, value)
        lines.append(line)
    return "\n".join(lines).strip()


def copy_share(caption: str, original: str, n: int = 5) -> float | None:
    """The share of the caption's n-word runs found word for word in the original; None without an original or for a caption
    shorter than n words."""
    a, b = WORD.findall(str(caption or "").lower()), WORD.findall(str(original or "").lower())
    if len(a) < n or len(b) < n:
        return None
    theirs = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    runs = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    return round(sum(r in theirs for r in runs) / len(runs), 3)


def append_pose(caption_rows: str, pose_rows: str, out_path: str) -> dict:
    """Join a caption run's rows with a pose run's rows (run(pose_fit=...)) by file name: each caption row gains the pose record
    (rebuilt from its keypoints with this version's angles and words; the keypoints stay in the pose run's rows), the pose sentence
    (pose_line) and caption_pose, the caption with that sentence appended. A photo without a fitted body (a flat lay) keeps its
    caption alone. Returns the counts."""
    from . import pose
    with open(pose_rows, encoding="utf-8") as fh:
        poses = {r["file"]: r for r in map(json.loads, fh)}
    counts, seen = collections.Counter(), set()
    with open(caption_rows, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out:
        for line in fh:
            row = json.loads(line)
            seen.add(row["file"])
            rec = pose.rebuild((poses.get(row["file"]) or {}).get("pose"))
            if rec is not None:
                rec.pop("keypoints_3d", None)
                said = pose.sentence(rec)
                row.update({"pose": rec, "pose_line": said, "caption_pose": f"{row.get('caption', '')} {said}".strip()})
                counts["with a pose"] += 1
            else:
                row.update({"pose": None, "pose_line": "", "caption_pose": row.get("caption", "")})
                counts["without a pose"] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["pose rows without a caption"] = len(set(poses) - seen)
    return dict(counts)


def photos(images_zip: str, n: int | None = None) -> list:
    """The zip's photos (JPEG / PNG members) in name order, the first n when n is given."""
    with zipfile.ZipFile(images_zip) as zf:
        names = sorted(m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png")))
    return names[:n] if n else names


def run(images_zip: str, descriptions: str | None = None, out_repo: str = "AbstractPhil/qwen-thinking-images",
        model_id: str | None = "Qwen/Qwen3.5-4B", ask: str = FASHION_ASK, batch: int = 64, chunk: int = 512,
        max_new_tokens: int = 160, tagger: str | None = "animetimm/eva_giant_patch14_560.dbv4-full", tag_max: int = booru.MAX_TAGS,
        tag_omit: tuple = (), tag_min_precision: float = 0.0, n: int | None = None, out_dir: str = "/content/captions",
        tag: str = "captions", ship_every: int = 2048, resume: str | None = None, repo_type: str = "dataset", do_ship: bool = True,
        pose_fit: str | None = None, pose_ask: bool = False, pose_skip=None) -> dict:
    """Caption and tag every photo of the zip (n: the first n in name order) and ship the rows; returns the summary.
    descriptions: a JSON file {file name: description} whose text fills the ask's {original}. tag_omit / tag_min_precision: the
    reviewer's filter (booru.keep), applied to each row's tags_kept beside the full tag list. pose_fit: the 3D body fit's
    weights (pose.WEIGHTS; None: no fit); pose_ask: its pose line fills the ask's {pose} (or a POSE_LINE added to the ask);
    pose_skip: a test on the file name for photos the fit leaves out (a flat lay); model_id=None: no captions."""
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    stamp = resume.rstrip("/").split("/")[-1] if resume else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    folder, ship_dir = os.path.join(out_dir, stamp), os.path.join(out_dir, "ship", stamp)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(ship_dir, exist_ok=True)
    rows_path, missing_path = os.path.join(folder, "rows.jsonl"), os.path.join(folder, "missing.jsonl")
    if resume and not os.path.exists(rows_path):
        got = hf_hub_download(out_repo, f"extractions/{stamp}/rows.jsonl", repo_type=repo_type, token=token(),
                              local_dir=os.path.join(out_dir, "resume"))
        shutil.copyfile(got, rows_path)
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path, encoding="utf-8") as fh:
            done = {json.loads(line)["file"] for line in fh if line.strip()}
    env = preflight()
    texts = json.load(open(descriptions, encoding="utf-8")) if descriptions else {}
    members = photos(images_zip, n)
    todo = [m for m in members if os.path.basename(m) not in done]
    print(f"captions {stamp}: {len(todo)} photos to do of {len(members)} ({len(done)} done before); {model_id or 'no captions'}"
          + (f" + {tagger}" if tagger else "") + (f" + {pose_fit}" + (" (in the ask)" if pose_ask else "") if pose_fit else "")
          + f"; batch {batch}, chunk {chunk}, {max_new_tokens} new tokens at most", flush=True)
    processor, model = load_model(model_id, False) if model_id else (None, None)
    mark = family.marker(model) if model is not None else None
    tags_bundle = booru.load(booru.fetch(tagger, os.path.join(out_dir, "taggers"), token()), booru.kind_of(tagger)) if tagger else None
    prec = tags_bundle["precision"] if tags_bundle else {}
    fit = None
    if pose_fit:
        from . import pose
        fit = pose.load(pose_fit)
        if pose_ask and "{pose}" not in ask:
            ask = ask + "\n" + POSE_LINE
    api = HfApi(token=token()) if do_ship else None
    zf = zipfile.ZipFile(images_zip)
    reader, shipper = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    oom, copies, poses, seconds = collections.Counter(), [], collections.Counter(), collections.Counter()
    t0, n_done, n_tok, n_missing, since_ship, shipping = time.time(), 0, 0, 0, 0, None

    def decode(group):
        out = {}
        for m in group:
            try:
                im = Image.open(io.BytesIO(zf.read(m)))
                im.load()
                out[m] = im.convert("RGB")
            except Exception as e:  # noqa: BLE001
                out[m] = f"{type(e).__name__}: {str(e)[:200]}"
        return out

    def gen(part, pictures, asks):
        convs = [[{"role": "user", "content": family.content(asks[m], pictures[m], mark)}] for m in part]
        try:
            return generate(processor, model, convs, [pictures[m] for m in part], max_new_tokens, False, None)
        except Exception as e:  # noqa: BLE001
            is_oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
            gc.collect()
            torch.cuda.empty_cache()
            if is_oom and len(part) > 1:
                oom["splits"] += 1
                h = len(part) // 2
                a, b = gen(part[:h], pictures, asks), gen(part[h:], pictures, asks)
                return a[0] + b[0], a[1] + b[1]
            oom["rows_failed" if is_oom else "errors"] += len(part)
            return [f"<error {type(e).__name__}: {str(e)[:200]}>"] * len(part), [0] * len(part)

    def upload(count: int, final: bool) -> bool:
        for attempt in range(5):
            try:
                api.upload_folder(folder_path=ship_dir, path_in_repo=f"extractions/{stamp}", repo_id=out_repo, repo_type=repo_type,
                                  commit_message=f"captions {stamp}: {count} rows" + (" (final)" if final else ""))
                print(f"  shipped {count} rows to {out_repo}/extractions/{stamp}", flush=True)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"  ship failed ({type(e).__name__}: {str(e)[:120]}); retry", flush=True)
                time.sleep(15 * (attempt + 1))
        return False

    def snapshot():                                   # on this thread, between batches: never a half-written line
        for name in ("rows.jsonl", "missing.jsonl", "summary.json"):
            if os.path.exists(os.path.join(folder, name)):
                shutil.copyfile(os.path.join(folder, name), os.path.join(ship_dir, name))

    chunks = [todo[c0:c0 + chunk] for c0 in range(0, len(todo), chunk)]
    pending = reader.submit(decode, chunks[0]) if chunks else None
    for ci, group in enumerate(chunks):
        pictures = pending.result()
        pending = reader.submit(decode, chunks[ci + 1]) if ci + 1 < len(chunks) else None
        bad = {m: v for m, v in pictures.items() if isinstance(v, str)}
        if bad:
            with open(missing_path, "a", encoding="utf-8") as fh:
                for m, why in bad.items():
                    fh.write(json.dumps({"file": os.path.basename(m), "why": why}) + "\n")
            n_missing += len(bad)
        group = [m for m in group if m not in bad]
        for b0 in range(0, len(group), batch):
            part = group[b0:b0 + batch]
            fits, lines = {}, {}
            if fit is not None:
                ts = time.time()
                for m in part:
                    if pose_skip is not None and pose_skip(os.path.basename(m)):
                        fits[m] = "skipped"
                        continue
                    try:
                        fits[m] = pose.predict(fit, pictures[m]) or "none"
                    except Exception as e:  # noqa: BLE001
                        fits[m] = f"<error {type(e).__name__}: {str(e)[:200]}>"
                    if isinstance(fits[m], dict):
                        lines[m] = pose.words(fits[m])
                seconds["pose"] += time.time() - ts
            asks = {m: fill(ask, texts.get(os.path.basename(m), ""), lines.get(m, "") if pose_ask else "") for m in part}
            ts = time.time()
            outs, lengths = gen(part, pictures, asks) if model is not None else ([""] * len(part), [0] * len(part))
            seconds["caption"] += time.time() - ts
            ts = time.time()
            found = booru.predict_batch(tags_bundle, [pictures[m] for m in part], None, tag_max) if tags_bundle else [[]] * len(part)
            seconds["tags"] += time.time() - ts
            n_tok += sum(lengths)
            with open(rows_path, "a", encoding="utf-8") as fh:
                for m, text, k, tags in zip(part, outs, lengths, found):
                    file = os.path.basename(m)
                    if text.startswith("<error "):
                        with open(missing_path, "a", encoding="utf-8") as mh:
                            mh.write(json.dumps({"file": file, "why": text}) + "\n")
                        n_missing += 1
                        continue
                    row = {"file": file}
                    if model is not None:
                        original = texts.get(file, "")
                        caption = booru.us(one_line(text))
                        share = copy_share(caption, original)
                        if share is not None:
                            copies.append(share)
                        row.update({"caption": caption, "new_tokens": k, "truncated": k >= max_new_tokens,
                                    "helped": bool(one_line(original)), "copy5": share})
                    row.update({"tags": [[booru.tag_name(t), round(s, 4), c, prec.get(t)] for t, s, c in tags],
                                "tags_kept": ", ".join(booru.tag_name(t) for t, s, c in tags
                                                       if booru.keep(t, prec.get(t), tag_omit, tag_min_precision))})
                    if fit is not None:
                        got = fits.get(m)
                        status = "fitted" if isinstance(got, dict) else ("error" if str(got).startswith("<error ") else got)
                        poses[status] += 1
                        row.update({"pose": got if isinstance(got, dict) else None, "pose_status": status,
                                    "pose_line": lines.get(m, ""), "pose_asked": bool(pose_ask and lines.get(m))})
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += len(part)
            since_ship += len(part)
            spent = time.time() - t0
            print(f"  {n_done}/{len(todo)} photos, {spent:.0f} s spent, {spent / n_done * (len(todo) - n_done):.0f} s left, "
                  f"{spent / n_done:.3f} s a photo, {n_tok / max(spent, 1e-9):.0f} tok/s"
                  + (f"; pose {seconds['pose'] / n_done:.3f} s a photo, {dict(poses)}" if fit is not None else ""), flush=True)
            if n_done == len(part):
                if model is not None:
                    print("  linear attention:", fast_path_note(), flush=True)
                    print("  first caption:", booru.us(one_line(outs[0]))[:300], flush=True)
                if lines:
                    print("  first pose line:", next(iter(lines.values())), flush=True)
            if api and since_ship >= ship_every:
                if shipping is not None:
                    shipping.result()                 # the previous upload has finished before its files are replaced
                snapshot()
                shipping = shipper.submit(upload, len(done) + n_done, False)
                since_ship = 0
    reader.shutdown(wait=False)
    spent = time.time() - t0
    summary = {"package": __version__, "model": model_id, "tagger": tagger, "ask": ask, "images_zip": os.path.basename(images_zip),
               "descriptions": os.path.basename(descriptions) if descriptions else None, "photos": len(members),
               "done_before": len(done), "done_now": n_done, "missing": n_missing, "oom": dict(oom), "batch": batch, "chunk": chunk,
               "max_new_tokens": max_new_tokens, "tag_filter": {"omit": list(tag_omit), "min_precision": tag_min_precision},
               "s_per_photo": round(spent / max(n_done, 1), 3), "tok_per_s": round(n_tok / max(spent, 1e-9), 1),
               "copy5_mean": round(sum(copies) / len(copies), 3) if copies else None,
               "copy5_over_half": sum(1 for c in copies if c >= 0.5),
               "s_per_photo_by_op": {k: round(v / max(n_done, 1), 4) for k, v in seconds.items()},
               "pose": {"weights": pose_fit, "commit": fit["commit"], "asked": pose_ask, "counts": dict(poses)} if fit else None,
               "env": env, "library_notes": LIBRARY_NOTES[-20:]}
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\ncaptions done: {n_done} photos in {spent / 60:.1f} min ({summary['s_per_photo']} s a photo); {n_missing} missing; "
          f"out of memory {dict(oom)}; copy5 mean {summary['copy5_mean']}, {summary['copy5_over_half']} rows at 0.5 or more"
          + (f"; pose {dict(poses)}" if fit else "") + f"; seconds a photo by op {summary['s_per_photo_by_op']}", flush=True)
    if api:
        if shipping is not None:
            shipping.result()
        snapshot()
        upload(len(done) + n_done, True)
    shipper.shutdown(wait=True)
    zf.close()
    if fit is not None:
        pose.unload(fit)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    summary["path_in_repo"] = f"extractions/{stamp}"
    return summary
