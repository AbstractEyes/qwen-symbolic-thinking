"""Extraction: the trained model writes whole rows for the pictures of a manifest (the qwen-thinking-images list), each row
audited and shipped.

    from qwen_symbolic_thinking.extract import run
    run(manifest="hub:AbstractPhil/qwen-thinking-images/manifest_v1.jsonl", adapter="AbstractPhil/qwen3.5-9b-think-symbolic-image",
        per_source=250, tag="pilot")

A manifest row names its picture by a locator: a url (with fallbacks), a row of a parquet shard on the Hub (the image column),
a file on the Hub, a member of a zip on the Hub, or a keyed row of a Hub dataset split. Pictures are fetched in groups so a
shard or a zip is downloaded once. Each picture is asked for the whole row at the depth the manifest plans for it; the answer
is parsed, scored (chain.score, with the row's captions as the grounding hint when it has any) and audited the way the
synthetic rows were (form, depth, no definitions, no remarks, no stray lines; grounding gates only where captions exist)."""
from __future__ import annotations

import collections
import gc
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone

from . import __version__, chain, prompts
from .hub import token
from .screen import LIBRARY_NOTES, fast_path_note, generate, load_model, preflight
from .synth import audit

DEPTHS = {"quick": "simple", "closer": "medium", "long": "deep"}
CAPTION_KEYS_IGNORED = ("vlm_0.8b_json", "animetimm_json")      # structured JSON captions are not grounding text


def load_manifest(manifest: str) -> list:
    """Rows of a manifest: a local jsonl path, or hub:<repo>/<path> (a dataset repo)."""
    if manifest.startswith("hub:"):
        from huggingface_hub import hf_hub_download
        parts = manifest[4:].split("/")
        manifest = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]), repo_type="dataset", token=token())
    with open(manifest, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def hint_of(row: dict) -> str:
    """The grounding text of a manifest row: every plain-text caption it carries, joined."""
    caps = row.get("captions") or {}
    return " ".join(str(v) for k, v in caps.items() if k not in CAPTION_KEYS_IGNORED and isinstance(v, str) and v.strip())


def _image_from_bytes(b: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")


def _fetch_url(urls: list, cache_path: str):
    import requests
    from PIL import Image
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")
    for url in urls:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "qwen-symbolic-thinking/" + __version__})
        except requests.RequestException:
            continue
        if r.status_code == 200 and len(r.content) > 1000:
            try:
                im = _image_from_bytes(r.content)
            except Exception:
                continue
            with open(cache_path, "wb") as fh:
                fh.write(r.content)
            return im
    return None


def fetch_images(rows: list, cache_dir: str, workers: int = 16) -> dict:
    """{sample_id: image} for the rows whose picture answers; grouped by locator kind so shards and zips download once."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[(r.get("image") or {}).get("kind", "pending")].append(r)
    # urls, in parallel
    url_rows = by_kind.get("url", [])
    with ThreadPoolExecutor(workers) as pool:
        ims = list(pool.map(lambda r: _fetch_url(r["image"]["urls"], os.path.join(cache_dir, r["sample_id"] + ".img")), url_rows))
    out.update({r["sample_id"]: im for r, im in zip(url_rows, ims) if im is not None})
    # parquet shards: one download and one read per shard
    by_shard = collections.defaultdict(list)
    for r in by_kind.get("hf_parquet", []):
        by_shard[(r["image"]["repo"], r["image"].get("repo_type", "dataset"), r["image"]["path"])].append(r)
    for (repo, rtype, path), rs in by_shard.items():
        try:
            import pyarrow.parquet as pq
            local = hf_hub_download(repo, path, repo_type=rtype, token=token())
            pf = pq.ParquetFile(local)
            wanted = {r["image"]["row_index"]: r for r in rs}
            col = rs[0]["image"].get("column", "image")
            start = 0
            for g in range(pf.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                idx = [i for i in wanted if start <= i < start + n]
                if idx:
                    tbl = pf.read_row_group(g, columns=[col])
                    for i in idx:
                        cell = tbl.column(col)[i - start].as_py()
                        b = cell.get("bytes") if isinstance(cell, dict) else cell
                        try:
                            out[wanted[i]["sample_id"]] = _image_from_bytes(b)
                        except Exception:
                            pass
                start += n
        except Exception as e:
            print(f"  shard {path} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    # single files on the Hub
    def one_file(r):
        try:
            local = hf_hub_download(r["image"]["repo"], r["image"]["path"], repo_type=r["image"].get("repo_type", "dataset"), token=token())
            return _image_from_bytes(open(local, "rb").read())
        except Exception:
            return None
    file_rows = by_kind.get("hf_file", [])
    with ThreadPoolExecutor(workers) as pool:
        ims = list(pool.map(one_file, file_rows))
    out.update({r["sample_id"]: im for r, im in zip(file_rows, ims) if im is not None})
    # zip members: one download per zip
    by_zip = collections.defaultdict(list)
    for r in by_kind.get("zip_member", []):
        by_zip[(r["image"]["repo"], r["image"].get("repo_type", "dataset"), r["image"]["path"])].append(r)
    for (repo, rtype, path), rs in by_zip.items():
        try:
            local = hf_hub_download(repo, path, repo_type=rtype, token=token())
            with zipfile.ZipFile(local) as zf:
                for r in rs:
                    try:
                        out[r["sample_id"]] = _image_from_bytes(zf.read(r["image"]["member"]))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  zip {path} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    # keyed rows of a Hub dataset split (DOCCI): the split's parquet files scanned once for the keys
    by_split = collections.defaultdict(list)
    for r in by_kind.get("hf_dataset_row", []):
        by_split[(r["image"]["repo"], r["image"].get("split", ""))].append(r)
    for (repo, split), rs in by_split.items():
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi
            files = sorted(s.rfilename for s in HfApi(token=token()).dataset_info(repo).siblings
                           if s.rfilename.endswith(".parquet") and (not split or f"/{split}-" in s.rfilename or f"/{split}/" in s.rfilename))
            wanted = {r["image"]["key"]: r for r in rs}
            key_col, img_col = rs[0]["image"]["key_column"], rs[0]["image"]["image_column"]
            for f in files:
                if not wanted:
                    break
                local = hf_hub_download(repo, f, repo_type="dataset", token=token())
                pf = pq.ParquetFile(local)
                for g in range(pf.num_row_groups):
                    keys = pf.read_row_group(g, columns=[key_col]).column(key_col).to_pylist()
                    hits = [(i, k) for i, k in enumerate(keys) if k in wanted]
                    if hits:
                        imgs = pf.read_row_group(g, columns=[img_col]).column(img_col)
                        for i, k in hits:
                            cell = imgs[i].as_py()
                            b = cell.get("bytes") if isinstance(cell, dict) else cell
                            try:
                                out[wanted.pop(k)["sample_id"]] = _image_from_bytes(b)
                            except Exception:
                                wanted.pop(k, None)
        except Exception as e:
            print(f"  dataset rows {repo}/{split} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    missing = [r["sample_id"] for r in rows if r["sample_id"] not in out]
    if missing:
        print(f"  {len(missing)} pictures did not answer (first {missing[:5]})", flush=True)
    return out


def run(manifest: str, out_repo: str = "AbstractPhil/qwen-thinking-images", model_id: str = "Qwen/Qwen3.5-9B",
        adapter: str | None = "AbstractPhil/qwen3.5-9b-think-symbolic-image", adapter_subfolder: str | None = None,
        per_source: int | None = None, sources: tuple | None = None, n: int | None = None, batch: int = 64,
        max_new_tokens: int = 400, min_backed: float = 0.2, min_said: float = 0.1, chunk: int = 512,
        out_dir: str = "/content/extract", repo_type: str = "dataset", do_ship: bool = True, tag: str = "pilot") -> dict:
    """Extract whole rows for the manifest's pictures; ship extractions/<stamp>_<tag>/ to the dataset repo."""
    import torch
    from huggingface_hub import HfApi
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    env = preflight()
    rows = load_manifest(manifest)
    rows = [r for r in rows if (r.get("image") or {}).get("kind") not in (None, "pending")]
    if sources:
        rows = [r for r in rows if r["source"] in set(sources)]
    if per_source:
        seen = collections.Counter()
        kept = []
        for r in rows:
            if seen[r["source"]] < per_source:
                kept.append(r)
                seen[r["source"]] += 1
        rows = kept
    rows = rows[:n]
    print(f"extract {stamp}: {len(rows)} pictures from {len({r['source'] for r in rows})} sources; {model_id}"
          + (f" + {adapter}/{adapter_subfolder or ''}" if adapter else ""), flush=True)
    processor, model = load_model(model_id, False, adapter, adapter_subfolder)
    results_path = os.path.join(out_dir, "rows.jsonl")
    counts, reasons, t0, n_done, n_tok, n_missing = collections.Counter(), collections.Counter(), time.time(), 0, 0, 0
    said, backed = collections.defaultdict(list), collections.defaultdict(list)
    for c0 in range(0, len(rows), chunk):
        group = rows[c0:c0 + chunk]
        pictures = fetch_images(group, os.path.join(os.path.dirname(out_dir), "images"))
        n_missing += sum(1 for r in group if r["sample_id"] not in pictures)
        group = [r for r in group if r["sample_id"] in pictures]
        for b0 in range(0, len(group), batch):
            part = group[b0:b0 + batch]
            convs = [[{"role": "user", "content": [{"type": "image", "image": pictures[r["sample_id"]]},
                                                   {"type": "text", "text": prompts.full(DEPTHS.get(r.get("depth_plan"), "medium"))}]}] for r in part]
            try:
                texts, lengths = generate(processor, model, convs, [pictures[r["sample_id"]] for r in part], max_new_tokens, False, None)
            except Exception as e:
                texts, lengths = [f"<error {type(e).__name__}: {str(e)[:200]}>"] * len(part), [0] * len(part)
            n_tok += sum(lengths)
            with open(results_path, "a", encoding="utf-8") as fh:
                for r, text in zip(part, texts):
                    depth = DEPTHS.get(r.get("depth_plan"), "medium")
                    hint = hint_of(r)
                    parsed = chain.parse(text)
                    scores = chain.score(parsed, {"depth": depth, "spec": {}, "thoughts": [], "hint": hint})
                    why = audit(scores, min_backed if hint else 0.0, min_said if hint else 0.0)
                    if why is None and not scores.get("full_ok"):
                        why = "row shape"
                    counts["kept" if why is None else "rejected"] += 1
                    if why:
                        reasons[why] += 1
                    if hint and scores.get("hint_said") is not None:
                        said[r["source"]].append(scores["hint_said"])
                        backed[r["source"]].append(scores["hint_backed"] or 0)
                    fh.write(json.dumps({"sample_id": r["sample_id"], "source": r["source"], "group": r["group"], "depth": depth,
                                         "text": text, "thoughts": parsed["thoughts"], "wordings": parsed["wordings"],
                                         "questions": parsed["questions"], "scores": scores, "rejected": why,
                                         "hinted": bool(hint), "image": r["image"]}, ensure_ascii=False) + "\n")
            n_done += len(part)
            spent = time.time() - t0
            print(f"  {n_done}/{len(rows)} rows, kept {counts['kept']} ({counts['kept'] / max(n_done, 1):.0%}), {spent:.0f} s spent, "
                  f"{spent / n_done * (len(rows) - n_done):.0f} s left, {n_tok / max(spent, 1e-9):.0f} tok/s", flush=True)
            if n_done == len(part):
                print("  linear attention:", fast_path_note(), flush=True)
    by_source = {}
    for src in sorted({r["source"] for r in rows}):
        by_source[src] = {"rows": sum(1 for r in rows if r["source"] == src),
                          "said": (sum(said[src]) / len(said[src])) if said[src] else None,
                          "backed": (sum(backed[src]) / len(backed[src])) if backed[src] else None}
    summary = {"package": __version__, "model": model_id, "adapter": adapter, "adapter_subfolder": adapter_subfolder, "manifest": manifest,
               "rows": len(rows), "pictures_missing": n_missing, "kept": counts["kept"], "rejected": dict(reasons),
               "s_per_row": round((time.time() - t0) / max(n_done, 1), 3), "tok_per_s": round(n_tok / max(time.time() - t0, 1e-9), 1),
               "by_source": by_source, "env": env, "library_notes": LIBRARY_NOTES[-20:]}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nextract done: {counts['kept']} of {n_done} kept; rejected {dict(reasons)}; {summary['s_per_row']} s a row", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if do_ship:
        api = HfApi(token=token())
        for attempt in range(4):
            try:
                api.upload_folder(folder_path=out_dir, path_in_repo=f"extractions/{stamp}", repo_id=out_repo, repo_type=repo_type,
                                  commit_message=f"extraction {stamp}: {counts['kept']} rows kept of {n_done}")
                print("  shipped:", f"{out_repo}/extractions/{stamp}", flush=True)
                break
            except Exception as e:
                print(f"  ship failed ({type(e).__name__}); retry", flush=True)
                time.sleep(15 * (attempt + 1))
    summary["path_in_repo"] = f"extractions/{stamp}"
    return summary
