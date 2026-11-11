"""Pictures pinned for training: copies kept in a Hugging Face storage bucket, packed into parquet shards in the project repo with
a manifest of hf_parquet locators, so a training set reads the same bytes on every later run.

A row-writing system that shows its writers pictures may keep its own copies in a storage bucket and list them in picture-list
files: `*.jsonl` under one folder of the bucket, one {"image": "<dir>/<file>", "urls": [<link>, ...]} a line, where a link is an
http(s) address or "store:<path>", a file under the same bucket folder. A bucket is mutable (its files can be replaced or
deleted), so pack_staged copies the pictures a training set needs into the project repo once:

    from qwen_symbolic_thinking.pack import pack_staged
    packed = pack_staged(["ffhq:65013"], "owner/bucket/folder", dirs={"ffhq": ["ffhq"]}, out_dir="/content/pack", tag="rows_nc")

The pictures keep their bytes as stored (the copies the writers were shown). manifest.jsonl, shipped beside the shards, names each
picture by (source, source_id), the two halves of its seed id, so rows.build_api_rows finds it by seed id."""
from __future__ import annotations

import io
import json
import os
import tempfile
from collections import Counter

from . import REPO
from .hub import ship, token

LISTS = "control/pictures/"       # where the picture-list files sit under the bucket folder
SHARD = 500                       # pictures a shard
BATCH = 200                       # bucket files a download call


def parse_bucket(value: str) -> tuple:
    """'owner/name' or 'owner/name/folder[/more]' -> (bucket id, folder, '' for none)."""
    parts = [p for p in str(value).split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"a bucket is named owner/name or owner/name/folder, not {value!r}")
    return "/".join(parts[:2]), "/".join(parts[2:])


def _client(api=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    if not hasattr(api, "download_bucket_files"):
        raise RuntimeError("this huggingface_hub has no storage-bucket calls: install a newer huggingface_hub")
    return api


def read_lists(api, bucket_id: str, folder: str, lists: str = LISTS, workdir: str | None = None) -> dict:
    """{(dir, file stem): [links]} from every picture-list file under <folder>/<lists> of the bucket (a later line for the same
    picture wins). dir is the image path's first part, so a list made for another purpose (copies under bench/...) never
    answers for a picture of a source's own folder."""
    prefix = f"{folder}/{lists}" if folder else lists
    files = sorted(it.path for it in api.list_bucket_tree(bucket_id, prefix=prefix, recursive=True)
                   if getattr(it, "type", "file") == "file" and it.path.endswith(".jsonl"))
    workdir = workdir or tempfile.mkdtemp(prefix="lists_")
    os.makedirs(workdir, exist_ok=True)
    pairs = [(p, os.path.join(workdir, f"{i:05d}.jsonl")) for i, p in enumerate(files)]
    if pairs:
        api.download_bucket_files(bucket_id, pairs, raise_on_missing_files=True)
    out = {}
    for _, local in pairs:
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                image = str((d or {}).get("image") or "") if isinstance(d, dict) else ""
                if "/" not in image or not d.get("urls"):
                    continue
                parts = image.split("/")
                out[(parts[0], os.path.splitext(parts[-1])[0])] = [str(u) for u in d["urls"]]
    return out


def find(seed_id: str, lists: dict, dirs: dict | None = None) -> list | None:
    """The links of a seed id's picture: '<source>:<key>' looked up as (dir, key) for each dir the source maps to in dirs (the
    source's own name when dirs names none for it); None when no list has it."""
    source, _, key = seed_id.partition(":")
    for d in (dirs or {}).get(source) or [source]:
        if (d, key) in lists:
            return lists[(d, key)]
    return None


def first_link(links: list) -> str | None:
    """The first link this module can read: a store: path or an http(s) address."""
    for link in links or []:
        if link.startswith("store:") or link.lower().startswith(("http://", "https://")):
            return link
    return None


def _picture(data: bytes) -> tuple:
    """(width, height) of bytes that decode as a whole picture; raises when they do not."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return im.size


def pack_staged(seed_ids, bucket: str, dirs: dict | None = None, out_dir: str = "/content/pack", tag: str = "pack",
                repo: str = REPO, path_in_repo: str | None = None, lists: str = LISTS, shard: int = SHARD,
                do_ship: bool = True, api=None, timeout: float = 30.0) -> dict:
    """Copy the pictures of seed_ids, found through the bucket's picture lists, into parquet shards (columns id, source,
    source_id, image = the bytes as stored, width, height) with manifest.jsonl beside them (one hf_parquet locator a picture,
    pointing at <repo>/<path_in_repo>, default pictures/<tag>), and ship the folder there. Returns the counts, the seeds that
    were not packed by reason (in no list, no readable link, missing from the bucket, an http link that failed, bytes that do
    not open as a picture) and the local path of the manifest, for rows.build_api_rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    api = _client(api)
    bucket_id, folder = parse_bucket(bucket)
    path_in_repo = (path_in_repo or f"pictures/{tag}").strip("/")
    stage, packed_dir = os.path.join(out_dir, "_files"), os.path.join(out_dir, "packed")
    os.makedirs(stage, exist_ok=True)
    os.makedirs(packed_dir, exist_ok=True)
    table = read_lists(api, bucket_id, folder, lists, os.path.join(out_dir, "_lists"))
    seeds = sorted({str(s) for s in seed_ids if s})
    not_packed: dict = {}
    store, web = [], []
    for sid in seeds:
        links = find(sid, table, dirs)
        link = first_link(links) if links else None
        if not links:
            not_packed.setdefault("in no picture list", []).append(sid)
        elif not link:
            not_packed.setdefault("no readable link", []).append(sid)
        elif link.startswith("store:"):
            store.append((sid, f"{folder}/{link[6:]}" if folder else link[6:]))
        else:
            web.append((sid, link))
    raw = {}
    for start in range(0, len(store), BATCH):
        chunk = store[start:start + BATCH]
        pairs = [(remote, os.path.join(stage, f"{start + i:06d}.bin")) for i, (_, remote) in enumerate(chunk)]
        api.download_bucket_files(bucket_id, pairs, raise_on_missing_files=False)
        for (sid, _), (_, local) in zip(chunk, pairs):
            if os.path.exists(local):
                with open(local, "rb") as fh:
                    raw[sid] = fh.read()
            else:
                not_packed.setdefault("missing from the bucket", []).append(sid)
    if web:
        import requests
        for sid, url in web:
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200 and r.content:
                    raw[sid] = r.content
                    continue
                why = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                why = type(e).__name__
            not_packed.setdefault(f"link failed ({why})", []).append(sid)
    good = []
    for sid in sorted(raw):
        try:
            w, h = _picture(raw[sid])
        except Exception:
            not_packed.setdefault("not a picture", []).append(sid)
            continue
        good.append((sid, raw[sid], w, h))
    manifest, n_bytes = [], 0
    for start in range(0, len(good), shard):
        part = good[start:start + shard]
        name = f"shard_{start // shard:04d}.parquet"
        tbl = pa.table({"id": [p[0] for p in part],
                        "source": [p[0].split(":", 1)[0] for p in part],
                        "source_id": [p[0].split(":", 1)[1] for p in part],
                        "image": pa.array([p[1] for p in part], type=pa.binary()),
                        "width": [p[2] for p in part], "height": [p[3] for p in part]})
        pq.write_table(tbl, os.path.join(packed_dir, name), row_group_size=100)
        for j, (sid, data, w, h) in enumerate(part):
            n_bytes += len(data)
            source, key = sid.split(":", 1)
            manifest.append({"sample_id": f"{tag}-{start + j:06d}", "split": "packed", "source": source, "source_id": key,
                             "width": w, "height": h,
                             "image": {"kind": "hf_parquet", "repo": repo, "repo_type": "model", "path": f"{path_in_repo}/{name}",
                                       "row_index": j, "column": "image"}})
    manifest_path = os.path.join(packed_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    summary = {"tag": tag, "bucket": bucket, "asked": len(seeds), "packed": len(good), "shards": -(-len(good) // shard) if good else 0,
               "bytes": n_bytes, "by_source": dict(Counter(sid.split(":", 1)[0] for sid, *_ in good)),
               "not_packed": {k: len(v) for k, v in not_packed.items()},
               "not_packed_examples": {k: v[:5] for k, v in not_packed.items()},
               "path_in_repo": path_in_repo, "manifest": manifest_path, "repo": repo}
    with open(os.path.join(packed_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in summary.items() if k != "manifest"}, fh, indent=1)
    print(f"pack {tag}: {len(good)} of {len(seeds)} pictures in {summary['shards']} shards ({n_bytes / 1e6:.1f} MB); "
          f"not packed {summary['not_packed']}", flush=True)
    if do_ship and good:
        summary["shipped"] = ship(packed_dir, path_in_repo, f"pack {tag}: {len(good)} pictures", repo)
    return summary
