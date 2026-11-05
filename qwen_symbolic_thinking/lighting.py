"""Lighting read from a photo by two published models: a trial before either feeds a helper line.

ShotVL-7B (Vchitect, a Qwen2.5-VL-7B fine-tune for cinematography, Apache-2.0) answers its benchmark's two lighting questions,
asked open and as a choice over every option the benchmark uses: the lighting type (daylight, artificial light, HMI ...) and the
lighting (soft light, side light, backlight, high contrast ...). CinemaCLIP (OZU-Technology, a CLIP ViT-B/32 with classifier
heads, OpenRAIL) scores its lighting heads: cast (soft / neutral / hard), contrast (low / neutral / high), direction (ambient,
back, front, left, right, top, under; several can hold) and edge light and silhouette likelihoods; it runs from its released ONNX
image encoder with the label schema beside it, so no package of its own is installed.

`trial(images_zip, files, ...)` runs both on the photos named, one row a photo, and ships the folder to a Hub dataset.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone

from . import __version__
from .hub import token

SHOTVL = "Vchitect/ShotVL-7B"
SHOTVL_BASE = "Qwen/Qwen2.5-VL-7B-Instruct"        # the processor to fall back on (ShotVL is its fine-tune)
SYSTEM = "You are a helpful assistant."            # the model card's system message
CINEMACLIP = "OZU-Technology/CinemaCLIP"
ONNX_FILE = "ImageEncoder.onnx"
SCHEMA_FILE = "CinemaNetSchema.json"
# The released ONNX encoder gives 101 probabilities (23 heads), which the package's own schema (1.0.0) names; the schema beside it
# on the Hub (2.0.0) names 315 (28 heads). The package's schema, at a fixed commit of its repository:
SCHEMA_V1 = ("https://raw.githubusercontent.com/Synopsis/cinemaclip/8680411620aef92bb661306ef2957da5eee8bede/"
             "cinemaclip/assets/CinemaNetSchema.json")
SIDE = 256                                         # the encoder's input: 256 x 256, bicubic, values in [0, 1], no normalization
HEADS = ("shot.lighting.cast", "shot.lighting.contrast", "shot.lighting.direction", "shot.lighting.edge",
         "shot.lighting.silhouette")

# The benchmark's own questions and the single options its multiple-choice items draw from (ShotBench test.tsv, categories
# 'lighting type' and 'lighting'; an item's option can join two or three of these).
TYPE_QUESTION = "What's the lighting type of this shot?"
LIGHT_QUESTION = "What's the lighting of this shot?"
TYPES = ("Daylight", "Sunny", "Overcast", "Artificial light", "Practical light", "Fluorescent", "HMI", "Tungsten", "LED",
         "Mixed light", "Firelight", "Moonlight")
LIGHTS = ("Soft light", "Hard light", "Side light", "Top light", "Underlight", "Backlight", "Edge light", "Silhouette",
          "High contrast", "Low contrast")
ASKS = {"type_open": (TYPE_QUESTION, None), "light_open": (LIGHT_QUESTION, None), "type_choice": (TYPE_QUESTION, TYPES),
        "light_choice": (LIGHT_QUESTION, LIGHTS)}
LETTERS = "ABCDEFGHIJKLMNOP"


def choice_text(question: str, options) -> str:
    """A multiple-choice ask in the model card's form: the question, 'Options:' and one lettered option a line, then the
    instruction to pick."""
    lines = [question, "Options:"] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return "\n".join(lines) + "\nPlease select the most likely answer from the options above.\n"


def picked(answer: str, options) -> list:
    """The options an answer picks: by letter ('B', 'B. Hard light', 'A, C') or by name, in the options' order."""
    text = (answer or "").strip()
    letters = re.findall(r"(?<![A-Za-z])([A-P])(?:[.):]|\s*$|\s*,)", text)
    out = [options[LETTERS.index(x)] for x in letters if LETTERS.index(x) < len(options)]
    for o in options:
        if o not in out and re.search(rf"\b{re.escape(o.lower())}\b", text.lower()):
            out.append(o)
    return [o for o in options if o in out]


def shotvl_load(model_id: str = SHOTVL, dtype: str = "bfloat16") -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="cuda",
                                                               attn_implementation="sdpa").eval()
    try:
        processor, source = AutoProcessor.from_pretrained(model_id), model_id
    except Exception as e:  # noqa: BLE001
        print(f"  {model_id}'s processor did not load ({type(e).__name__}: {str(e)[:120]}); using {SHOTVL_BASE}'s")
        processor, source = AutoProcessor.from_pretrained(SHOTVL_BASE), SHOTVL_BASE
    return {"model": model, "processor": processor, "processor_from": source, "id": model_id}


def shotvl_ask(bundle: dict, image, ask: str, max_new_tokens: int = 64) -> str:
    """One photo and one ask, greedy, with the card's system message."""
    import torch
    processor, model = bundle["processor"], bundle["model"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": ask}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def pick_schema(widths, schemas):
    """The first schema whose label count is one of the encoder's output widths, else None."""
    return next((s for s in schemas if s and len(s.get("probabilities_labels") or []) in set(widths)), None)


def cinemaclip_load(repo: str = CINEMACLIP) -> dict:
    """The ONNX image encoder and the schema that names its outputs: the Hub's own when its labels fit the encoder, else the
    package's schema (SCHEMA_V1)."""
    import urllib.request
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    session = ort.InferenceSession(hf_hub_download(repo, ONNX_FILE), providers=["CPUExecutionProvider"])
    widths = [o.shape[-1] for o in session.get_outputs() if isinstance(o.shape[-1], int)]
    schema = pick_schema(widths, [json.load(open(hf_hub_download(repo, SCHEMA_FILE), encoding="utf-8"))])
    if schema is None:
        schema = pick_schema(widths, [json.loads(urllib.request.urlopen(SCHEMA_V1, timeout=60).read())])
    if schema is None:
        raise ValueError(f"no schema names the encoder's outputs (widths {widths})")
    return {"session": session, "schema": schema, "labels": schema["probabilities_labels"], "heads": schema["categories_meta"],
            "schema_version": schema.get("version")}


def cinemaclip_scores(bundle: dict, image) -> dict:
    """{head: {label: probability}} for the lighting heads, from one photo."""
    import numpy as np
    from PIL import Image
    x = np.asarray(image.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    outputs = bundle["session"].run(None, {bundle["session"].get_inputs()[0].name: x})
    found = [o for o in outputs if o.shape[-1] == len(bundle["labels"])]
    if not found:
        raise ValueError(f"no encoder output has the schema's {len(bundle['labels'])} labels: shapes {[o.shape for o in outputs]}")
    return lighting_heads(dict(zip(bundle["labels"], found[0][0].tolist())), bundle["heads"])


def lighting_heads(flat: dict, heads: dict) -> dict:
    """The lighting heads' scores out of the flat label -> probability map, with each head's pick: the top label of a softmax
    head, every label at or over the suggested threshold of a sigmoid head."""
    out = {}
    for head in HEADS:
        meta = heads[head]
        scores = {label: round(float(flat[f"{head}.{label}"]), 4) for label in meta["labels"]}
        if meta.get("activation") == "softmax":
            pick = [max(scores, key=scores.get)]
        else:
            pick = [label for label, p in scores.items() if p >= float(meta.get("suggested_thresh") or 0.5)]
        out[head.rsplit(".", 1)[1]] = {"scores": scores, "pick": pick}
    return out


def _versions() -> dict:
    from importlib import metadata
    out = {}
    for name in ("torch", "transformers", "onnxruntime"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        out["gpu"] = None
    return out


def trial(images_zip: str, files: list, out_repo: str | None = None, path_in_repo: str = "raw", run_name: str = "lighting_trial",
          views: dict | None = None, ship: bool = True, max_new_tokens: int = 64) -> dict:
    """Both models on the photos named (members `images/<file>` or `<file>` of the zip): one row a photo with ShotVL's four answers (the open
    ones as written, the choices as written and as picked options) and CinemaCLIP's lighting heads; rows.jsonl + summary.json in
    a stamped folder, shipped to `out_repo` (a dataset) under `path_in_repo` at the end."""
    from PIL import Image
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + run_name
    out_dir = os.path.join("/content" if os.path.isdir("/content") else ".", "lighting_out", stamp)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    shotvl = shotvl_load()
    clip = cinemaclip_load()
    load_s = time.time() - t0
    rows, seconds = [], {"shotvl": 0.0, "cinemaclip": 0.0}
    with zipfile.ZipFile(images_zip) as zf:
        members = set(zf.namelist())
        for i, file in enumerate(files):
            member = "images/" + file if "images/" + file in members else file      # the official zip, or the notebook's own
            image = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            t = time.time()
            answers = {}
            for key, (question, options) in ASKS.items():
                text = shotvl_ask(shotvl, image, choice_text(question, options) if options else question, max_new_tokens)
                answers[key] = {"text": text, "picked": picked(text, options)} if options else {"text": text}
            seconds["shotvl"] += time.time() - t
            t = time.time()
            heads = cinemaclip_scores(clip, image)
            seconds["cinemaclip"] += time.time() - t
            rows.append({"file": file, "view": (views or {}).get(file), "shotvl": answers, "cinemaclip": heads})
            if i % 20 == 0:
                print(f"  {i + 1}/{len(files)}  {file}: type {answers['type_choice']['picked']}, light "
                      f"{answers['light_choice']['picked']}, cast {heads['cast']['pick']}, direction {heads['direction']['pick']}")
    with open(os.path.join(out_dir, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"package": __version__, "made": datetime.now(timezone.utc).isoformat(timespec="seconds"), "photos": len(rows),
               **_versions(),
               "shotvl": SHOTVL, "processor_from": shotvl["processor_from"], "cinemaclip": f"{CINEMACLIP}/{ONNX_FILE}",
               "cinemaclip_schema": clip.get("schema_version"),
               "asks": {k: (choice_text(q, o) if o else q) for k, (q, o) in ASKS.items()}, "max_new_tokens": max_new_tokens,
               "load_s": round(load_s, 1), "s_per_photo": {k: round(v / max(1, len(rows)), 3) for k, v in seconds.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    if ship and out_repo:
        from huggingface_hub import HfApi
        HfApi(token=token()).upload_folder(folder_path=out_dir, path_in_repo=f"{path_in_repo}/{stamp}", repo_id=out_repo,
                                           repo_type="dataset", commit_message=f"{run_name}: {len(rows)} photos ({stamp})")
        print(f"shipped {out_repo}/{path_in_repo}/{stamp}")
    return {"out_dir": out_dir, "stamp": stamp, **summary}
