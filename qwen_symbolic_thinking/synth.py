"""Synthetic picture items: the base model, shown two worked examples, thinks through pictures no row was written about; each
chain is audited (the form, the depth, the dataset's banned habits, and the grounding against the picture's own captions) and
the ones that pass become training items under the short form.

    from qwen_symbolic_thinking.synth import generate
    generate("Qwen/Qwen3.5-9B", seeds="hub:synth/seeds_v1.json", n=3500, min_backed=0.2, min_said=0.2)"""
from __future__ import annotations

import gc
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

from . import REPO, __version__, chain
from .hub import ship, token
from .screen import fetch_images, generate as generate_batch, load_inputs, load_model, messages, preflight


def _upload(local: str, path_in_repo: str, message: str, repo: str) -> None:
    from huggingface_hub import HfApi
    for attempt in range(4):
        try:
            HfApi(token=token()).upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo, repo_type="model",
                                             commit_message=message)
            return
        except Exception as e:
            print(f"  upload of {path_in_repo} failed ({type(e).__name__}); retry", flush=True)
            time.sleep(15 * (attempt + 1))


def audit(scores: dict, min_backed: float, min_said: float) -> str | None:
    """Why a chain is rejected, or None when it passes."""
    if not scores["format_ok"]:
        return "format"
    if scores.get("named_frac", 0) != 1:
        return "unnamed step"
    if not scores["in_range"]:
        return "depth"
    if scores["stray_lines"]:
        return "stray lines"
    if scores["gloss_hits"]:
        return "defines a word"
    if scores["remark_hits"]:
        return "remarks on the task"
    if (scores["hint_backed"] or 0) < min_backed:
        return "not backed by the captions"
    if (scores["hint_said"] or 0) < min_said:
        return "says too little of the captions"
    return None


def generate(model_id: str = "Qwen/Qwen3.5-9B", seeds: str = "hub:synth/seeds_v1.json",
             examples: str = "hub:screens/inputs/screen_v1.json", n: int | None = None, batch: int = 64,
             max_new_tokens: int = 320, min_backed: float = 0.2, min_said: float = 0.2, out_dir: str = "/content/synth",
             repo: str = REPO, do_ship: bool = True, tag: str = "synth") -> dict:
    """Generate, audit and ship synthetic picture items; returns the summary (with path_in_repo of the item file)."""
    import torch
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    env = preflight()
    items = load_inputs(seeds)["items"][:n]
    shown = load_inputs(examples).get("examples", {})
    t0 = time.time()
    pictures = fetch_images(items, os.path.join(os.path.dirname(out_dir), "images"), workers=24)
    items = [it for it in items if it["id"] in pictures]
    print(f"synth {stamp}: {len(items)} pictures ready in {time.time() - t0:.0f} s; {model_id}, batch {batch}", flush=True)
    processor, model = load_model(model_id, False)
    kept, rejected, t_gen, n_tok = [], Counter(), time.time(), 0
    records_path = os.path.join(out_dir, "generated.jsonl")
    for start in range(0, len(items), batch):
        group = items[start:start + batch]
        convs = [messages("shown", it, shown, pictures[it["id"]]) for it in group]
        try:
            texts, lengths = generate_batch(processor, model, convs, [pictures[it["id"]] for it in group], max_new_tokens, False, None)
        except Exception as e:
            texts, lengths = [f"<error {type(e).__name__}: {str(e)[:200]}>"] * len(group), [0] * len(group)
        n_tok += sum(lengths)
        with open(records_path, "a", encoding="utf-8") as fh:
            for it, text in zip(group, texts):
                parsed = chain.parse(text)
                scores = chain.score(parsed, it)
                why = audit(scores, min_backed, min_said)
                fh.write(json.dumps({"id": it["id"], "text": text, "scores": scores, "rejected": why}, ensure_ascii=False) + "\n")
                if why:
                    rejected[why] += 1
                    continue
                kept.append(dict(it, thoughts=parsed["thoughts"], wordings={"caption": parsed["wordings"]["caption"]}, questions=[],
                                 forms=["short"],
                                 source={"kind": "synthetic-model", "model": model_id, "form": "shown", "license": it.get("license"),
                                         "hint_said": round(scores["hint_said"] or 0, 3), "hint_backed": round(scores["hint_backed"] or 0, 3),
                                         "spec_recall": scores["spec_recall"]}))
        done = start + len(group)
        spent = time.time() - t_gen
        print(f"  {done}/{len(items)} pictures, {len(kept)} kept ({len(kept) / done:.0%}), {spent:.0f} s spent, "
              f"{spent / done * (len(items) - done):.0f} s left", flush=True)
    summary = {"package": __version__, "model": model_id, "form": "shown", "depth": items[0]["depth"] if items else None,
               "min_backed": min_backed, "min_said": min_said, "generated": len(items), "kept": len(kept),
               "rejected": dict(rejected), "s_per_picture": round((time.time() - t_gen) / max(len(items), 1), 3),
               "tok_per_s": round(n_tok / max(time.time() - t_gen, 1e-9), 1), "env": env}
    data = {"version": 1, **{k: summary[k] for k in ("model", "form", "min_backed", "min_said", "generated", "kept", "rejected")},
            "items": kept}
    item_file = os.path.join(out_dir, f"{tag}.json")
    with open(item_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nsynth done: {len(kept)} of {len(items)} kept; rejected {dict(rejected)}", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    summary["path_in_repo"] = f"synth/{tag}.json"                   # the fixed name a later run's inputs can point at
    summary["path_in_repo_stamped"] = f"synth/{stamp}/{tag}.json"
    if do_ship:
        print("  shipped:", ship(out_dir, f"synth/{stamp}", f"synth {stamp}: {len(kept)} items of {len(items)} ({model_id})", repo), flush=True)
        _upload(item_file, f"synth/{tag}.json", f"synth {tag}: {len(kept)} items (from {stamp})", repo)
    return summary
