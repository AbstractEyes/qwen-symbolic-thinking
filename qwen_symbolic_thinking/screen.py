"""The behaviour screen: can a base model, shown a picture and asked plainly, think it through the way the rows do?

    from qwen_symbolic_thinking.screen import run
    run(models=["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"], inputs="hub:screens/inputs/screen_v1.json")

For every model and prompt form the screen generates a chain for each item, parses and scores it (chain.score), writes one
line per answer to results.jsonl and a summary.json per model, prints the summary table, and ships the folder to the project
repo under screens/<stamp>/. Thinking mode is off; decoding is greedy unless a sampling dict is given."""
from __future__ import annotations

import gc
import io
import json
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone

from . import REPO, __version__, chain, prompts
from .hub import fetch, ship, token

FORMS = ("bare", "shown", "spec")


def preflight() -> dict:
    """What the card and the environment are; printed once before anything loads."""
    import torch
    import transformers
    info = {"package": __version__, "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1) if torch.cuda.is_available() else None,
            "hf_token": bool(token())}
    print(json.dumps(info, indent=1))
    if not info["cuda"]:
        print("NO GPU: set the runtime to a GPU before running the screen.")
    elif "6000" not in (info["gpu"] or ""):
        print("note: the card is not the RTX PRO 6000; the screen runs, the timings are not comparable.")
    return info


def load_inputs(inputs: str) -> dict:
    path = fetch(inputs[4:]) if inputs.startswith("hub:") else inputs
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fetch_image(urls: list, cache_dir: str):
    """The first url that answers, as an RGB PIL image; cached on disk by file name."""
    import requests
    from PIL import Image
    os.makedirs(cache_dir, exist_ok=True)
    for url in urls:
        local = os.path.join(cache_dir, url.rsplit("/", 2)[-2] + "_" + url.rsplit("/", 1)[-1])
        if os.path.exists(local):
            return Image.open(local).convert("RGB")
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException:
            continue
        if r.status_code == 200 and r.content[:3] == b"\xff\xd8\xff":
            with open(local, "wb") as fh:
                fh.write(r.content)
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None


def messages(form: str, it: dict, examples: dict, image=None) -> list:
    if form == "bare":
        text = prompts.bare(it["depth"])
    elif form == "shown":
        text = prompts.shown(it["depth"], examples.get(it["depth"]) or examples.get("medium") or [])
    elif form == "spec":
        text = prompts.spec(it["depth"], it["spec"])
    else:
        raise ValueError(f"unknown form {form!r}")
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    return [{"role": "user", "content": content}]


def load_model(model_id: str, thinking: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return processor, model


def generate(processor, model, convs: list, images: list, max_new_tokens: int, thinking: bool, sampling: dict | None) -> tuple:
    """One batch: the conversations rendered through the chat template, the images beside them; returns (texts, new_tokens)."""
    import torch
    texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False, enable_thinking=thinking) for c in convs]
    kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
    if any(im is not None for im in images):
        kwargs["images"] = [im for im in images if im is not None]
    inputs = processor(**kwargs).to(model.device)
    gen = {"max_new_tokens": max_new_tokens, "do_sample": False,
           "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id}
    if sampling:
        gen.update(sampling, do_sample=True)
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    new = out[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(new, skip_special_tokens=True)
    lengths = [int((row != gen["pad_token_id"]).sum()) for row in new]
    return decoded, lengths


def summarize(results: list) -> dict:
    """Per (model, form): the rates and means the screen reads."""
    groups = defaultdict(list)
    for r in results:
        groups[(r["model"], r["form"])].append(r)
    out = {}
    for (model, form), rs in sorted(groups.items()):
        s = [r["scores"] for r in rs]
        recalls = [x["spec_recall"] for x in s if x["spec_recall"] is not None]
        refs = [r["reference"]["spec_recall"] for r in rs if r["reference"]["spec_recall"] is not None]
        n = len(rs)
        out[f"{model} | {form}"] = {
            "rows": n,
            "format_ok": sum(x["format_ok"] for x in s) / n,
            "in_range": sum(x["in_range"] for x in s) / n,
            "has_caption": sum(x["has_caption"] for x in s) / n,
            "mean_thoughts": sum(x["n_thoughts"] for x in s) / n,
            "spec_recall": (sum(recalls) / len(recalls)) if recalls else None,
            "spec_recall_of_rows": (sum(refs) / len(refs)) if refs else None,
            "rows_with_gloss": sum(1 for x in s if x["gloss_hits"]) / n,
            "rows_with_remark": sum(1 for x in s if x["remark_hits"]) / n,
            "stray_lines": sum(x["stray_lines"] for x in s) / n,
            "step_jaccard": sum(x["step_jaccard"] or 0 for x in s) / n,
            "mean_chars": sum(x["chars"] for x in s) / n,
            "s_per_row": sum(r["gen_s"] for r in rs) / n,
            "tok_per_s": sum(r["new_tokens"] for r in rs) / max(sum(r["gen_s"] for r in rs), 1e-9),
        }
    return out


def table(summary: dict) -> str:
    cols = ["rows", "format_ok", "in_range", "spec_recall", "spec_recall_of_rows", "rows_with_gloss", "rows_with_remark",
            "mean_thoughts", "step_jaccard", "s_per_row", "tok_per_s"]
    lines = ["| model | form | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for key, v in summary.items():
        model, form = key.split(" | ")
        cells = [f"{v[c]:.2f}" if isinstance(v[c], float) else ("-" if v[c] is None else str(v[c])) for c in cols]
        lines.append(f"| {model.split('/')[-1]} | {form} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(models: list, inputs: str = "hub:screens/inputs/screen_v1.json", out_dir: str = "/content/screen",
        forms: tuple = FORMS, n_images: int | None = None, n_specs: int | None = None, batch: int = 4,
        max_new_tokens: int = 700, thinking: bool = False, sampling: dict | None = None, repo: str = REPO,
        do_ship: bool = True) -> dict:
    """The screen over every model and form; results and summaries on disk and on the Hub."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    env = preflight()
    data = load_inputs(inputs)
    items = data["items"]
    image_items = [it for it in items if it["kind"] == "image"][:n_images]
    spec_items = [it for it in items if it["kind"] == "spec"][:n_specs]
    print(f"screen {stamp}: {len(image_items)} picture rows, {len(spec_items)} specification rows, forms {forms}, "
          f"models {models}", flush=True)
    t0 = time.time()
    pictures = {}
    for it in image_items:
        im = fetch_image(it["image_urls"], os.path.join(os.path.dirname(out_dir), "images"))
        if im is None:
            print(f"  no image for {it['id']} ({it['seed_id']}); skipped", flush=True)
        else:
            pictures[it["id"]] = im
    image_items = [it for it in image_items if it["id"] in pictures]
    print(f"  {len(pictures)} images ready in {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(out_dir, "env.json"), "w", encoding="utf-8") as fh:
        json.dump({**env, "models": models, "forms": list(forms), "inputs": inputs, "batch": batch,
                   "max_new_tokens": max_new_tokens, "thinking": thinking, "sampling": sampling,
                   "n_images": len(image_items), "n_specs": len(spec_items)}, fh, indent=1)
    results_path = os.path.join(out_dir, "results.jsonl")
    all_results = []
    for model_id in models:
        print(f"\n== {model_id}", flush=True)
        t_load = time.time()
        processor, model = load_model(model_id, thinking)
        print(f"  loaded in {time.time() - t_load:.0f} s", flush=True)
        results = []
        for form in forms:
            todo = spec_items if form == "spec" else image_items
            jobs = []
            for it in todo:
                im = None if form == "spec" else pictures[it["id"]]
                jobs.append((it, messages(form, it, data.get("examples", {}), im), im))
            t_form, done = time.time(), 0
            for start in range(0, len(jobs), batch):
                group = jobs[start:start + batch]
                t_b = time.time()
                try:
                    texts, lengths = generate(processor, model, [j[1] for j in group], [j[2] for j in group],
                                              max_new_tokens, thinking, sampling)
                except Exception as e:  # one failed batch is recorded, never fatal
                    texts, lengths = [f"<error {type(e).__name__}: {str(e)[:200]}>"] * len(group), [0] * len(group)
                dt = (time.time() - t_b) / len(group)
                for (it, _, _), text, n_new in zip(group, texts, lengths):
                    parsed = chain.parse(text)
                    rec = {"model": model_id, "form": form, "id": it["id"], "depth": it["depth"], "kind": it["kind"],
                           "text": text, "parsed": parsed, "scores": chain.score(parsed, it),
                           "reference": chain.reference(it), "gen_s": dt, "new_tokens": n_new}
                    results.append(rec)
                    with open(results_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += len(group)
                spent = time.time() - t_form
                print(f"  [{form}] {done}/{len(jobs)} rows, {spent:.0f} s spent, {spent / done * (len(jobs) - done):.0f} s left, "
                      f"{dt:.1f} s/row", flush=True)
        summary = summarize(results)
        name = model_id.replace("/", "__")
        with open(os.path.join(out_dir, f"summary_{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        print("\n" + table(summary), flush=True)
        all_results += results
        del model, processor
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        if do_ship:
            url = ship(out_dir, f"screens/{stamp}", f"screen {stamp}: {model_id} ({len(results)} answers)", repo)
            print(f"  shipped: {url}", flush=True)
    summary = summarize(all_results)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(table(summary) + "\n")
    if do_ship:
        ship(out_dir, f"screens/{stamp}", f"screen {stamp}: summary", repo)
    print(f"\nscreen done in {(time.time() - t0) / 60:.1f} min; results in {out_dir}", flush=True)
    return summary
