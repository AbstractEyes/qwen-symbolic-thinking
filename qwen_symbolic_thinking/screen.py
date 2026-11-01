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
import logging
import os
import platform
import re
import time
from collections import defaultdict
from datetime import datetime, timezone

from . import REPO, __version__, chain, prompts
from .hub import fetch, ship, token
from .template import stop_ids

FORMS = ("bare", "shown", "spec")                        # the base model's screen
TRAINED_FORMS = ("full", "short", "spec_full")           # the trained model's screen: the whole row, the chain, a specification
SPEC_FORMS = ("spec", "spec_short", "spec_full")         # the forms that take a specification instead of a picture
LIBRARY_NOTES = []                                # what the transformers logger says (the kernel fallback warning lands here)


class _Notes(logging.Handler):
    def emit(self, record):
        LIBRARY_NOTES.append(record.getMessage())


def _listen():
    """Attach once to the transformers logger, so a 'falling back to its reference PyTorch implementation' note is kept."""
    lg = logging.getLogger("transformers")
    if not any(isinstance(h, _Notes) for h in lg.handlers):
        lg.addHandler(_Notes())


def fast_path_note() -> str:
    """What the run has learnt about the linear-attention kernels so far."""
    fallbacks = [m for m in LIBRARY_NOTES if "falling back" in m]
    return ("TORCH FALLBACK: " + fallbacks[0][:200]) if fallbacks else "no fallback warning (the kernels serve the linear attention)"


def preflight() -> dict:
    """What the card and the environment are; printed once before anything loads."""
    import importlib.util
    import torch
    import transformers
    info = {"package": __version__, "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "qwen3_5": importlib.util.find_spec("transformers.models.qwen3_5") is not None, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1) if torch.cuda.is_available() else None,
            "hf_token": bool(token())}
    print(json.dumps(info, indent=1))
    if not info["cuda"]:
        print("NO GPU: set the runtime to a GPU before running the screen.")
    elif "6000" not in (info["gpu"] or ""):
        print("note: the card is not the RTX PRO 6000; the screen runs, the timings are not comparable.")
    if not info["qwen3_5"]:
        print("this transformers has no Qwen3.5 model code: install transformers 5.x and restart the session.")
    return info


def vlm_class():
    """The auto class for a vision-language model: the model card's name on transformers 5, the older name before it."""
    try:
        from transformers import AutoModelForMultimodalLM as cls
    except ImportError:
        from transformers import AutoModelForImageTextToText as cls
    return cls


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


def fetch_images(items: list, cache_dir: str, workers: int = 16) -> dict:
    """{item id: image} for the picture items, fetched in parallel; the items whose image never answers are left out."""
    from concurrent.futures import ThreadPoolExecutor
    picture_items = [it for it in items if it["kind"] == "image"]
    with ThreadPoolExecutor(workers) as pool:
        images = list(pool.map(lambda it: fetch_image(it["image_urls"], cache_dir), picture_items))
    out = {it["id"]: im for it, im in zip(picture_items, images) if im is not None}
    for it, im in zip(picture_items, images):
        if im is None:
            print(f"  no image for {it['id']} ({it['seed_id']}); skipped", flush=True)
    return out


def messages(form: str, it: dict, examples: dict, image=None, system: str | None = None) -> list:
    """The conversation of one screen row: the user turn (the picture, when the form takes one, and the form's text), after a
    system turn when a system prompt is given."""
    if form == "bare":
        text = prompts.bare(it["depth"])
    elif form == "shown":
        text = prompts.shown(it["depth"], examples.get(it["depth"]) or examples.get("medium") or [])
    elif form == "spec":
        text = prompts.spec(it["depth"], it["spec"])
    elif form == "short":
        text = prompts.short(it["depth"])
    elif form == "spec_short":
        text = prompts.spec_short(it["depth"], it["spec"])
    elif form == "full":
        text = prompts.full(it["depth"])
    elif form == "spec_full":
        text = prompts.spec_full(it["depth"], it["spec"])
    else:
        raise ValueError(f"unknown form {form!r}")
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]


def load_model(model_id: str, thinking: bool, adapter: str | None = None, adapter_subfolder: str | None = None):
    """The processor and the model on the card; with an adapter (a local folder or a Hub repo, with a subfolder), the LoRA
    weights are merged into the model. A 'hub:<path>' model id is a full fine-tune shipped under the project repo."""
    import torch
    from transformers import AutoProcessor
    from .hub import resolve_model
    _listen()
    model_id = resolve_model(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = vlm_class().from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        kwargs = {"subfolder": adapter_subfolder} if adapter_subfolder else {}
        model = PeftModel.from_pretrained(model, adapter, token=token(), **kwargs).merge_and_unload()
    model.eval()
    return processor, model


def encode(processor, texts: list, images: list | None = None, **kwargs):
    """The processor over a batch of texts with at most one picture each (images aligned with texts, None where a text has none).
    Qwen3.5's processor takes the pictures as one flat list, matched to the placeholders in order; Gemma 4's takes one list of
    pictures per text and reads a flat list as a single text with several pictures ("Received inconsistently sized batches of
    images (1) and text (4)"): the flat form is tried first, the per-text form on that error."""
    images = list(images or [])
    flat = [im for im in images if im is not None]
    kwargs = dict(text=texts, return_tensors="pt", **kwargs)
    if not flat:
        return processor(**kwargs)
    try:
        return processor(images=flat, **kwargs)
    except ValueError as e:
        if "inconsistently sized" not in str(e):
            raise
        assert len(images) == len(texts), "one picture slot per text (None where a text has no picture)"
        return processor(images=[[im] if im is not None else [] for im in images], **kwargs)


_CHANNEL = re.compile(r"<\|channel>thought\n?(.*?)(?:<channel\|>|$)", re.S)   # Gemma 4's thought channel


def special_tokens(tokenizer) -> list:
    out = list(getattr(tokenizer, "all_special_tokens", None) or [])
    for t in (getattr(tokenizer, "added_tokens_decoder", None) or {}).values():
        if getattr(t, "special", False) and str(t) not in out:
            out.append(str(t))
    return out


def plain(text: str, tokenizer) -> str:
    """Generated text with its special tokens removed and the model's thinking kept as <think>...</think>, so chain.parse strips
    and counts it the same for every family: Gemma 4 thinks inside a channel of special tokens (which skip_special_tokens dropped,
    leaving the thought inline with the answer); Qwen3.5 thinks inside <think> tags, which are ordinary tokens."""
    text = _CHANNEL.sub(lambda m: f"<think>{m.group(1).strip()}</think>", text)
    for tok in special_tokens(tokenizer):
        text = text.replace(tok, "")
    return text


def generate(processor, model, convs: list, images: list, max_new_tokens: int, thinking: bool, sampling: dict | None) -> tuple:
    """One batch: the conversations rendered through the chat template, the images beside them (one or none per conversation);
    returns (texts, new_tokens). With thinking on, the text keeps the thinking as <think>...</think> for the parse."""
    import torch
    try:
        texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False, enable_thinking=thinking) for c in convs]
    except TypeError:  # a processor that takes no template switch; the parse strips any thinking and the summary counts it
        texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
    inputs = encode(processor, texts, images, padding=True).to(model.device)
    pad = processor.tokenizer.pad_token_id   # Gemma's pad id is 0: a falsy id is still an id
    gen = {"max_new_tokens": max_new_tokens, "do_sample": False, "eos_token_id": stop_ids(processor, model),
           "pad_token_id": pad if pad is not None else processor.tokenizer.eos_token_id}
    if sampling:
        gen.update(sampling, do_sample=True)
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    new = out[:, inputs["input_ids"].shape[1]:]
    if thinking:
        decoded = [plain(t, processor.tokenizer) for t in processor.batch_decode(new, skip_special_tokens=False)]
    else:
        decoded = processor.batch_decode(new, skip_special_tokens=True)
    lengths = [int((row != gen["pad_token_id"]).sum()) for row in new]
    return decoded, lengths


def _mean(values: list):
    xs = [v for v in values if v is not None]
    return (sum(xs) / len(xs)) if xs else None


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
            "named_frac": (sum(x["named_frac"] for x in s if x.get("named_frac") is not None)
                           / max(sum(1 for x in s if x.get("named_frac") is not None), 1)),
            "wordings_ok": sum(bool(x.get("wordings_ok")) for x in s) / n,
            "questions_ok": sum(bool(x.get("questions_in_range")) for x in s) / n,
            "mean_questions": sum(x.get("questions_n", 0) for x in s) / n,
            "questions_formed": _mean([(x["questions_formed"] / x["questions_n"]) for x in s if x.get("questions_n")]),
            "full_ok": sum(bool(x.get("full_ok")) for x in s) / n,
            "hint_said": _mean([x.get("hint_said") for x in s]),
            "hint_backed": _mean([x.get("hint_backed") for x in s]),
            "spec_recall": (sum(recalls) / len(recalls)) if recalls else None,
            "spec_recall_of_rows": (sum(refs) / len(refs)) if refs else None,
            "rows_with_gloss": sum(1 for x in s if x["gloss_hits"]) / n,
            "rows_with_remark": sum(1 for x in s if x["remark_hits"]) / n,
            "stray_lines": sum(x["stray_lines"] for x in s) / n,
            "step_jaccard": sum(x["step_jaccard"] or 0 for x in s) / n,
            "mean_chars": sum(x["chars"] for x in s) / n,
            "thinking_chars": sum(x["thinking_chars"] for x in s) / n,
            "s_per_row": sum(r["gen_s"] for r in rs) / n,
            "tok_per_s": sum(r["new_tokens"] for r in rs) / max(sum(r["gen_s"] for r in rs), 1e-9),
        }
    return out


def table(summary: dict) -> str:
    cols = ["rows", "format_ok", "in_range", "mean_thoughts", "wordings_ok", "questions_ok", "full_ok", "spec_recall", "hint_said",
            "hint_backed", "rows_with_gloss", "rows_with_remark", "s_per_row", "tok_per_s"]
    lines = ["| model | form | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for key, v in summary.items():
        model, form = key.split(" | ")
        cells = [f"{v[c]:.2f}" if isinstance(v[c], float) else ("-" if v[c] is None else str(v[c])) for c in cols]
        lines.append(f"| {model.split('/')[-1]} | {form} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(models: list, inputs: str = "hub:screens/inputs/screen_v1.json", out_dir: str = "/content/screen",
        forms: tuple = FORMS, n_images: int | None = None, n_specs: int | None = None, batch: int = 4,
        max_new_tokens: int = 700, thinking: bool = False, sampling: dict | None = None, repo: str = REPO,
        do_ship: bool = True, adapter: str | None = None, adapter_subfolder: str | None = None, tag: str = "",
        system: str | None = None) -> dict:
    """The screen over every model and form; results and summaries on disk and on the Hub (screens/<stamp>[_<tag>]/). With
    system (see prompts.system) every row's conversation opens with that system turn."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    env = preflight()
    system_text = prompts.system(system)
    data = load_inputs(inputs)
    items = data["items"]
    image_items = [it for it in items if it["kind"] == "image"][:n_images]
    spec_items = [it for it in items if it["kind"] == "spec"][:n_specs]
    print(f"screen {stamp}: {len(image_items)} picture rows, {len(spec_items)} specification rows, forms {forms}, "
          f"models {models}", flush=True)
    t0 = time.time()
    pictures = fetch_images(image_items, os.path.join(os.path.dirname(out_dir), "images"))
    image_items = [it for it in image_items if it["id"] in pictures]
    print(f"  {len(pictures)} images ready in {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(out_dir, "env.json"), "w", encoding="utf-8") as fh:
        json.dump({**env, "models": models, "forms": list(forms), "inputs": inputs, "batch": batch,
                   "max_new_tokens": max_new_tokens, "thinking": thinking, "sampling": sampling,
                   "adapter": adapter, "adapter_subfolder": adapter_subfolder, "system": system,
                   "n_images": len(image_items), "n_specs": len(spec_items)}, fh, indent=1)
    results_path = os.path.join(out_dir, "results.jsonl")
    all_results = []
    for model_id in models:
        print(f"\n== {model_id}" + (f" + {adapter}/{adapter_subfolder or ''}" if adapter else ""), flush=True)
        t_load = time.time()
        processor, model = load_model(model_id, thinking, adapter, adapter_subfolder)
        print(f"  loaded in {time.time() - t_load:.0f} s", flush=True)
        if system_text:
            print(f"  system prompt {system!r}: {len(processor.tokenizer(system_text)['input_ids'])} tokens before every row", flush=True)
        results = []
        for form in forms:
            todo = spec_items if form in SPEC_FORMS else image_items
            jobs = []
            for it in todo:
                im = None if form in SPEC_FORMS else pictures[it["id"]]
                jobs.append((it, messages(form, it, data.get("examples", {}), im, system_text), im))
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
                if start == 0 and not thinking and any(chain.parse(t)["thinking"] for t in texts):
                    print("  NOTE: the model is thinking although thinking is off; the parse strips it, the token budget pays for it",
                          flush=True)
                if start == 0 and form == forms[0]:
                    print("  linear attention:", fast_path_note(), flush=True)
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
        fh.write(table(summary) + "\n\nlinear attention: " + fast_path_note() + "\n")
    with open(os.path.join(out_dir, "library_notes.json"), "w", encoding="utf-8") as fh:
        json.dump(LIBRARY_NOTES[-50:], fh, indent=1)
    if do_ship:
        ship(out_dir, f"screens/{stamp}", f"screen {stamp}: summary", repo)
    print(f"\nscreen done in {(time.time() - t0) / 60:.1f} min; results in {out_dir}", flush=True)
    return summary


def rescore(results_path: str, inputs: str | None = None) -> dict:
    """The summary of a shipped results.jsonl under the current chain reader (the texts are kept, so a better reader re-reads
    them); with the screen set given, the rows are scored against it, otherwise against the reference kept in each record."""
    items = {}
    if inputs:
        items = {it["id"]: it for it in load_inputs(inputs)["items"]}
    out = []
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            row = items.get(r["id"]) or {"depth": r["depth"], "spec": {}, "thoughts": []}
            parsed = chain.parse(r["text"])
            r["parsed"], r["scores"] = parsed, chain.score(parsed, row)
            if row.get("spec"):
                r["reference"] = chain.reference(row)
            out.append(r)
    summary = summarize(out)
    print(table(summary))
    return summary


def throughput(model_id: str, inputs: str = "hub:screens/inputs/screen_v1.json", batches: tuple = (4, 8, 16, 32, 64),
               n_rows: int = 64, form: str = "shown", max_new_tokens: int = 700, thinking: bool = False,
               out_dir: str = "/content/screen", repo: str = REPO, do_ship: bool = True, tag: str = "throughput",
               adapter: str | None = None, adapter_subfolder: str | None = None) -> list:
    """Rows an hour against the batch size: the same rows generated at each batch size, the peak memory read each time; an
    out-of-memory batch is recorded and ends the sweep. With an adapter (a Hub repo and subfolder, or a local folder) the
    trained model is measured. Results ship under screens/<stamp>_<tag>/."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # read when torch loads (after a restart)
    import torch
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    env = preflight()
    data = load_inputs(inputs)
    kind = "spec" if form in SPEC_FORMS else "image"
    items = [it for it in data["items"] if it["kind"] == kind][:n_rows]
    pictures = {}
    if kind == "image":
        pictures = fetch_images(items, os.path.join(os.path.dirname(out_dir), "images"))
        items = [it for it in items if it["id"] in pictures]
    jobs = [(it, messages(form, it, data.get("examples", {}), pictures.get(it["id"])), pictures.get(it["id"])) for it in items]
    print(f"throughput {stamp}: {model_id}" + (f" + {adapter}/{adapter_subfolder or ''}" if adapter else "")
          + f", {len(jobs)} rows, form {form}, batches {batches}", flush=True)
    processor, model = load_model(model_id, thinking, adapter, adapter_subfolder)
    generate(processor, model, [j[1] for j in jobs[:4]], [j[2] for j in jobs[:4]], 64, thinking, None)   # warm-up
    print("  linear attention:", fast_path_note(), flush=True)
    results = []
    for b in batches:
        b = min(b, len(jobs))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0, n_tok, err = time.time(), 0, None
        try:
            for start in range(0, len(jobs), b):
                group = jobs[start:start + b]
                _, lengths = generate(processor, model, [j[1] for j in group], [j[2] for j in group], max_new_tokens, thinking, None)
                n_tok += sum(lengths)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            err = "out of memory"
            torch.cuda.empty_cache()
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"
        dt = time.time() - t0
        rec = {"model": model_id, "adapter": adapter, "adapter_subfolder": adapter_subfolder, "form": form, "batch": b,
               "rows": len(jobs), "ok": err is None, "error": err,
               "s_total": round(dt, 1), "s_per_row": round(dt / len(jobs), 3), "rows_per_hour": round(3600 * len(jobs) / dt),
               "tok_per_s": round(n_tok / dt, 1), "new_tokens": n_tok, "peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1),
               "max_new_tokens": max_new_tokens}
        results.append(rec)
        print(f"  batch {b}: {rec['rows_per_hour']} rows/h, {rec['s_per_row']} s/row, {rec['tok_per_s']} tok/s, peak {rec['peak_gb']} GB"
              + (f", {err}" if err else ""), flush=True)
        if err:
            break
    lines = ["| model | form | batch | rows/h | s/row | tok/s | peak GB | note |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model'].split('/')[-1]} | {r['form']} | {r['batch']} | {r['rows_per_hour']} | {r['s_per_row']} | {r['tok_per_s']} | "
              f"{r['peak_gb']} | {r['error'] or ''} |" for r in results]
    md = "\n".join(lines) + "\n\nlinear attention: " + fast_path_note() + "\n"
    with open(os.path.join(out_dir, "throughput.json"), "w", encoding="utf-8") as fh:
        json.dump({"env": env, "results": results, "library_notes": LIBRARY_NOTES[-50:]}, fh, indent=1)
    with open(os.path.join(out_dir, "throughput.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md, flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if do_ship:
        print("  shipped:", ship(out_dir, f"screens/{stamp}", f"throughput {stamp}: {model_id}", repo), flush=True)
    return results
