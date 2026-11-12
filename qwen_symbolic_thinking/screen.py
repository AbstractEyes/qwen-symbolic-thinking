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

from . import REPO, __version__, chain, family, prompts
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


def load_vlm(src: str, classes: list | None = None, trust_remote_code: bool = False, **kwargs):
    """The model through the multimodal auto class, or through the image-text-to-text class for a configuration the first does
    not map (which families each class maps differs between transformers versions); with trust_remote_code, the repository's
    own classes, the causal-LM class last (Phi-4-reasoning-vision registers only that one). Any other error surfaces as it is."""
    if classes is None:
        classes = [vlm_class()]
        try:
            from transformers import AutoModelForImageTextToText
            if AutoModelForImageTextToText not in classes:
                classes.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        if trust_remote_code:
            from transformers import AutoModelForCausalLM
            classes.append(AutoModelForCausalLM)
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    first = None
    for cls in classes:
        try:
            return cls.from_pretrained(src, **kwargs)
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            first = first or e
    raise first


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


def messages(form: str, it: dict, examples: dict, image=None, system: str | None = None, mark: str | None = None,
             keys: str = "record") -> list:
    """The conversation of one screen row: the user turn (the picture, when the form takes one, and the form's text), after a
    system turn when a system prompt is given. With a picture marker (family.marker) the content is one string. keys
    (chain.KEYS): how a specification's records are written, as the model was trained."""
    if form == "bare":
        text = prompts.bare(it["depth"])
    elif form == "shown":
        text = prompts.shown(it["depth"], examples.get(it["depth"]) or examples.get("medium") or [])
    elif form == "spec":
        text = prompts.spec(it["depth"], it["spec"], keys)
    elif form == "short":
        text = prompts.short(it["depth"])
    elif form == "spec_short":
        text = prompts.spec_short(it["depth"], it["spec"], keys)
    elif form == "full":
        text = prompts.full(it["depth"])
    elif form == "spec_full":
        text = prompts.spec_full(it["depth"], it["spec"], keys)
    else:
        raise ValueError(f"unknown form {form!r}")
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": family.content(text, image, mark)}]


def model_source(model_id: str, trust_remote_code: bool = False, revision: str | None = None) -> str:
    """Where the weights load from: the id as it is, or, for a model whose repository code runs (trust_remote_code), a local
    snapshot at the pinned revision (Phi-4-reasoning-vision's loader reads its own vision weights only from a local folder)."""
    from .hub import resolve_model
    src = resolve_model(model_id)
    if trust_remote_code and not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, revision=revision, token=token() or None)
    return src


def load_model(model_id: str, thinking: bool, adapter: str | None = None, adapter_subfolder: str | None = None,
               trust_remote_code: bool = False, revision: str | None = None, image_settings: dict | None = None,
               adapter_scale: float = 1.0):
    """The processor and the model on the card; with an adapter (a local folder or a Hub repo, with a subfolder), the LoRA
    weights are merged into the model, scaled by adapter_scale (keep.scale_adapter: 1 = the adapter as trained, 0 = the base,
    between = the weights interpolated). A 'hub:<path>' model id is a full fine-tune shipped under the project repo. With
    trust_remote_code the repository's own code loads (from a snapshot at the pinned revision); image_settings go to the
    image processor (family.apply_image_settings)."""
    import torch
    from transformers import AutoProcessor
    _listen()
    src = model_source(model_id, trust_remote_code, revision)
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=trust_remote_code)
    processor.tokenizer.padding_side = "left"
    model = load_vlm(src, trust_remote_code=trust_remote_code, dtype=torch.bfloat16, device_map="cuda")
    family.prepare_processor(processor, model)
    note = family.prepare_model(model)
    if note:
        print(f"  {note}", flush=True)
    family.apply_image_settings(processor, image_settings)
    family.prepare_inference(model)
    if adapter:
        from peft import PeftModel
        kwargs = {"subfolder": adapter_subfolder} if adapter_subfolder else {}
        model = PeftModel.from_pretrained(model, adapter, token=token(), **kwargs)
        if adapter_scale != 1.0:
            from .keep import scale_adapter
            print(f"  the adapter merged at scale {adapter_scale} ({scale_adapter(model, adapter_scale)} layers)", flush=True)
        model = model.merge_and_unload()
    elif adapter_scale != 1.0:
        raise ValueError("adapter_scale needs an adapter")
    model.eval()
    return processor, model


def encode(processor, texts: list, images: list | None = None, **kwargs):
    """The processor over a batch of texts with at most one picture each (images aligned with texts, None where a text has none).
    Qwen3.5's processor takes the pictures as one flat list, matched to the placeholders in order; Gemma 4's takes one list of
    pictures per text and reads a flat list as a single text with several pictures ("Received inconsistently sized batches of
    images (1) and text (4)"); other processors refuse a flat list with errors worded their own way. The flat form is tried
    first, the per-text form on any ValueError; when the per-text form fails too, the first error surfaces."""
    images = list(images or [])
    flat = [im for im in images if im is not None]
    kwargs = dict(text=texts, return_tensors="pt", **kwargs)
    if not flat:
        return processor(**kwargs)
    try:
        return processor(images=flat, **kwargs)
    except ValueError as first:
        if len(images) != len(texts):   # one picture slot per text (None where a text has no picture) is what the per-text form needs
            raise
        try:
            return processor(images=[[im] if im is not None else [] for im in images], **kwargs)
        except ValueError:
            raise first from None


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
        texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False, **family.thinking_kwargs(processor, thinking))
                 for c in convs]
    except TypeError:  # a processor that takes no template switch; the parse strips any thinking and the summary counts it
        texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
    inputs = encode(processor, texts, family.fill_images(model, images), padding=True).to(model.device)
    pad = processor.tokenizer.pad_token_id   # Gemma's pad id is 0: a falsy id is still an id
    gen = {"max_new_tokens": max_new_tokens, "do_sample": False, "eos_token_id": stop_ids(processor, model),
           "pad_token_id": pad if pad is not None else processor.tokenizer.eos_token_id}
    if sampling:
        gen.update(sampling, do_sample=True)
    with torch.no_grad():
        kwargs, start = family.generation_inputs(model, inputs)
        out = model.generate(**kwargs, **gen)
    new = out[:, start:]
    if thinking:
        decoded = [plain(t, processor.tokenizer) for t in processor.batch_decode(new, skip_special_tokens=False)]
    else:
        decoded = processor.batch_decode(new, skip_special_tokens=True)
    decoded = [family.clean(t) for t in decoded]
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
        system: str | None = None, trust_remote_code: bool = False, revision: str | None = None,
        image_settings: dict | None = None, keys: str = "record", adapter_scale: float = 1.0,
        wordings: str | None = None) -> dict:
    """The screen over every model and form; results and summaries on disk and on the Hub (screens/<stamp>[_<tag>]/). With
    system (see prompts.system) every row's conversation opens with that system turn. trust_remote_code / revision: for a
    family whose repository code runs (see family), pinned to a revision; image_settings: the image processor's (a picture
    budget); keys (chain.KEYS): how the specification rows' records are written, as the model was trained. adapter_scale: the
    strength the adapter is merged at (load_model: 1 = as trained, 0 = the base); it is recorded in env.json, on every result
    and in summary.md. wordings: the convention a whole row's wordings are read by (chain.wordings_whole: None = version 1's
    count; "v1.1" for a model taught the version 1.1 rows); recorded in env.json."""
    if keys not in chain.KEYS:
        raise ValueError(f"unknown keys {keys!r}: one of {chain.KEYS}")
    if wordings is not None and wordings not in chain.CONVENTIONS:
        raise ValueError(f"unknown wordings convention {wordings!r}: one of {sorted(chain.CONVENTIONS)}")
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
                   "trust_remote_code": trust_remote_code, "revision": revision, "image_settings": image_settings,
                   "keys": keys, "adapter_scale": adapter_scale, "wordings": wordings or "v1",
                   "n_images": len(image_items), "n_specs": len(spec_items)},
                  fh, indent=1)
    results_path = os.path.join(out_dir, "results.jsonl")
    all_results = []
    for model_id in models:
        print(f"\n== {model_id}" + (f" + {adapter}/{adapter_subfolder or ''}" if adapter else ""), flush=True)
        t_load = time.time()
        processor, model = load_model(model_id, thinking, adapter, adapter_subfolder, trust_remote_code, revision, image_settings,
                                      adapter_scale)
        mark = family.marker(model)
        print(f"  loaded in {time.time() - t_load:.0f} s" + (f" ({family.model_type(model)}: picture marker {mark})" if mark else ""), flush=True)
        if system_text:
            print(f"  system prompt {system!r}: {len(processor.tokenizer(system_text)['input_ids'])} tokens before every row", flush=True)
        results = []
        for form in forms:
            todo = spec_items if form in SPEC_FORMS else image_items
            jobs = []
            for it in todo:
                im = None if form in SPEC_FORMS else pictures[it["id"]]
                jobs.append((it, messages(form, it, data.get("examples", {}), im, system_text, mark, keys), im))
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
                           "text": text, "parsed": parsed, "scores": chain.score(parsed, it, wordings),
                           "reference": chain.reference(it), "gen_s": dt, "new_tokens": n_new, "adapter_scale": adapter_scale}
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
        fh.write(table(summary) + ("" if adapter_scale == 1.0 else f"\n\nadapter scale: {adapter_scale}")
                 + "\n\nlinear attention: " + fast_path_note() + "\n")
    with open(os.path.join(out_dir, "library_notes.json"), "w", encoding="utf-8") as fh:
        json.dump(LIBRARY_NOTES[-50:], fh, indent=1)
    if do_ship:
        ship(out_dir, f"screens/{stamp}", f"screen {stamp}: summary", repo)
    print(f"\nscreen done in {(time.time() - t0) / 60:.1f} min; results in {out_dir}", flush=True)
    return summary


def rescore(results_path: str, inputs: str | None = None, wordings: str | None = None) -> dict:
    """The summary of a shipped results.jsonl under the current chain reader (the texts are kept, so a better reader re-reads
    them); with the screen set given, the rows are scored against it, otherwise against the reference kept in each record.
    wordings: the convention a whole row's wordings are read by (chain.wordings_whole)."""
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
            r["parsed"], r["scores"] = parsed, chain.score(parsed, row, wordings)
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
