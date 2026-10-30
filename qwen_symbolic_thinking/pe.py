"""The prompt enhancers shipped beside Qwen-Image 2.1 (Qwen/Qwen-Image-2.1-PE-I2I, Qwen/Qwen-Image-2.1-PE-T2I: full
fine-tunes of the Qwen3.5-9B architecture that think, then write a picture prompt as JSON) read our pictures and seeds, and
their thinking and prompts are scored on the extraction's rulers.

    from qwen_symbolic_thinking.pe import run
    run(kind="i2i", manifest="hub:AbstractPhil/qwen-thinking-images/manifest_v1.2.jsonl", split="main", per_source=60,
        exclude=("instagram", "trdg_words", "wikidata"), tag="PE_i2i_read")
    run(kind="t2i", seeds="hub:synth/seeds_v1.json", n=1000, tag="PE_t2i_read")

An image-to-image row: the picture and a no-change request (the enhancer's thinking of what stays fixed becomes a full
reading); a text-to-image row: the seed's own captions as the short prompt. Each answer is split at </think> into the thinking
and the JSON prompt; both are scored against the row's captions (said: the share of the caption's content words the text
uses; backed: the share of the text's content words the captions back), the thinking's sentences are mapped onto our step
names by their words, and the run ships pe/<stamp>_<tag>/ (rows.jsonl, summary.json) to the project repo."""
from __future__ import annotations

import collections
import gc
import json
import os
import re
import time
from datetime import datetime, timezone

from . import REPO, __version__
from .chain import STEPS, words
from .extract import MAX_SIDE, fetch_images, hint_of, load_manifest, select_rows
from .hub import ship, token
from .screen import LIBRARY_NOTES, fast_path_note, generate, load_inputs, load_model, preflight

MODELS = {"i2i": "Qwen/Qwen-Image-2.1-PE-I2I", "t2i": "Qwen/Qwen-Image-2.1-PE-T2I"}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}       # the model cards' recipe (the models were trained for sampling)
NO_CHANGE_REQUEST = "Keep this image exactly as it is. Change nothing. Write the prompt that reproduces it."
RATIO_KEYS = ("wh_ratio", "ratio_follow")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_JSON = re.compile(r"\{.*\}", re.S)

# Our sixteen steps as the words that mark them in free text: whole words (a trailing * takes the word's continuations, so
# light* reads lighting and lights); a sentence takes the first step, in the chain's order, whose words it holds.
STEP_WORDS = {
    "see": ("i see", "shows", "showing", "visible", "there is", "there are", "depict*", "we see", "can see", "feature*", "present*"),
    "name": ("called", "known as", "named", "a type of", "species", "breed", "brand", "titled", "identif*", "recogni*"),
    "place": ("left", "right", "center", "centre", "top", "bottom", "foreground", "background", "behind", "in front", "beside",
              "corner", "middle", "edge*", "positioned", "located", "upper", "lower"),
    "count": ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "several", "single", "pair", "group of",
              "number of", "dozen*", "multiple", "many"),
    "relate": ("next to", "holding", "wearing", "facing", "toward", "towards", "between", "attached", "leaning", "touching",
               "interact*", "connected", "looking at", "carrying"),
    "form": ("shape*", "round", "curved", "angular", "silhouette", "outline", "proportion*", "texture*", "surface*", "material*",
             "composition", "framing", "aspect ratio", "orientation", "pose", "posture"),
    "light": ("light*", "lit", "shadow*", "glow*", "bright*", "dim", "sunlight", "highlight*", "illuminat*", "exposure", "backlit",
              "overcast", "dusk", "dawn", "shade*"),
    "colour": ("color*", "colour*", "red", "blue", "green", "yellow", "white", "black", "gray", "grey", "brown", "orange", "pink",
               "purple", "golden", "pale", "dark", "hue*", "tint*", "saturat*", "monochrome", "beige", "tan"),
    "mood": ("mood", "atmosphere", "feel*", "calm", "tense", "serene", "melanchol*", "cheerful", "peaceful", "dramatic",
             "quiet", "lively", "somber", "sombre", "eerie", "cozy", "cosy", "tranquil", "gloomy"),
    "mean": ("suggest*", "implies", "implying", "symbol*", "represent*", "convey*", "evoke*", "meaning", "signif*", "indicat*"),
    "intend": ("intend*", "intention*", "purpose*", "the artist", "the photographer", "designed to", "meant to", "aims", "chosen to",
               "the goal", "deliberate*"),
    "story": ("story", "moment", "about to", "narrative", "event*", "happening", "caught", "mid-", "while"),
    "compare": ("compared", "unlike", "similar*", "resembl*", "like a", "reminiscent", "contrast*", "than", "differ*", "same as"),
    "change": ("change*", "modif*", "replace*", "remove*", "add", "adding", "added", "adjust*", "alter*", "transform*", "convert*",
               "edit*", "keep*", "kept", "preserv*", "retain*", "maintain*", "unchanged", "fixed", "stay", "stays", "remain*"),
    "check": ("check*", "verif*", "confirm*", "make sure", "ensure*", "double-check", "consistent", "final*", "review*", "re-read",
              "all set"),
    "say": ("prompt*", "describe*", "description*", "write", "writing", "wording*", "phrase*", "sentence*", "output*", "json", "directive"),
}
assert tuple(STEP_WORDS) == tuple(STEPS), "the step map follows the chain's step order"
_PATTERNS = {st: re.compile(r"\b(?:" + "|".join((re.escape(w[:-1]) + r"\w*") if w.endswith("*") else (re.escape(w) + r"\b") for w in ws) + ")")
             for st, ws in STEP_WORDS.items()}


def system_prompt(model_id: str) -> str:
    """The system prompt shipped with the model, verbatim."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(model_id, "system_prompt.txt", token=token()), encoding="utf-8") as fh:
        return fh.read().strip()


def split_answer(text: str) -> dict:
    """The enhancer's text as {thinking, prompt, ratio, ratio_key, json_ok, cut}: the thinking before </think>, the JSON after
    it (the first {...} block; a broken block leaves prompt None), cut when the text ends without a closing think tag."""
    thinking, tag, answer = text.partition("</think>")
    thinking = thinking.replace("<think>", "").strip()
    out = {"thinking": thinking, "prompt": None, "ratio": None, "ratio_key": None, "json_ok": False, "cut": tag == ""}
    if not tag:
        return out
    m = _JSON.search(answer)
    if m:
        try:
            data = json.loads(m.group(0))
        except ValueError:
            data = None
        if isinstance(data, dict):
            out["prompt"] = data.get("rewritten_prompt") if isinstance(data.get("rewritten_prompt"), str) else None
            for k in RATIO_KEYS:
                if data.get(k):
                    out["ratio"], out["ratio_key"] = str(data[k]), k
                    break
            out["json_ok"] = out["prompt"] is not None
    if not out["json_ok"] and answer.strip():
        out["prompt"] = None
    return out


def sentences(text: str) -> list:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def step_map(text: str) -> dict:
    """Which of our steps a free text touches: each sentence takes the first step (in the chain's order) whose words it holds;
    returns the step of each sentence (None when none matches), the steps in order of first use, and the coverage of the 16."""
    out = []
    for s in sentences(text):
        low = s.lower()
        out.append(next((st for st, p in _PATTERNS.items() if p.search(low)), None))
    order = []
    for st in out:
        if st and st not in order:
            order.append(st)
    return {"sentences": len(out), "classed": sum(1 for s in out if s), "per_sentence": out, "order": order,
            "counts": dict(collections.Counter(s for s in out if s)), "coverage": len(order) / len(STEPS)}


def grounding(text: str, hint: str) -> dict:
    """said / backed of a text against the captions (the chain's rulers, on free text)."""
    h, t = words(hint), words(text)
    return {"said": (len(t & h) / len(h)) if h else None, "backed": (len(t & h) / len(t)) if (h and t) else None,
            "words": len(t)}


def score(answer: dict, hint: str) -> dict:
    """What a read of one enhancer answer records: the thinking's and the prompt's grounding, lengths, the step map."""
    think, prompt = answer.get("thinking") or "", answer.get("prompt") or ""
    g_t, g_p = grounding(think, hint), grounding(prompt, hint)
    sm = step_map(think)
    return {"think_said": g_t["said"], "think_backed": g_t["backed"], "think_words": g_t["words"],
            "think_sentences": sm["sentences"], "think_chars": len(think),
            "prompt_said": g_p["said"], "prompt_backed": g_p["backed"], "prompt_words": g_p["words"],
            "prompt_sentences": len(sentences(prompt)), "prompt_chars": len(prompt),
            "steps": sm["order"], "step_counts": sm["counts"], "step_coverage": sm["coverage"], "classed": sm["classed"],
            "json_ok": answer.get("json_ok", False), "cut": answer.get("cut", False), "ratio": answer.get("ratio")}


def conversation(system: str, text: str, image=None) -> list:
    """The enhancer's conversation: the shipped system prompt, then the user turn (a picture and the request, or a prompt)."""
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def seed_prompt(item: dict) -> str:
    """A seed's short prompt for the text-to-image enhancer: its first caption sentence (the hint's first sentence)."""
    hint = item.get("hint") or ""
    first = sentences(hint)
    return (first[0] if first else hint).strip() or str(item.get("id"))


def _mean(xs: list):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def run(kind: str = "i2i", model_id: str | None = None, manifest: str = "hub:AbstractPhil/qwen-thinking-images/manifest_v1.2.jsonl",
        split: str | None = "main", sources: tuple | None = None, exclude: tuple | None = ("instagram", "trdg_words", "wikidata"),
        per_source: int | None = 60, n: int | None = None, seeds: str = "hub:synth/seeds_v1.json", request: str = NO_CHANGE_REQUEST,
        batch: int = 16, max_new_tokens: int = 4096, max_side: int = MAX_SIDE, sampling: dict | None = None, seed: int = 0,
        out_dir: str = "/content/pe", repo: str = REPO, do_ship: bool = True, tag: str = "pe") -> dict:
    """The enhancer of `kind` reads the manifest's pictures (i2i: the no-change request) or rewrites the seeds' prompts (t2i);
    every answer split, scored and written to rows.jsonl; the summary by source; the folder shipped to pe/<stamp>_<tag>/."""
    import torch
    if kind not in MODELS:
        raise ValueError(f"kind must be one of {tuple(MODELS)}, not {kind!r}")
    model_id = model_id or MODELS[kind]
    sampling = dict(SAMPLING, **(sampling or {}))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{tag}"
    out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    env = preflight()
    system = system_prompt(model_id)
    if kind == "i2i":
        rows = select_rows(load_manifest(manifest), split, sources, exclude, per_source, n)
        ids = [r["sample_id"] for r in rows]
        hints = {r["sample_id"]: hint_of(r) for r in rows}
        srcs = {r["sample_id"]: r["source"] for r in rows}
        texts_in = {r["sample_id"]: request for r in rows}
        pictures = fetch_images(rows, os.path.join(os.path.dirname(out_dir), "images"), max_side=max_side)
        missing = [i for i in ids if i not in pictures]
        ids = [i for i in ids if i in pictures]
    else:
        items = load_inputs(seeds)["items"][:n]
        ids = [it["id"] for it in items]
        hints = {it["id"]: it.get("hint") or "" for it in items}
        srcs = {it["id"]: str(it.get("seed_id", "seed")).split(":")[0] for it in items}
        texts_in = {it["id"]: seed_prompt(it) for it in items}
        pictures, missing = {}, []
    print(f"pe {stamp}: {kind} {model_id}; {len(ids)} rows ({len(missing)} pictures missing); batch {batch}, {max_new_tokens} new tokens, "
          f"sampling {sampling}", flush=True)
    torch.manual_seed(seed)
    processor, model = load_model(model_id, True)
    rows_path = os.path.join(out_dir, "rows.jsonl")
    t0, n_done, n_tok = time.time(), 0, 0
    by_src = collections.defaultdict(list)
    errors = 0
    for b0 in range(0, len(ids), batch):
        part = ids[b0:b0 + batch]
        convs = [conversation(system, texts_in[i], pictures.get(i)) for i in part]
        images = [pictures.get(i) for i in part]
        try:
            texts, lengths = generate(processor, model, convs, images, max_new_tokens, True, sampling)
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            texts, lengths = [f"<error {type(e).__name__}: {str(e)[:200]}>"] * len(part), [0] * len(part)
            errors += len(part)
        n_tok += sum(lengths)
        with open(rows_path, "a", encoding="utf-8") as fh:
            for i, text, length in zip(part, texts, lengths):
                answer = split_answer(text)
                s = score(answer, hints[i])
                s["new_tokens"] = length
                by_src[srcs[i]].append(s)
                fh.write(json.dumps({"id": i, "source": srcs[i], "kind": kind, "input": texts_in[i], "hint": hints[i], "text": text,
                                     "thinking": answer["thinking"], "prompt": answer["prompt"], "ratio": answer["ratio"],
                                     "ratio_key": answer["ratio_key"], "scores": s}, ensure_ascii=False) + "\n")
        n_done += len(part)
        spent = time.time() - t0
        print(f"  {n_done}/{len(ids)} rows, {spent:.0f} s spent, {spent / n_done * (len(ids) - n_done):.0f} s left, "
              f"{n_tok / max(spent, 1e-9):.0f} tok/s, mean {n_tok / n_done:.0f} new tokens", flush=True)
        if n_done == len(part):
            print("  linear attention:", fast_path_note(), flush=True)
    by_source = {}
    for src in sorted(by_src):
        ss = by_src[src]
        by_source[src] = {"rows": len(ss), "json_ok": _mean([float(s["json_ok"]) for s in ss]), "cut": _mean([float(s["cut"]) for s in ss]),
                          "think_said": _mean([s["think_said"] for s in ss]), "think_backed": _mean([s["think_backed"] for s in ss]),
                          "prompt_said": _mean([s["prompt_said"] for s in ss]), "prompt_backed": _mean([s["prompt_backed"] for s in ss]),
                          "think_sentences": _mean([s["think_sentences"] for s in ss]), "prompt_words": _mean([s["prompt_words"] for s in ss]),
                          "step_coverage": _mean([s["step_coverage"] for s in ss]), "new_tokens": _mean([s["new_tokens"] for s in ss])}
    allrows = [s for ss in by_src.values() for s in ss]
    step_first = collections.Counter(s["steps"][0] for s in allrows if s["steps"])
    summary = {"package": __version__, "kind": kind, "model": model_id, "manifest": manifest if kind == "i2i" else None,
               "seeds": seeds if kind == "t2i" else None, "split": split, "sources": list(sources) if sources else None,
               "exclude": list(exclude) if exclude else None, "per_source": per_source, "n": n, "request": request if kind == "i2i" else None,
               "batch": batch, "max_new_tokens": max_new_tokens, "max_side": max_side, "sampling": sampling, "seed": seed,
               "rows": len(ids), "pictures_missing": len(missing), "errors": errors,
               "json_ok": _mean([float(s["json_ok"]) for s in allrows]), "cut": _mean([float(s["cut"]) for s in allrows]),
               "think_said": _mean([s["think_said"] for s in allrows]), "think_backed": _mean([s["think_backed"] for s in allrows]),
               "prompt_said": _mean([s["prompt_said"] for s in allrows]), "prompt_backed": _mean([s["prompt_backed"] for s in allrows]),
               "step_coverage": _mean([s["step_coverage"] for s in allrows]), "first_step": dict(step_first),
               "s_per_row": round((time.time() - t0) / max(n_done, 1), 3), "tok_per_s": round(n_tok / max(time.time() - t0, 1e-9), 1),
               "mean_new_tokens": round(n_tok / max(n_done, 1), 1), "by_source": by_source, "env": env, "library_notes": LIBRARY_NOTES[-20:]}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\npe done: {n_done} rows; json ok {summary['json_ok']:.0%}; cut {summary['cut']:.0%}; thinking said / backed "
          f"{summary['think_said'] or 0:.2f} / {summary['think_backed'] or 0:.2f}; prompt {summary['prompt_said'] or 0:.2f} / "
          f"{summary['prompt_backed'] or 0:.2f}; {summary['s_per_row']} s a row", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if do_ship:
        print("  shipped:", ship(out_dir, f"pe/{stamp}", f"pe {stamp}: {kind} {n_done} rows ({model_id})", repo), flush=True)
    summary["path_in_repo"] = f"pe/{stamp}"
    return summary
