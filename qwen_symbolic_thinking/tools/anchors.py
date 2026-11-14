"""General rows: requests that do not ask for the row form, answered by the base model itself.

    from qwen_symbolic_thinking.tools.anchors import run
    run(model_id="Qwen/Qwen3.5-4B", rows="hub:sft/api_rows.json", tag="g1")

A fine-tune that teaches one answer form can leave the model writing that form where it was not asked for, and can move its
other answers. Holding the model near the base (keep.py) needs rows to read the drift on: ordinary requests the base model
answers on its own. This tool writes them on a GPU, with no adapter, greedy decoding and thinking off:

    picture rows   one plain request about each of the training items' pictures (a fixed bank of everyday questions, one drawn
                   for each picture), the pictures taken round-robin over their sources so no source crowds the rest;
    text rows      requests the base model writes itself: the chat template's opening of a user turn is given to the model,
                   which continues it as a user would (Xu et al., "Magpie: Alignment Data Synthesis from Scratch by Prompting
                   Aligned LLMs with Nothing", arXiv 2406.08464), then each request is answered;
    held rows      the same, for pictures and requests kept out of training, to read a trained adapter on (tools.keep_read);
                   each held picture also gets one request for a detailed description, the reference for the read of how much
                   detail the row form keeps.

An answer that reaches its token cap never ended, so it is dropped. Items are {"kind": "anchor", "id", "prompt", "answer",
"writer" (the base model), "split", "category", "source", "seed_id"} plus the picture's locator copied from its source item, and
ship to the project repo as anchors/<tag>/anchors.json (split train), held.json (split held) and summary.json."""
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone

from .. import REPO, __version__, family, screen
from ..extract import fetch_images
from ..hub import ship
from ..screen import encode, load_inputs, load_model, preflight
from ..template import stop_ids

# Everyday requests about a picture; none of them asks for the row form.
BANK = (
    "Describe this picture in one sentence.",
    "What is the main subject of this picture?",
    "List the things you can see in this picture.",
    "How many people are in this picture?",
    "What colors stand out in this picture?",
    "Is there any text in this picture? If so, what does it say?",
    "Where might this picture have been taken?",
    "What time of day does it look like, and why?",
    "What is the weather like here?",
    "What are the people in this picture wearing?",
    "Write a short social media caption for this picture.",
    "Write a two-line poem about this picture.",
    "Is there an animal in this picture? Answer yes or no, then name it if there is one.",
    "Give this picture a title.",
    "Describe the lighting in this picture.",
    "What is happening in this picture?",
    "Describe this picture to someone who cannot see it.",
    "What is in the background of this picture?",
    "Is this a photograph, a painting, a drawing or something else?",
    "Describe this picture in French, in two sentences.",
    "What would you change to make this picture better?",
    "What objects are on the left side of the picture?",
)
DETAIL = "Describe this picture in detail: every person, garment, object, color, and any text you can see."
TEXT_BANK = (   # the asks about the text a picture shows (bank="text"): the base model's own answers keep its reading of text
    "What does the text in this picture say? Write it out exactly.",
    "Read every word you can see in this picture, in order.",
    "Is there any writing in this picture? Copy it exactly.",
    "What is written on the sign, label or screen in this picture?",
    "Transcribe the text in this picture, line by line.",
    "Where in this picture is the text, and what does it say?",
    "Describe this picture, quoting any text you can read exactly.",
    "What name, brand or number can you read in this picture?",
    "Write a one-sentence caption for this picture that includes any text you can read.",
)
BANKS = {"general": BANK, "text": TEXT_BANK}
CATEGORIES =("picture-train", "picture-held", "picture-held-detail", "text-train", "text-held")
REQUEST_CHARS = (20, 500)       # a request shorter than a phrase or longer than a paragraph is not an everyday request
REQUEST_TOKENS = 160            # a request of 500 characters takes about 120 tokens; one still open at 160 is not one request
USER_MARK = "USERTEXTMARKER"
DEFAULT_HELD = "hub:screens/inputs/screen_v1.json"


# ------------------------------------------------------------------------------------------------------ pure parts
def source_of(it: dict) -> str:
    """Where an item's picture comes from: its source, else the prefix of its seed id."""
    return str(it.get("source") or (it.get("seed_id") or "").split(":")[0] or "unknown")


def pick_pictures(items: list, n: int, seed: int, exclude=()) -> list:
    """Up to n picture items, one per picture, round-robin over their sources (sources sorted, the pictures of each source in a
    seeded order), so a large source does not crowd out a small one. Items without a locator, a second item of a picture
    already taken, and pictures whose seed id is in `exclude` are skipped."""
    rng = random.Random(seed)
    excluded, seen, by_source = set(exclude), set(), {}
    for it in items:
        if it.get("kind") != "image" or not (it.get("image_urls") or it.get("image")):
            continue
        key = it.get("seed_id") or it["id"]
        if key in seen or it.get("seed_id") in excluded:
            continue
        seen.add(key)
        by_source.setdefault(source_of(it), []).append(it)
    queues = [by_source[s] for s in sorted(by_source)]
    for q in queues:
        rng.shuffle(q)
    out, depth = [], 0
    while len(out) < n and any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q) and len(out) < n:
                out.append(q[depth])
        depth += 1
    return out


def load_held(held_pictures) -> list:
    """The picture items of one input file, or of several (a list or tuple of files, in their order): the held pictures."""
    files = [held_pictures] if isinstance(held_pictures, str) else list(held_pictures)
    out = []
    for f in files:
        out += [it for it in load_inputs(f)["items"] if it.get("kind") == "image"]
    return out


def bank_of(bank) -> tuple:
    """The asks of a bank: a name of BANKS ("general", "text"), a list of asks, or None (the general bank)."""
    if bank is None:
        return BANK
    if isinstance(bank, str):
        if bank not in BANKS:
            raise ValueError(f"unknown bank {bank!r}: one of {sorted(BANKS)}, or a list of asks")
        return BANKS[bank]
    asks = tuple(str(a).strip() for a in bank)
    if not asks or not all(asks):
        raise ValueError("a bank of asks must hold at least one non-empty ask")
    return asks


def draw_requests(n: int, seed: int, bank=None) -> list:
    """n requests from the bank, drawn in seeded shuffled passes over the whole bank, so each request is used about equally."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        bank_list = list(bank_of(bank))
        rng.shuffle(bank_list)
        out += bank_list
    return out[:n]


def user_turn(render) -> tuple:
    """(the template's opening of a user turn, what the template writes after a user turn's text), read from the template
    itself: a one-message user conversation is rendered through render(messages) and cut around the user text. A request the
    model continues from the opening ends where the closing begins."""
    last = None
    for content in ([{"type": "text", "text": USER_MARK}], USER_MARK):
        try:
            text = render([{"role": "user", "content": content}])
        except Exception as e:      # a template that wants the content as a plain string
            last = e
            continue
        if USER_MARK in text:
            at = text.index(USER_MARK)
            return text[:at], text[at + len(USER_MARK):]
    raise ValueError(f"the chat template did not render a user turn's text{f': {last}' if last else ''}")


def cut_request(decoded: str, closing: str, n_new: int, cap: int) -> str | None:
    """The request in a sampled continuation: the text before the template's closing of a user turn; None when the model never
    wrote one (the request did not end). A template with nothing after a user turn ends at the model's own stop: a
    continuation shorter than the cap."""
    end = closing.strip()
    if end:
        return decoded[:decoded.index(end)].strip() if end in decoded else None
    return decoded.strip() if n_new < cap else None


def keep_requests(candidates: list, seen: set, chars: tuple = REQUEST_CHARS) -> tuple:
    """(the requests kept, the drops by reason). A candidate is dropped when it never ended (None), is outside the length range
    in characters, still holds chat markup, or repeats a request in `seen` (case-folded); the kept ones join `seen`."""
    kept, drops = [], Counter()
    for text in candidates:
        if text is None:
            drops["unfinished"] += 1
        elif not chars[0] <= len(text) <= chars[1]:
            drops["length"] += 1
        elif "<|" in text or "|>" in text:
            drops["markup"] += 1
        elif text.casefold() in seen:
            drops["repeat"] += 1
        else:
            seen.add(text.casefold())
            kept.append(text)
    return kept, drops


def reached_cap(n_new: int, cap: int) -> bool:
    """An answer with as many new tokens as its cap never ended (an answer that ends counts its end token, and still stops
    short of the cap unless it ended on its last token, a case dropped with the rest)."""
    return n_new >= cap


def make_item(n: int, tag: str, model_id: str, prompt: str, answer: str, split: str, category: str, source: str | None = None,
              seed_id: str | None = None, locator_from: dict | None = None) -> dict:
    """One general row; the picture's locator (image_urls, image) is copied from its source item."""
    it = {"kind": "anchor", "id": f"anc-{tag}-{n}", "prompt": prompt, "answer": answer, "writer": model_id, "split": split,
          "category": category, "source": source, "seed_id": seed_id}
    for key in ("image_urls", "image"):
        if locator_from and locator_from.get(key):
            it[key] = locator_from[key]
    return it


# ------------------------------------------------------------------------------------------------------ on the card
def _size(conv: list) -> int:
    n = 0
    for m in conv:
        c = m["content"]
        n += len(c) if isinstance(c, str) else sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return n


def generate_texts(processor, model, convs: list, images: list, cap: int, batch: int, label: str = "") -> tuple:
    """(texts, new-token counts) for the conversations, in their order: greedy, thinking off (screen.generate), the longest
    prompts together so a batch pads little; a batch that runs out of memory is retried in halves. A conversation that fails
    comes back as (None, 0). Prints the rows done, the time spent and the time left after every batch."""
    import torch
    order = sorted(range(len(convs)), key=lambda i: _size(convs[i]))
    texts, lengths = [None] * len(convs), [0] * len(convs)

    def go(idx: list):
        try:
            t, n = screen.generate(processor, model, [convs[i] for i in idx], [images[i] for i in idx], cap, False, None)
            for i, tt, nn in zip(idx, t, n):
                texts[i], lengths[i] = tt, nn
        except Exception as e:
            oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
            gc.collect()
            torch.cuda.empty_cache()
            if oom and len(idx) > 1:
                go(idx[:len(idx) // 2])
                go(idx[len(idx) // 2:])
            else:
                print(f"  {len(idx)} rows failed: {type(e).__name__}: {str(e)[:160]}", flush=True)

    t0 = time.time()
    for start in range(0, len(order), batch):
        go(order[start:start + batch])
        done = min(start + batch, len(order))
        spent = time.time() - t0
        print(f"  [{label}] {done}/{len(order)} rows, {spent:.0f} s spent, {spent / done * (len(order) - done):.0f} s left", flush=True)
    return texts, lengths


def sample_requests(processor, model, need: int, batch: int, seed: int) -> tuple:
    """(up to `need` requests the base model writes, the drops by reason): sampled at temperature 1.0 with top-p 0.95 from the
    template's opening of a user turn until the turn ends, in seeded rounds, until enough pass keep_requests."""
    import torch
    opening, closing = user_turn(lambda msgs: processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False))
    tok = processor.tokenizer
    stops = set(stop_ids(processor, model))
    piece = tok(closing.strip(), add_special_tokens=False)["input_ids"] if closing.strip() else []
    if len(piece) == 1:
        stops.add(int(piece[0]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kept, seen, drops = [], set(), Counter()
    rounds = 8 + 3 * math.ceil(need / batch)         # a round yields fewer than a batch once repeats and odd lengths drop out
    t0 = time.time()
    for r in range(rounds):
        if len(kept) >= need:
            break
        torch.manual_seed(seed + r)
        inputs = encode(processor, [opening] * batch, None, padding=True).to(model.device)
        gen = {"max_new_tokens": REQUEST_TOKENS, "do_sample": True, "temperature": 1.0, "top_p": 0.95,
               "eos_token_id": sorted(stops), "pad_token_id": pad}
        with torch.no_grad():
            kwargs, start = family.generation_inputs(model, inputs)
            out = model.generate(**kwargs, **gen)
        new = out[:, start:]
        decoded = processor.batch_decode(new, skip_special_tokens=False)
        counts = [int((row != pad).sum()) for row in new]
        got, d = keep_requests([cut_request(t, closing, n, REQUEST_TOKENS) for t, n in zip(decoded, counts)], seen)
        kept += got
        drops.update(d)
        spent = time.time() - t0
        print(f"  [requests] round {r + 1}: {len(kept)}/{need} kept, {spent:.0f} s spent", flush=True)
    return kept[:need], dict(drops)


def run(model_id: str = "Qwen/Qwen3.5-4B", rows: str | None = None, n_picture: int = 1600, n_text: int = 800,
        n_held_picture: int = 200, n_held_text: int = 200, held_pictures: str = DEFAULT_HELD, batch: int = 32,
        max_new_tokens: int = 512, max_new_tokens_text: int = 768, seed: int = 20261110, tag: str = "", repo: str = REPO,
        out_dir: str = "/content/anchors", do_ship: bool = True, bank=None, detail: str | None = DETAIL) -> dict:
    """Write the general rows with the base model (no adapter) and ship anchors/<tag>/{anchors,held,summary}.json to the
    project repo. rows: an input file of training items (built by rows.build_api_rows), whose pictures the training picture rows
    use; held_pictures: an input file, or a list of them, whose picture items are held out of training (the screen set; the
    pictures rows.build_api_rows draws out with hold_out), whose pictures the held rows use. The detail rows and the text rows
    answer under max_new_tokens_text (a detailed description or a free request runs longer than a one-line answer)."""
    if not rows:
        raise ValueError("rows: the input file of training items whose pictures the general rows use (rows.build_api_rows)")
    label = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, label)
    os.makedirs(out, exist_ok=True)
    env = preflight()
    held_items = load_held(held_pictures)
    held_seeds = {it["seed_id"] for it in held_items if it.get("seed_id")}
    train_pics = pick_pictures(load_inputs(rows)["items"], n_picture, seed, exclude=held_seeds)
    held_pics = pick_pictures(held_items, n_held_picture, seed + 1)
    print(f"anchors {label}: {len(train_pics)} training pictures from {len({source_of(p) for p in train_pics})} sources, "
          f"{len(held_pics)} held pictures, {n_text} + {n_held_text} text requests; {model_id}", flush=True)

    locators = {}
    for group, pics in (("train", train_pics), ("held", held_pics)):
        for i, p in enumerate(pics):
            locators[f"{group}-{i}"] = {"id": f"{group}-{i}", "image_urls": p.get("image_urls") or [], "image": p.get("image")}
    t0 = time.time()
    pictures = fetch_images(list(locators.values()), os.path.join(out_dir, "images"))
    print(f"  {len(pictures)} of {len(locators)} pictures ready in {time.time() - t0:.0f} s", flush=True)

    processor, model = load_model(model_id, False)
    mark = family.marker(model)
    drops = {c: Counter() for c in CATEGORIES}
    requests, request_drops = sample_requests(processor, model, n_text + n_held_text, batch, seed)
    text_train, text_held = requests[:n_text], requests[n_text:n_text + n_held_text]
    print(f"  {len(requests)} requests written ({request_drops})", flush=True)

    jobs = []     # (category, prompt, source item or None, picture key or None, token cap)
    for i, (p, ask) in enumerate(zip(train_pics, draw_requests(len(train_pics), seed, bank))):
        jobs.append(("picture-train", ask, p, f"train-{i}", max_new_tokens))
    for t in text_train:
        jobs.append(("text-train", t, None, None, max_new_tokens_text))
    held_asks = draw_requests(len(held_pics), seed + 2, bank)
    for i, (p, ask) in enumerate(zip(held_pics, held_asks)):
        jobs.append(("picture-held", ask, p, f"held-{i}", max_new_tokens))
    for i, p in enumerate(held_pics if detail else []):   # detail None: no detail rows (a bank of its own asks instead)
        jobs.append(("picture-held-detail", detail, p, f"held-{i}", max_new_tokens_text))
    for t in text_held:
        jobs.append(("text-held", t, None, None, max_new_tokens_text))
    ready = []
    for job in jobs:
        if job[3] is not None and job[3] not in pictures:
            drops[job[0]]["no_picture"] += 1
        else:
            ready.append(job)

    answers, t_gen, n_tok = {}, 0.0, 0
    for name, group, cap in (("pictures", [j for j in ready if j[0] in ("picture-train", "picture-held")], max_new_tokens),
                             ("details", [j for j in ready if j[0] == "picture-held-detail"], max_new_tokens_text),
                             ("text", [j for j in ready if j[3] is None], max_new_tokens_text)):
        if not group:
            continue
        convs = [[{"role": "user", "content": family.content(j[1], pictures.get(j[3]), mark)}] for j in group]
        t1 = time.time()
        texts, lengths = generate_texts(processor, model, convs, [pictures.get(j[3]) for j in group], cap, batch, name)
        t_gen += time.time() - t1
        for j, text, n in zip(group, texts, lengths):
            answers[id(j)] = (text, n, cap)
            n_tok += n
    items = {"train": [], "held": []}
    n = 0
    for job in ready:
        category, prompt, src, key, cap = job
        text, n_new, _ = answers[id(job)]
        if text is None:
            drops[category]["error"] += 1
        elif reached_cap(n_new, cap):
            drops[category]["reached_cap"] += 1
        elif not text.strip():
            drops[category]["empty"] += 1
        else:
            n += 1
            split = "train" if category.endswith("-train") else "held"
            items[split].append(make_item(n, label, model_id, prompt, text.strip(), split, category,
                                          source_of(src) if src else None, (src or {}).get("seed_id"), src))
    n_rows = sum(len(v) for v in items.values())
    summary = {"package": __version__, "model": model_id, "tag": label, "seed": seed, "rows": rows, "held_pictures": held_pictures,
               "asked": {"n_picture": n_picture, "n_text": n_text, "n_held_picture": n_held_picture, "n_held_text": n_held_text},
               "bank": bank if isinstance(bank, str) or bank is None else list(bank_of(bank)), "detail": detail,
               "counts": dict(Counter(it["category"] for v in items.values() for it in v)),
               "by_source": dict(Counter(it["source"] for v in items.values() for it in v if it["source"])),
               "dropped": {c: dict(d) for c, d in drops.items() if d}, "requests_dropped": request_drops,
               "max_new_tokens": max_new_tokens, "max_new_tokens_text": max_new_tokens_text, "batch": batch,
               "s_per_row": round(t_gen / max(len(ready), 1), 3), "tok_per_s": round(n_tok / max(t_gen, 1e-9), 1),
               "answered": len(ready), "kept": n_rows, "env": env}
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, split in (("anchors.json", "train"), ("held.json", "held")):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"version": 1, "built": built, "model": model_id, "tag": label, "split": split, "items": items[split]},
                      fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nanchors done: {len(items['train'])} general rows for training, {len(items['held'])} held; dropped "
          f"{summary['dropped']}; {summary['s_per_row']} s a row, {summary['tok_per_s']} tok/s", flush=True)
    del model, processor
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    if do_ship:
        print("  shipped:", ship(out, f"anchors/{label}", f"anchors {label}: {len(items['train'])} train, {len(items['held'])} held", repo),
              flush=True)
    return summary
