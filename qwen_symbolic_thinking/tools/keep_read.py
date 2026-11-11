"""How far a trained adapter moved the model from its base, read on held general rows.

    from qwen_symbolic_thinking.tools.keep_read import run
    run(model_id="Qwen/Qwen3.5-4B", adapter="owner/repo", adapter_subfolder="lora/.../final", held="hub:anchors/g1/held.json",
        scales=(0.0, 0.5, 1.0))

The held rows are the base model's own answers to requests that do not ask for the row form (tools.anchors). The adapter is
loaded beside the base model without merging, and the base side of every read is the same model inside the adapter's off
switch, so one copy of the weights serves both. A scale s multiplies every LoRA layer's scaling by s (the trained value stays
the reference and is restored before each scale), so the read can be repeated at each strength the adapter may ship at; scale 0 is
the base itself. The base side runs once and is reused at every scale.

    read 1, drift       teacher-forced on each held row's own answer: the KL(p_base || p_tuned) at the answer positions (the row's
                        mean, and the mean at each of the first eight positions), and the answer's negative log-likelihood
                        under both models;
    read 2, takeover    greedy answers to the held requests: the share that take the row form, the share that end before the
                        token cap, the length against the base's answer, and the word overlap (token F1) with the base's answer;
    read 3, detail      the full row form asked of each held picture the base also described in detail: the share of the base's
                        content words the row still holds (recall), and the row's length.

With no adapter the tuned side is the base against itself (every drift 0) and the generations run once."""
from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

from .. import REPO, __version__, chain, family, keep, prompts
from ..extract import fetch_images
from ..hub import ship, token
from ..screen import load_inputs, load_model, preflight
from .anchors import generate_texts

DETAIL_CATEGORY = "picture-held-detail"
FIRST_POSITIONS = 8
_FORM_LINE = re.compile(r"^\s*(step|question)\s*:", re.I | re.M)     # the form's own generic labels
STOPWORDS = frozenset("""
the and for are but not you all any can had her was one our out day get has him his how its let may new now old see two way who
did yes she too use this that with have from they been were will your what when where which while would could should their there
these those then than them into onto over under about after before again also because between both each few more most other
some such only own same very just here once does doing being having above below down off through during until against
among along around across behind beside near upon within without toward towards picture image photo shows showing shown looks
looking looks like seems appears appear seem can cannot thing things lot lots kind sort part side
""".split())


# ------------------------------------------------------------------------------------------------------ pure parts
def percentile(values: list, q: float):
    """The q-th percentile (0-100) of values by linear interpolation between the closest ranks; None for no values."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: list):
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def takes_row_form(text: str) -> bool:
    """Whether an answer is written in the row form: two or more thoughts under step names (chain.parse), or one such thought
    with a wording or a question line beside it, or two or more lines under the form's own generic labels (step, question).
    A single labelled line in a plain answer ('Story: Once upon a time ...', 'Mood: calm') and labelled lines that are not
    thoughts ('Caption: ...' and 'Tags: ...' under a social media caption) do not count: a plain answer writes those too."""
    parsed = chain.parse(text or "")
    named = sum(1 for t in parsed["thoughts"] if t["step"] != chain.UNNAMED)
    return (named >= 2 or (named >= 1 and bool(parsed["wordings"] or parsed["questions"]))
            or len(_FORM_LINE.findall(text or "")) >= 2)


def word_tokens(text: str) -> list:
    """Lowercase word tokens (letters and digits, accents kept)."""
    return re.findall(r"\w+", (text or "").lower())


def token_f1(answer: str, reference: str) -> float:
    """The word-token F1 of an answer against a reference (a bag of words, each word counted as often as it occurs); 1.0 for two
    empty texts, 0.0 when only one is empty."""
    a, b = Counter(word_tokens(answer)), Counter(word_tokens(reference))
    if not a and not b:
        return 1.0
    shared = sum((a & b).values())
    if not shared:
        return 0.0
    p, r = shared / sum(a.values()), shared / sum(b.values())
    return 2 * p * r / (p + r)


def content_words(text: str) -> set:
    """The words that carry content: lowercase alphabetic tokens of three letters or more that are not common English words,
    each with one trailing "s" removed (so a plural and its singular are one word)."""
    out = set()
    for w in re.findall(r"[^\W\d_]+", (text or "").lower()):
        if len(w) >= 3 and w not in STOPWORDS:
            out.add(w[:-1] if w.endswith("s") else w)
    return out


def recall(reference: str, generated: str):
    """The share of the reference's content words that the generated text holds; None when the reference has none."""
    ref = content_words(reference)
    return len(ref & content_words(generated)) / len(ref) if ref else None


def lora_layers(model) -> list:
    """[(layer, its trained scaling)] for every LoRA layer: the reference set_scale multiplies."""
    layers = [(m, dict(m.scaling)) for m in model.modules() if isinstance(getattr(m, "scaling", None), dict) and m.scaling]
    if not layers:
        raise ValueError("no LoRA layer to scale")
    return layers


def set_scale(layers: list, factor: float) -> None:
    """Every LoRA layer's scaling = its trained value x factor (the trained value is kept, so the call can be repeated)."""
    for module, trained in layers:
        for name, value in trained.items():
            module.scaling[name] = value * factor


def drift_summary(records: list) -> dict:
    """Per category and overall: the rows, the mean KL, the 50th, 90th and 99th percentile of the rows' KL, the mean KL at each
    of the first eight answer positions, and exp(mean tuned NLL) / exp(mean base NLL)."""
    groups = {"all": list(records)}
    for r in records:
        groups.setdefault(r["category"], []).append(r)
    out = {}
    for name, rs in groups.items():
        kls = [r["kl"] for r in rs]
        nb, nt = mean([r["nll_base"] for r in rs]), mean([r["nll_tuned"] for r in rs])
        out[name] = {"rows": len(rs), "kl_mean": mean(kls), "kl_p50": percentile(kls, 50), "kl_p90": percentile(kls, 90),
                     "kl_p99": percentile(kls, 99),
                     "kl_by_position": [mean([r["kl_first"][p] for r in rs if len(r["kl_first"]) > p]) for p in range(FIRST_POSITIONS)],
                     "nll_base": nb, "nll_tuned": nt,
                     "perplexity_ratio": math.exp(nt - nb) if nb is not None and nt is not None else None}
    return out


def takeover_summary(records: list) -> dict:
    """Per category and overall: the share that take the row form, the share that end before the token cap, the mean length
    ratio against the base's answer and the mean token F1 against it."""
    groups = {"all": list(records)}
    for r in records:
        groups.setdefault(r["category"], []).append(r)
    return {name: {"rows": len(rs), "takes_row_form": mean([float(r["takes_row_form"]) for r in rs]),
                   "ended": mean([float(r["ended"]) for r in rs]), "length_ratio": mean([r["length_ratio"] for r in rs]),
                   "f1": mean([r["f1"] for r in rs])} for name, rs in groups.items()}


def detail_summary(records: list) -> dict:
    """The detail rows: how many, the mean recall of the base's detail, the mean length of the row (characters and tokens), and the
    share that ended before the token cap."""
    return {"rows": len(records), "recall": mean([r["recall"] for r in records]),
            "row_chars": mean([float(r["row_chars"]) for r in records]),
            "row_tokens": mean([float(r["row_tokens"]) for r in records]),
            "ended": mean([float(r["ended"]) for r in records])}


# ------------------------------------------------------------------------------------------------------ on the card
def kl_per_position(z_base, z_tuned):
    """KL(p_base || p_tuned) at each row of two logit matrices [N, V], in fp32 (the same quantity as keep.kl_full, before its
    mean)."""
    import torch
    lb = torch.log_softmax(z_base.float(), dim=-1)
    lt = torch.log_softmax(z_tuned.float(), dim=-1)
    return (lb.exp() * (lb - lt)).sum(dim=-1)


def _row_conversations(rows: list, pictures: dict, mark, form: str = "anchor") -> tuple:
    from ..train import conversation
    convs, images = [], []
    for it in rows:
        image = pictures.get(it["id"])
        base = {"depth": "medium"} if form == "full" else it
        convs.append(conversation(base, form, image, None, mark)[0])
        images.append(image)
    return convs, images


def _generate_rows(processor, model, rows: list, pictures: dict, mark, cap: int, batch: int, label: str, form: str = "anchor") -> tuple:
    """(texts, new-token counts) for rows, a picture row and a text row never in one batch."""
    texts, lengths = [None] * len(rows), [0] * len(rows)
    for name, keep_idx in (("picture", [i for i, it in enumerate(rows) if it["id"] in pictures]),
                           ("text", [i for i, it in enumerate(rows) if it["id"] not in pictures])):
        if not keep_idx:
            continue
        convs, images = _row_conversations([rows[i] for i in keep_idx], pictures, mark, form)
        t, n = generate_texts(processor, model, convs, images, cap, batch, f"{label} {name}")
        for i, tt, nn in zip(keep_idx, t, n):
            texts[i], lengths[i] = tt, nn
    return texts, lengths


def drift_pass(processor, model, peft: bool, layers, scales: tuple, rows: list, pictures: dict, mark, kind: str, micro: int,
               max_len: int) -> dict:
    """{scale: {row id: drift record}} by teacher forcing on each row's own answer. The base logits at the answer positions are
    computed once per micro-batch (adapter off) and read against the tuned logits at every scale. Without an adapter the tuned
    side is the base: every KL is 0."""
    import torch
    from ..train import Collator
    processor.tokenizer.padding_side = "right"          # the collator's prompt lengths count from the left edge
    collate = Collator(processor, pictures, max_len, None, mark, kind)
    out = {s: {} for s in scales}
    t0 = time.time()
    for start in range(0, len(rows), micro):
        group = rows[start:start + micro]
        batch = collate([(it, "anchor") for it in group])
        batch = batch.to(model.device) if hasattr(batch, "to") else batch
        labels = batch.pop("labels")
        shifted = labels[:, 1:]
        mask = shifted != -100
        with torch.no_grad():
            with (model.disable_adapter() if peft else contextlib.nullcontext()):
                z = model(**batch).logits[:, :-1]
                base = [z[i][mask[i]] for i in range(len(group))]
                del z
            ys = [shifted[i][mask[i]] for i in range(len(group))]
            nll_base = [float(keep.ce_mean(base[i], ys[i])) for i in range(len(group))]
            for s in scales:
                if peft:
                    set_scale(layers, s)
                    z = model(**batch).logits[:, :-1]
                    tuned = [z[i][mask[i]] for i in range(len(group))]
                    del z
                else:
                    tuned = base
                for i, it in enumerate(group):
                    if peft:
                        kl = float(keep.kl_full(base[i], tuned[i]))
                        first = [float(v) for v in kl_per_position(base[i][:FIRST_POSITIONS], tuned[i][:FIRST_POSITIONS])]
                        nll_t = float(keep.ce_mean(tuned[i], ys[i]))
                    else:
                        kl, first, nll_t = 0.0, [0.0] * min(FIRST_POSITIONS, int(ys[i].numel())), nll_base[i]
                    out[s][it["id"]] = {"id": it["id"], "category": it["category"], "kl": kl, "kl_first": first,
                                        "nll_base": nll_base[i], "nll_tuned": nll_t, "n_answer_tokens": int(ys[i].numel())}
                del tuned
            del base
        done = start + len(group)
        spent = time.time() - t0
        print(f"  [drift] {done}/{len(rows)} rows, {spent:.0f} s spent, {spent / done * (len(rows) - done):.0f} s left", flush=True)
    processor.tokenizer.padding_side = "left"
    if peft:
        set_scale(layers, 1.0)
    return out


def run(model_id: str = "Qwen/Qwen3.5-4B", adapter: str | None = None, adapter_subfolder: str | None = None,
        held: str | None = None, scales: tuple = (1.0,), batch: int = 16, max_new_tokens: int = 512, tag: str = "",
        repo: str = REPO, out_dir: str = "/content/keep", do_ship: bool = True, max_new_tokens_row: int = 800,
        max_len: int = 2048) -> dict:
    """Read an adapter against its base on the held general rows, at each scale; ship keep/<stamp>_<tag>/ (rows.jsonl: every
    row at every scale with every number and text; summary.json) to the project repo and return the summary. held defaults to
    the held rows of the general-row set of the same tag (hub:anchors/<tag>/held.json). The drift pass runs a quarter of
    `batch` rows at a time (it holds the logits of the whole vocabulary at every answer position); max_new_tokens caps the
    answers to the plain requests and max_new_tokens_row the row-form answers of read 3 (a whole row is longer)."""
    import torch
    held = held or f"hub:anchors/{tag}/held.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    out = os.path.join(out_dir, stamp)
    os.makedirs(out, exist_ok=True)
    env = preflight()
    peft = adapter is not None
    scales = tuple(float(s) for s in scales) if peft else (0.0,)      # no adapter: the base read against itself, once
    items = load_inputs(held)["items"]
    pictures = fetch_images([it for it in items if it.get("image_urls") or it.get("image")], os.path.join(out_dir, "images"))
    rows = [it for it in items if not (it.get("image_urls") or it.get("image")) or it["id"] in pictures]
    dropped = len(items) - len(rows)
    detail_rows = [it for it in rows if it["category"] == DETAIL_CATEGORY]
    plain_rows = [it for it in rows if it["category"] != DETAIL_CATEGORY]
    print(f"keep read {stamp}: {len(rows)} held rows ({len(detail_rows)} detail, {dropped} dropped for a missing picture); "
          f"{model_id}" + (f" + {adapter}/{adapter_subfolder or ''}" if peft else " (no adapter)") + f"; scales {list(scales)}", flush=True)
    t0 = time.time()
    processor, model = load_model(model_id, False)
    mark, kind = family.marker(model), family.model_type(model)
    layers = None
    if peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, token=token(), **({"subfolder": adapter_subfolder} if adapter_subfolder else {}))
        model.eval()
        layers = lora_layers(model)
        print(f"  adapter loaded beside the base, not merged ({len(layers)} LoRA layers)", flush=True)

    drift = drift_pass(processor, model, peft, layers, scales, rows, pictures, mark, kind, max(1, batch // 4), max_len)

    # read 2: the base's answers once, then the tuned model's at each scale
    with (model.disable_adapter() if peft else contextlib.nullcontext()):
        base_text, base_len = _generate_rows(processor, model, plain_rows, pictures, mark, max_new_tokens, batch, "base answers")
    full_user = prompts.full("medium")                 # the user turn train.conversation builds for the form "full"
    records = []
    per_scale = {}
    for s in scales:
        if peft:
            set_scale(layers, s)
            texts, lens = _generate_rows(processor, model, plain_rows, pictures, mark, max_new_tokens, batch, f"scale {s} answers")
        else:
            texts, lens = base_text, base_len
        take, detail_recs = [], []
        by_id = {}
        for it, text, n, btext, bn in zip(plain_rows, texts, lens, base_text, base_len):
            text, btext = text or "", btext or ""
            rec = {"takes_row_form": takes_row_form(text), "ended": n < max_new_tokens,
                   "length_ratio": len(text) / len(btext) if btext else None, "f1": token_f1(text, btext)}
            take.append({"category": it["category"], **rec})
            by_id[it["id"]] = {**rec, "text": text, "new_tokens": n, "base_text": btext, "base_new_tokens": bn,
                               "base_takes_row_form": takes_row_form(btext), "base_ended": bn < max_new_tokens}
        # read 3: the whole row asked of each held picture, the base's detail as the reference
        row_texts, row_lens = _generate_rows(processor, model, detail_rows, pictures, mark, max_new_tokens_row, batch,
                                             f"scale {s} rows", form="full")
        for it, text, n in zip(detail_rows, row_texts, row_lens):
            text = text or ""
            rec = {"recall": recall(it["answer"], text), "row_chars": len(text), "row_tokens": n, "ended": n < max_new_tokens_row}
            detail_recs.append(rec)
            by_id[it["id"]] = {**rec, "row_text": text, "reference": it["answer"], "ask": full_user}
        for it in rows:
            records.append({"scale": s, "id": it["id"], "category": it["category"], "source": it.get("source"), "seed_id": it.get("seed_id"),
                            "prompt": it["prompt"], "answer": it["answer"], **drift[s][it["id"]], **by_id.get(it["id"], {})})
        per_scale[str(s)] = {"drift": drift_summary(list(drift[s].values())), "takeover": takeover_summary(take),
                             "detail": detail_summary(detail_recs)}
        t = per_scale[str(s)]
        print(f"  scale {s}: KL {t['drift']['all']['kl_mean']:.5f}, perplexity ratio {t['drift']['all']['perplexity_ratio']:.4f}, "
              f"row form taken {t['takeover']['all']['takes_row_form']}, detail recall {t['detail']['recall']}", flush=True)
    if peft:
        set_scale(layers, 1.0)
    base_take = [{"category": it["category"], "takes_row_form": takes_row_form(t or ""), "ended": n < max_new_tokens,
                  "length_ratio": 1.0, "f1": 1.0} for it, t, n in zip(plain_rows, base_text, base_len)]
    summary = {"package": __version__, "model": model_id, "adapter": adapter, "adapter_subfolder": adapter_subfolder, "held": held,
               "tag": tag, "scales": list(scales), "rows": len(rows), "detail_rows": len(detail_rows), "dropped_no_picture": dropped,
               "batch": batch, "max_new_tokens": max_new_tokens, "max_new_tokens_row": max_new_tokens_row,
               "base": {"takeover": takeover_summary(base_take)}, "by_scale": per_scale,
               "minutes": round((time.time() - t0) / 60, 1), "env": env}
    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if do_ship:
        print("  shipped:", ship(out, f"keep/{stamp}", f"keep read {stamp}: {len(rows)} rows at scales {list(scales)}", repo), flush=True)
    summary["path_in_repo"] = f"keep/{stamp}"
    return summary
