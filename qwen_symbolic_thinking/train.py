"""The fine-tune: a LoRA on a vision-language model (Qwen3.5, Gemma 4), taught the rows under their prompt forms.

    from qwen_symbolic_thinking.train import run
    run(model_id="Qwen/Qwen3.5-9B", inputs=["hub:sft/pictures_v2.json"], epochs=3, tag="pictures")

Each item becomes one conversation per form it carries: the picture (or the specification) with the form's one-line prompt as
the user turn, the row's chain (short) or the whole row (full) as the assistant turn. The loss is on the assistant turn only.
Plain Adam with a cosine schedule and warmup. The adapter ships to the project repo during training and at the end, under
lora/<model>/r<rank>/<stamp>_<tag>/. Rank 0 is a full fine-tune: the language model's weights train in fp32 under bf16
autocast (the vision tower frozen, as for the LoRA), and the model ships in bf16 with its processor under
full/<model>/<stamp>_<tag>/final, loadable afterwards by the model id hub:full/<model>/<stamp>_<tag>/final."""
from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
from datetime import datetime, timezone

from . import REPO, __version__, chain, family, keep, prompts
from . import resume as saved_state
from .hub import fetch_folder, model_short, resolve_model, ship
from .resume import Counts
from .extract import fetch_images                # by url list or by locator (a pretrain shard row, a Hub file, a zip member)
from .screen import encode, load_inputs, load_vlm, preflight
from .template import stop_ids, turn_end                 # the template's turn end (training examples end with it) and the stop ids

# The language model's projections, by name (the vision side, under `visual` in Qwen3.5 or `embed_vision` in Gemma 4, is left
# alone): the attention layers' q/k/v/o, every layer's MLP, and Qwen3.5's linear-attention token/value projections (in_proj_qkv,
# in_proj_z, out_proj; in_proj_qkvz in the Qwen3-Next naming). The tiny gate projections (in_proj_a, in_proj_b) are not adapted.
# Gemma 4's language model (model.language_model.layers.N.self_attn / .mlp) carries the same seven names, so one pattern serves both;
# GLM-4V's MLP fuses the gate and up projections into one (gate_up_proj), named here so the whole MLP adapts there too.
# Two families keep their language model outside a `language_model` module, so their layers are named from the root: Phi-4-
# reasoning-vision (model.layers.N: a fused qkv_proj, o_proj, gate_up_proj, down_proj) and Molmo2 (model.transformer.blocks.N:
# att_proj, attn_out, ff_proj, ff_out); their vision towers (model.vision_tower..., model.vision_backbone...) stay out.
TARGETS = (r"(.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|gate_up_proj|down_proj|in_proj_qkv|in_proj_z|"
           r"in_proj_qkvz|out_proj)"
           r"|model\.layers\.\d+\.(self_attn\.(qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))"
           r"|model\.transformer\.blocks\.\d+\.(self_attn\.(att_proj|attn_out)|mlp\.(ff_proj|ff_out)))$")
PICTURE_FORMS = ("short", "full")


def overrun(attention_mask, max_len: int) -> tuple:
    """(examples, examples longer than max_len, the longest example's tokens) for one batch's attention mask. An example longer
    than max_len loses the tail of its answer and its turn end to the cut, so a family whose pictures take more tokens than
    Qwen's is read for it in the training summary."""
    real = [int(x) for x in attention_mask.sum(dim=1).tolist()]
    return len(real), sum(1 for x in real if x > max_len), max(real, default=0)
SPEC_FORMS = ("spec_short", "spec_full")


def conversation(it: dict, form: str, image=None, system: str | None = None, mark: str | None = None,
                 keys: str = "record") -> tuple:
    """(the messages, the answer text) of one item under one form: the user turn, after a system turn when a system prompt is given.
    With a picture marker (family.marker) the user content is one string. keys (chain.KEYS): how the answer's question keys and
    a specification's records are written, the same way on both sides. The form "anchor" is a general row (keep.py)."""
    depth = it.get("depth")       # a general row has none
    if form == "short":
        user, answer = family.content(prompts.short(depth), image, mark), chain.render(it, keys=keys)
    elif form == "full":
        user, answer = family.content(prompts.full(depth), image, mark), chain.render(it, full=True, keys=keys)
    elif form == "spec_short":
        user, answer = family.content(prompts.spec_short(depth, it["spec"], keys), None, mark), chain.render(it, keys=keys)
    elif form == "spec_full":
        user, answer = family.content(prompts.spec_full(depth, it["spec"], keys), None, mark), chain.render(it, full=True, keys=keys)
    elif form == "anchor":     # a general row (keep.py): its own request, its picture when it has one, the base model's answer
        user, answer = family.content(it["prompt"], image, mark), it["answer"]
    else:
        raise ValueError(f"unknown form {form!r}")
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}], answer


def examples_of(items: list) -> list:
    """(item, form) pairs: every form an item carries; a picture item without forms is taught the short form, a specification
    item the spec_short form."""
    out = []
    for it in items:
        forms = it.get("forms") or (["short"] if it["kind"] == "image" else ["spec_short"])
        for form in forms:
            if (form in PICTURE_FORMS) != (it["kind"] == "image"):
                raise ValueError(f"{it['id']}: form {form} does not fit a {it['kind']} item")
            out.append((it, form))
    return out


class Collator:
    """A batch from (item, form) pairs: the prompt rendered through the chat template (thinking off), the answer and the turn end
    appended, the images beside; labels cover the answer only. The prompt length of each pair is cached across epochs."""

    def __init__(self, processor, pictures: dict, max_len: int, system: str | None = None, mark: str | None = None,
                 kind: str = "", keys: str = "record"):
        self.processor, self.pictures, self.max_len, self.prefix_len, self.system = processor, pictures, max_len, {}, system
        self.mark, self.kind = mark, kind   # the picture marker and the family key (family.marker, family.model_type)
        self.keys = keys                    # how question keys and records are written (chain.KEYS)
        self.turn_end = turn_end(processor)
        self.counts = Counts()   # per batch: examples collated, examples longer than max_len (tail cut), the longest (resume.py)

    @property
    def collated(self) -> int:
        return self.counts.collated

    @property
    def cut(self) -> int:
        return self.counts.cut

    @property
    def longest(self) -> int:
        return self.counts.longest

    def prefix_length(self, key: tuple, prefix: str, image) -> int:
        if key not in self.prefix_len:
            self.prefix_len[key] = int(encode(self.processor, [prefix], [image])["input_ids"].shape[1])
        return self.prefix_len[key]

    def __call__(self, pairs: list) -> dict:
        texts, images, lengths = [], [], []
        for it, form in pairs:
            image = self.pictures.get(it["id"]) if it["kind"] in ("image", "anchor") else None   # a general row may have none
            messages, answer = conversation(it, form, image, self.system, self.mark, self.keys)
            prefix = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                        **family.thinking_kwargs(self.processor, False))
            texts.append(prefix + answer + self.turn_end)
            images.append(image)                      # one slot per text: None for a specification row
            lengths.append(self.prefix_length((it["id"], form), prefix, image))
        batch = encode(self.processor, texts, family.fill_images(self.kind, images), padding=True)
        family.fix_padding(self.kind, batch, self.processor.tokenizer.pad_token_id)
        self.counts.add(*overrun(batch["attention_mask"], self.max_len))
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, n in enumerate(lengths):
            labels[i, :n] = -100
        if labels.shape[1] > self.max_len:  # the longest sequence sets the batch; the overrun is cut from the right (the answer's tail)
            for k in list(batch):
                if hasattr(batch[k], "shape") and batch[k].dim() >= 2 and batch[k].shape[1] == labels.shape[1]:
                    batch[k] = batch[k][:, :self.max_len]
            labels = labels[:, :self.max_len]
        batch["labels"] = labels
        return batch


class Pairs:
    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def trains_in_full(name: str) -> bool:
    """Whether a parameter trains in a full fine-tune: the language model (its embeddings included) and the output head; the
    vision tower and its merger stay frozen, as under the LoRA."""
    return "language_model" in name or name.startswith("lm_head") or ".lm_head." in name


def lora_report(model) -> dict:
    """How many modules each target name matched; the names that matched nothing are reported, not assumed."""
    from collections import Counter
    hits = Counter()
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            hits[name.rsplit(".", 1)[-1]] += 1
    return dict(hits)


def load_items(inputs) -> list:
    """The items of one input file or of several, in order."""
    out = []
    for path in ([inputs] if isinstance(inputs, str) else list(inputs)):
        out += load_inputs(path)["items"]
    return out


def adapter_source(spec: str, repo: str = REPO) -> tuple:
    """Where an adapter to continue from lives: 'hub:lora/<base>/r32/<stamp>/final' -> (the project repo, that subfolder);
    a local folder or a Hub repo id -> (itself, no subfolder)."""
    if spec.startswith("hub:"):
        return repo, spec[4:].strip("/")
    return spec, None


def name_base(folder: str, local: str, model_id: str, repo: str = REPO) -> list:
    """Make a saved adapter name its base model by the Hub id. A model whose repository code runs loads from a local snapshot
    (a 'hub:' model from its downloaded folder), and PEFT writes that folder into adapter_config.json and the card's metadata;
    the Hub refuses a card whose base model is a local path, so every ship of such an adapter failed. Returns the files changed."""
    hub_id = repo if model_id.startswith("hub:") else model_id
    if not local or local == hub_id:
        return []
    changed = []
    for name in ("adapter_config.json", "README.md"):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            text = fh.read()
        new = text.replace(json.dumps(local)[1:-1], json.dumps(hub_id)[1:-1]).replace(local, hub_id)   # escaped (JSON), then plain
        if new != text:
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(new)
            changed.append(name)
    return changed


def restore(state: dict, collate, anchor_collate, data_collator, keep_empty=None) -> None:
    """Take back the run's own part of a saved state (resume.py) before the Trainer reads its checkpoint: the collators' counts
    as they stood after the batches trained, the general rows' draws replayed for those batches, the keeping terms' counts of
    empty micro-batches. (The progress log and the keeping terms' running sums are read back where run() makes them.)"""
    collate.counts.restore(state["collate"])
    if anchor_collate is not None and state.get("anchor_collate") is not None:
        anchor_collate.counts.restore(state["anchor_collate"])
    if isinstance(data_collator, keep.PairedCollator):
        data_collator.advance(state["micro_batches"])
    if keep_empty is not None:
        keep_empty.clear()
        keep_empty.update(state.get("keep_empty") or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def carried(found: dict | None) -> tuple:
    """(the progress log, the keeping terms' running sums, the seconds trained) a run starts from: empty for a new run, a saved
    state's for a resumed one (resume.py)."""
    if found is None:
        return [], {}, 0.0
    return list(found.get("log") or []), dict(found.get("keep_stats") or {}), float(found.get("train_s") or 0.0)


class RunState:
    """The run's own part of a saved training state (resume.json, beside the Trainer's checkpoint; resume.py). Made from the
    run's pieces, it takes a found state back into them (restore) and writes the record at each save (record). The Trainer is
    attached once it exists (its loss sum is read at a save), and the saver once made (its saves are listed)."""

    def __init__(self, found, *, tag, stamp, path_in_repo, settings, pairs, examples_print, general_print, batch, grad_accum,
                 total, collate, anchor_collate, anchor_batch, data_collator, log, keep_stats, keep_empty, progress, t0):
        self.tag, self.stamp, self.path_in_repo, self.settings = tag, stamp, path_in_repo, settings
        self.pairs, self.examples_print, self.general_print = pairs, examples_print, general_print
        self.batch, self.grad_accum, self.total = batch, grad_accum, total
        self.collate, self.anchor_collate, self.anchor_batch = collate, anchor_collate, anchor_batch
        self.log, self.keep_stats, self.keep_empty, self.progress, self.t0 = log, keep_stats, keep_empty, progress, t0
        self.trainer = self.saver = None
        self.resumed = list(found.get("resumed") or []) if found else []
        self.saves_before = list(found.get("saves") or []) if found else []
        self.loss_before = float(found.get("loss_sum") or 0.0) if found else 0.0
        self.minutes_before = float(found.get("minutes") or 0.0) if found else 0.0
        if found is not None:
            self.resumed.append({"step": found["step"], "at": _now(), "package": __version__})
            restore(found, collate, anchor_collate, data_collator, keep_empty)

    def loss_sum(self) -> float:
        """The training loss summed over the steps trained, on every machine (on this one, the Trainer's own sum)."""
        return self.loss_before + float(getattr(self.trainer, "_total_loss_scalar", 0.0) or 0.0)

    def saves(self, step: int | None = None, at: str | None = None) -> list:
        """The saves made, on every machine, and the one being made (step, at) when given."""
        out = self.saves_before + (list(self.saver.saves) if self.saver is not None else [])
        return out + ([{"step": step, "at": at}] if step is not None else [])

    def record(self, state) -> dict:
        n = saved_state.consumed(state.global_step, len(self.pairs), self.batch, self.grad_accum)
        now = _now()
        anchors = self.anchor_collate is not None and self.anchor_batch > 0
        return {"version": 1, "package": __version__, "tag": self.tag, "stamp": self.stamp, "path_in_repo": self.path_in_repo,
                "settings": self.settings, "examples": self.examples_print, "general_rows": self.general_print,
                "n_examples": len(self.pairs), "step": state.global_step, "max_steps": self.total, "micro_batches": n,
                "collate": self.collate.counts.upto(n), "anchor_collate": self.anchor_collate.counts.upto(n) if anchors else None,
                "log": self.log, "keep_stats": dict(self.keep_stats), "keep_empty": dict(self.keep_empty or {}),
                "loss_sum": self.loss_sum(), "train_s": round(self.progress.train_s(), 1),
                "minutes": round(self.minutes_before + (time.time() - self.t0) / 60, 2), "resumed": self.resumed,
                "saves": self.saves(state.global_step, now), "saved_at": now}


def progress(total: int, first_line: str, log: list, keep_stats: dict, at_log=None, spent_before: float = 0.0):
    """The callback that prints the training's progress and keeps its log: at every log step a record {step, spent_s, left_s,
    s_per_step, loss, learning_rate, and the keeping terms' means since the last log line (keep.keep_trainer's running sums)}
    goes into log; at_log(step), when given, runs after it (the step ships). A resumed run passes the seconds its earlier
    machines trained (spent_before), and its pace is read from the steps this machine trained."""
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def __init__(self):
            self.t0, self.step0 = None, 0

        def train_s(self) -> float:
            return spent_before + (time.time() - self.t0 if self.t0 is not None else 0.0)

        def on_train_begin(self, args, state, control, **kw):
            self.t0, self.step0 = time.time(), state.global_step
            print(f"  {first_line}" + (f"; continuing at step {self.step0}" if self.step0 else ""), flush=True)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None and "loss" not in logs:     # the end-of-training summary log (train_loss, runtime): not a step
                return
            spent = time.time() - self.t0
            done = max(state.global_step - self.step0, 1)
            rec = {"step": state.global_step, "spent_s": round(spent_before + spent), "left_s": round(spent / done * (total - state.global_step)),
                   "s_per_step": round(spent / done, 2), **{k: v for k, v in (logs or {}).items() if k in ("loss", "learning_rate")}}
            n = keep_stats.pop("n", 0)
            terms = {k: round(keep_stats.pop(k) / n, 5) for k in list(keep_stats)} if n else {}
            keep_stats.clear()
            rec.update(terms)
            log.append(rec)
            print(f"  step {rec['step']}/{total}: loss {rec.get('loss', float('nan')):.4f}"
                  + "".join(f", {k} {v}" for k, v in terms.items())
                  + f", {rec['spent_s']} s spent, {rec['left_s']} s left, {rec['s_per_step']} s/step", flush=True)
            if at_log is not None:
                at_log(state.global_step)

    return Progress()


def run(model_id: str = "Qwen/Qwen3.5-9B", inputs="hub:sft/sft_v1.json", out_dir: str = "/content/train",
        rank: int = 32, alpha: int = 64, dropout: float = 0.05, lr: float = 2e-4, epochs: float = 1.0, batch: int = 8,
        grad_accum: int = 2, max_len: int = 2048, warmup: float = 0.03, n_items: int | None = None, ship_every: int | None = None,
        repo: str = REPO, do_ship: bool = True, seed: int = 20261030, tag: str = "", init_adapter: str | None = None,
        system: str | None = None, trust_remote_code: bool = False, revision: str | None = None,
        targets: str | None = None, image_settings: dict | None = None, keys: str = "record",
        max_steps: int | None = None, anchors=None, anchor_batch: int = 4, general_kl: float = 0.0,
        target_kl: float = 0.0, rehearsal: float = 0.0, writers: list | None = None, seed_adapter: bool = False,
        resume: bool = False, resume_every_h: float = 2.0, resume_before_h: float | None = 23.0) -> dict:
    """Train the adapter; return {adapter_dir, path_in_repo, steps, s_per_step, train_loss, ...}. With init_adapter (see
    adapter_source) training continues from that trained adapter instead of a fresh LoRA; its rank and alpha must be the ones passed.
    With system (see prompts.system) every example carries that system turn before its user turn. Rank 0 is a full fine-tune of
    the language model (fp32 weights under bf16 autocast; the vision tower frozen); adapter_dir then holds the whole model in bf16
    with its processor. ship_every: the adapter ships every so many steps (100 by default; a full model ships only at the end).
    keys (chain.KEYS): how the answers' question keys and the specification rows' records are written ("record", the dataset's
    names, or "relation": relation_a, relation_b, ..., thing_position). max_steps: the run's length in steps instead of epochs (the
    cosine schedule and its warmup span those steps).

    Keeping the base model's behavior (keep.py; a LoRA only): general_kl weighs KL(base || tuned) on general rows (anchors: one
    input file or a list, items of kind "anchor" answered by the base model; anchor_batch of them ride beside every micro-batch),
    target_kl weighs the target-masked KL on the training rows, rehearsal weighs cross-entropy on the general rows. With all three
    at zero the plain Trainer runs. writers: the writers whose text this run may learn from; an item whose `writer` is missing or
    not listed stops the run before anything loads (None: not checked). seed_adapter: seed the global generator before the LoRA
    is built, so the adapter's starting matrices follow the seed (by default they do not: the seed reaches the Trainer only).

    resume (resume.py; a LoRA only, found by its tag): the whole training state saves to the repo under resume/<tag>/ every
    resume_every_h hours and once when the machine's age passes resume_before_h (None: no such save), and a run of the same tag
    that finds a saved state continues from it, in the same folder, on the same path; once the training has ended and shipped,
    a run of the same tag returns that run's summary and its shipped adapter without training."""
    if keys not in chain.KEYS:
        raise ValueError(f"unknown keys {keys!r}: one of {chain.KEYS}")
    full = rank == 0
    keep_on = general_kl > 0 or target_kl > 0 or rehearsal > 0
    if keep_on and full:
        raise ValueError("general_kl, target_kl and rehearsal read the base model through the adapter's off switch: a LoRA only")
    if (general_kl > 0 or rehearsal > 0) and not anchors:
        raise ValueError("general_kl and rehearsal are read on general rows: pass anchors")
    if resume and full:
        raise ValueError("resume saves a LoRA's training state; a full fine-tune's would be the whole model and its optimizer")
    if resume and not tag:
        raise ValueError("resume finds a run's saved state by its tag: pass tag (a queue run passes its name)")
    settings = saved_state.normalized({   # what defines the run: a resumed run must match the saved state's
        "model_id": model_id, "inputs": inputs, "rank": rank, "alpha": alpha, "dropout": dropout, "lr": lr, "epochs": epochs,
        "batch": batch, "grad_accum": grad_accum, "max_len": max_len, "warmup": warmup, "n_items": n_items, "seed": seed,
        "init_adapter": init_adapter, "system": system, "trust_remote_code": trust_remote_code, "revision": revision,
        "targets": targets, "image_settings": image_settings, "keys": keys, "max_steps": max_steps, "anchors": anchors,
        "anchor_batch": anchor_batch, "general_kl": general_kl, "target_kl": target_kl, "rehearsal": rehearsal,
        "writers": writers, "seed_adapter": seed_adapter})
    store = saved_state.HubStore(repo) if resume else None
    found = saved_state.load(store, tag, os.path.join(out_dir, "resume", tag)) if resume else None
    if found is not None:
        saved_state.check_settings(found["settings"], settings)
        if found.get("finished"):
            summary = dict(found["summary"])
            summary["adapter_dir"] = fetch_folder(f"{found['path_in_repo']}/final", repo)
            print(f"train {tag}: its training ended before ({summary.get('steps')} steps, shipped to {found['path_in_repo']}/final); "
                  f"the shipped adapter is read back and nothing trains", flush=True)
            return summary
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments
    if not full:
        from .checks import peft_beside_torchao
        peft_check = peft_beside_torchao()
        assert peft_check["peft_lora_ok"], f"PEFT cannot build a LoRA here: {peft_check.get('error')}; {peft_check.get('advice', '')}"
        from peft import LoraConfig, TaskType, get_peft_model
    if ship_every is None:
        ship_every = 0 if full else 100
    if found is not None:        # the first run's stamp: the step ships and the final adapter land in its folder
        stamp = found["stamp"]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    short = model_short(model_id)
    path_in_repo = f"full/{short}/{stamp}" if full else f"lora/{short}/r{rank}/{stamp}"
    run_dir = os.path.join(out_dir, f"{short}_{'full' if full else f'r{rank}'}_{stamp}")
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    env = preflight()
    src = resolve_model(model_id, repo)
    items = load_items(inputs)[:n_items]
    anchor_items = load_items(anchors) if (anchors and (general_kl > 0 or rehearsal > 0)) else []
    writers_seen = keep.check_writers(items + anchor_items, writers)
    if writers is not None:
        print(f"writers accepted {sorted(writers)}; items by writer {writers_seen}", flush=True)
    t0 = time.time()
    why_missing = {}              # the picture items whose picture did not load, by reason (they leave the run)
    picture_items = [it for it in items if it["kind"] == "image"]
    pictures = fetch_images(picture_items, os.path.join(out_dir, "images"), failures=why_missing)
    missing = sorted(it["id"] for it in picture_items if it["id"] not in pictures)
    pictures_missing = {"n": len(missing), "by_reason": {}, "ids": missing[:200]}
    for m in missing:
        reason = why_missing.get(m, "?")
        pictures_missing["by_reason"][reason] = pictures_missing["by_reason"].get(reason, 0) + 1
    items = [it for it in items if it["kind"] != "image" or it["id"] in pictures]
    if anchor_items:
        with_picture = [a for a in anchor_items if a.get("image_urls") or a.get("image")]
        pictures.update(fetch_images(with_picture, os.path.join(out_dir, "images")))
        anchor_items = [a for a in anchor_items if not (a.get("image_urls") or a.get("image")) or a["id"] in pictures]
        print(f"general rows: {len(anchor_items)} ({sum(1 for a in anchor_items if a['id'] in pictures)} with a picture), "
              f"{anchor_batch} beside every micro-batch; general_kl {general_kl}, rehearsal {rehearsal}", flush=True)
    pairs = examples_of(items)
    random.Random(seed).shuffle(pairs)
    examples_print = saved_state.fingerprint(pairs)
    general_print = saved_state.fingerprint([(a, "anchor") for a in anchor_items]) if anchor_items else None
    if found is not None:
        saved_state.check_examples(found, examples_print, general_print, len(pairs))
        print(f"train {stamp}: continuing from its state at step {found['step']} of {found['max_steps']} (saved "
              f"{found['saved_at']} by package {found['package']}); the same {len(pairs)} examples in the same order", flush=True)
    n_img = sum(1 for it in items if it["kind"] == "image")
    forms = {}
    for _, form in pairs:
        forms[form] = forms.get(form, 0) + 1
    print(f"train {stamp}: {len(items)} items ({n_img} with a picture), {len(pairs)} examples by form {forms}; images fetched in "
          f"{time.time() - t0:.0f} s; pictures that did not load {pictures_missing['n']} {pictures_missing['by_reason']}", flush=True)
    if keys != "record":
        print(f"  question keys and records written as {keys!r} keys (thing_position; relation_a, relation_b, relation_a_b)", flush=True)

    if trust_remote_code and not os.path.isdir(src):   # the repository's code runs from a snapshot at the pinned revision
        from huggingface_hub import snapshot_download
        from .hub import token as hub_token
        src = snapshot_download(src, revision=revision, token=hub_token() or None)
    base_dir = src            # where the weights load from; the saved adapter names the Hub id instead (name_base)
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=trust_remote_code)
    processor.tokenizer.padding_side = "right"
    system_text = prompts.system(system)
    system_tokens = len(processor.tokenizer(system_text)["input_ids"]) if system_text else 0
    if system_text:
        print(f"system prompt {system!r}: {system_tokens} tokens before every example", flush=True)
    model = load_vlm(src, trust_remote_code=trust_remote_code, dtype=torch.float32 if full else torch.bfloat16, device_map="cuda")
    family.prepare_processor(processor, model)
    note = family.prepare_model(model)
    if note:
        print(f"  {note}", flush=True)
    family.apply_image_settings(processor, image_settings)
    mark = family.marker(model)
    targets = targets or TARGETS
    if trust_remote_code or mark or image_settings:
        print(f"  {family.model_type(model)}" + (f": picture marker {mark}" if mark else "")
              + (f"; repository code at {revision}" if trust_remote_code else "")
              + (f"; image processor {image_settings}" if image_settings else ""), flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    input_grads = family.input_grads(model)
    if input_grads:
        model.enable_input_require_grads()
    if full:
        n_train = 0
        for name, p in model.named_parameters():
            p.requires_grad = trains_in_full(name)
            n_train += p.numel() if p.requires_grad else 0
        report = {"full": n_train}
        print(f"full fine-tune: {n_train / 1e9:.2f}B parameters train (the language model), the vision tower frozen", flush=True)
    elif init_adapter:
        from peft import PeftModel
        from .hub import token
        src, sub = adapter_source(init_adapter, repo)
        print(f"continuing from the adapter {src}" + (f" / {sub}" if sub else ""), flush=True)
        model = PeftModel.from_pretrained(model, src, is_trainable=True, token=token(), **({"subfolder": sub} if sub else {}))
        cfg = model.peft_config["default"]
        assert (cfg.r, cfg.lora_alpha) == (rank, alpha), f"the adapter is r{cfg.r} alpha {cfg.lora_alpha}; pass its own rank and alpha"
    else:
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets,
                            task_type=TaskType.CAUSAL_LM)
        if seed_adapter:     # the starting A matrices follow the seed (otherwise every process draws its own)
            from transformers import set_seed
            set_seed(seed)
        model = get_peft_model(model, config)
    if not full and not input_grads and family.drop_input_grads(model):   # PEFT re-installs the hook on a checkpointing model
        print("  the input-embedding grad hook PEFT installed is removed (this family adds its pictures to that output in place)", flush=True)
    if not full:
        report = lora_report(model)
        print("LoRA modules by name:", report, flush=True)
        assert report, "no module matched the LoRA targets; read the model's module names"
        model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(pairs) / (batch * grad_accum))
    total = max_steps or math.ceil(steps_per_epoch * epochs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999))        # plain Adam, no weight decay
    wanted = dict(output_dir=run_dir, per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
                  num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=math.ceil(total * warmup),
                  bf16=True, logging_steps=10, save_strategy="no", report_to=[], remove_unused_columns=False,
                  dataloader_num_workers=0, seed=seed, gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False})
    if max_steps:
        wanted["max_steps"] = max_steps                       # the Trainer stops here, whatever num_train_epochs says
    allowed = set(inspect.signature(TrainingArguments).parameters)
    dropped = [k for k in wanted if k not in allowed]
    if dropped:
        print("  this transformers' TrainingArguments lacks", dropped, "(left out)", flush=True)
    args = TrainingArguments(**{k: v for k, v in wanted.items() if k in allowed})
    log, keep_stats, spent_before = carried(found)   # the log, the keeping terms' running sums between two log lines, the seconds trained

    def at_log(step: int) -> None:      # the step ships
        if do_ship and ship_every and step and step % ship_every == 0 and step < total:
            model.save_pretrained(adapter_dir)
            name_base(adapter_dir, base_dir, model_id, repo)
            ship(adapter_dir, f"{path_in_repo}/step_{step}", f"{short} r{rank} {stamp}: step {step}", repo)

    first_line = (f"{total} steps of {batch} x {grad_accum} sequences; "
                  + (f"max_steps {max_steps} ({total / steps_per_epoch:.2f} of an epoch)" if max_steps else f"{epochs} epoch(s)"))
    progress_cb = progress(total, first_line, log, keep_stats, at_log, spent_before)

    collate = Collator(processor, pictures, max_len, system_text, mark, family.model_type(model), keys)
    pic = next((p for p in pairs if p[1] in PICTURE_FORMS and pictures.get(p[0]["id"]) is not None), None)
    spc = next((p for p in pairs if p[1] in SPEC_FORMS), None)
    if pic and spc:   # a mixed batch encodes before the first step (a family whose processor takes pictures differently fails here, not at step 1)
        probe = collate([pic, spc])
        print(f"  a mixed batch (a picture row, a specification row) encodes: input_ids {tuple(probe['input_ids'].shape)}", flush=True)
        collate.counts.clear()   # the counts are the training batches'
    data_collator, anchor_collate = collate, None
    if anchor_items:
        anchor_collate = Collator(processor, pictures, max_len, system_text, mark, family.model_type(model), keys)
        data_collator = keep.PairedCollator(collate, anchor_items, anchor_batch, seed, anchor_collate)
    trainer_class = keep.keep_trainer(general_kl, target_kl, rehearsal, keep_stats) if keep_on else Trainer
    callbacks, rs = [progress_cb], None
    if resume:
        rs = RunState(found, tag=tag, stamp=stamp, path_in_repo=path_in_repo, settings=settings, pairs=pairs,
                      examples_print=examples_print, general_print=general_print, batch=batch, grad_accum=grad_accum, total=total,
                      collate=collate, anchor_collate=anchor_collate, anchor_batch=anchor_batch, data_collator=data_collator,
                      log=log, keep_stats=keep_stats, keep_empty=getattr(trainer_class, "keep_empty", None),
                      progress=progress_cb, t0=t0)
        rs.saver = saved_state.saver(store, tag, saved_state.Schedule(resume_every_h, resume_before_h), rs.record)
        callbacks.append(rs.saver)
        age = saved_state.machine_age_h()
        print(f"  the training state saves to {repo} under {saved_state.ROOT}/{tag}/ every {resume_every_h} h"
              + (f" and once after the machine's {resume_before_h} h (it is {age:.1f} h old)" if resume_before_h and age is not None else ""),
              flush=True)
    trainer = trainer_class(model=model, args=args, train_dataset=Pairs(pairs), data_collator=data_collator,
                            optimizers=(optimizer, None), callbacks=callbacks)
    if rs is not None:
        rs.trainer = trainer
    result = trainer.train(resume_from_checkpoint=found["dir"] if found else None)
    if found is not None and hasattr(trainer, "_total_loss_scalar"):    # the whole run's mean loss, over every machine
        train_loss = rs.loss_sum() / max(result.global_step, 1)
    else:
        train_loss = result.training_loss
    minutes_before = rs.minutes_before if rs is not None else 0.0
    if full:   # the whole model in bf16, with its processor, so the folder loads on its own
        from transformers import GenerationConfig
        stops = stop_ids(processor, model)      # the saved model stops on the turn end under a plain generate(), not only under ours
        gen = model.generation_config.to_dict() if model.generation_config is not None else {}
        gen.update(eos_token_id=stops, _from_model_config=False)   # not re-derived from the model config at generation time
        model.generation_config = GenerationConfig.from_dict(gen)
        print(f"  generation stops on {stops}", flush=True)
        model.to(torch.bfloat16).save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(adapter_dir)
    else:
        model.save_pretrained(adapter_dir)
    named = name_base(adapter_dir, base_dir, model_id, repo)
    if named:
        print(f"  {', '.join(named)} name the Hub id as the base (the weights loaded from {base_dir})", flush=True)
    summary = {"package": __version__, "model": model_id, "full": full, "tag": tag, "rank": rank, "alpha": alpha, "dropout": dropout, "lr": lr,
               "epochs": epochs, "batch": batch, "grad_accum": grad_accum, "max_len": max_len, "items": len(items),
               "items_with_picture": n_img, "pictures_missing": pictures_missing, "examples": len(pairs), "examples_by_form": forms,
               "steps": result.global_step,
               "train_loss": train_loss, "s_per_step": round((minutes_before * 60 + time.time() - t0) / max(result.global_step, 1), 2),
               "train_min": round(minutes_before + (time.time() - t0) / 60, 1), "lora_modules": report, "targets": targets, "inputs": inputs,
               "model_type": family.model_type(model), "trust_remote_code": trust_remote_code, "revision": revision,
               "image_settings": image_settings,
               "init_adapter": init_adapter, "system": system, "system_tokens": system_tokens, "path_in_repo": path_in_repo,
               "keys": keys, "max_steps": max_steps,
               "keep": {"general_kl": general_kl, "target_kl": target_kl, "rehearsal": rehearsal, "anchors": anchors,
                        "anchor_items": len(anchor_items), "anchor_batch": anchor_batch if anchor_items else 0,
                        "anchor_examples_cut_at_max_len": anchor_collate.cut if anchor_collate else 0,
                        "empty_batches": dict(getattr(trainer_class, "keep_empty", {}))},
               "writers": writers, "items_by_writer": writers_seen, "seed_adapter": seed_adapter,
               "collated_examples": collate.collated, "examples_cut_at_max_len": collate.cut, "longest_example_tokens": collate.longest,
               "resume": ({"every_h": resume_every_h, "before_h": resume_before_h, "examples_fingerprint": examples_print,
                           "resumed": rs.resumed, "saves": rs.saves()} if rs is not None else None),
               "env": env, "log": log}
    print(f"  examples longer than max_len {max_len} (answer tail cut): {collate.cut} of {collate.collated} collated "
          f"(the longest {collate.longest} tokens)", flush=True)
    if summary["keep"]["empty_batches"]:
        print(f"  micro-batches where a keeping term had no answer position to act on: {summary['keep']['empty_batches']}", flush=True)
    with open(os.path.join(adapter_dir, "train_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\ntrained in {summary['train_min']} min: {summary['steps']} steps, loss {summary['train_loss']:.4f}, "
          f"{summary['s_per_step']} s/step", flush=True)
    if do_ship:
        url = ship(adapter_dir, f"{path_in_repo}/final", f"{short} {'full' if full else f'r{rank}'} {stamp}: final ({summary['steps']} steps)", repo)
        summary["shipped"] = url is not None
        print(f"  shipped: {url}" if url else f"  NOT SHIPPED: the trained folder stays at {adapter_dir} (ship it by hand)", flush=True)
    if resume and summary.get("shipped"):    # the saved state gives way to the summary: a later run of this tag trains nothing
        try:
            store.finish(tag, {"finished": True, "tag": tag, "stamp": stamp, "path_in_repo": path_in_repo, "settings": settings,
                               "summary": summary, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        except Exception as e:      # the shipped adapter stands either way; a later run of this tag would train its tail again
            print(f"  the saved state of {tag} was not marked finished ({type(e).__name__}: {str(e)[:200]})", flush=True)
    summary["adapter_dir"] = adapter_dir
    del trainer, model, processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return summary
