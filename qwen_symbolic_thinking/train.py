"""The fine-tune: a LoRA on a Qwen3.5 vision-language model, taught the rows under their prompt forms.

    from qwen_symbolic_thinking.train import run
    run(model_id="Qwen/Qwen3.5-9B", inputs=["hub:sft/pictures_v2.json"], epochs=3, tag="pictures")

Each item becomes one conversation per form it carries: the picture (or the specification) with the form's one-line prompt as
the user turn, the row's chain (short) or the whole row (full) as the assistant turn. The loss is on the assistant turn only.
Plain Adam with a cosine schedule and warmup. The adapter ships to the project repo during training and at the end, under
lora/<model>/r<rank>/<stamp>_<tag>/."""
from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
from datetime import datetime, timezone

from . import REPO, __version__, chain, prompts
from .hub import ship
from .extract import fetch_images                # by url list or by locator (a pretrain shard row, a Hub file, a zip member)
from .screen import load_inputs, preflight, vlm_class

# The language model's projections, by name (the vision tower, under `visual`, is left alone): the full-attention layers'
# q/k/v/o, every layer's MLP, and the linear-attention layers' token/value projections (in_proj_qkv, in_proj_z, out_proj in
# Qwen3.5; in_proj_qkvz in the Qwen3-Next naming). The tiny gate projections (in_proj_a, in_proj_b) are not adapted.
TARGETS = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_qkvz|out_proj)$")
TURN_END = "<|im_end|>\n"
PICTURE_FORMS = ("short", "full")
SPEC_FORMS = ("spec_short", "spec_full")


def conversation(it: dict, form: str, image=None) -> tuple:
    """(the user messages, the answer text) of one item under one form."""
    depth = it["depth"]
    if form == "short":
        user, answer = [{"type": "image", "image": image}, {"type": "text", "text": prompts.short(depth)}], chain.render(it)
    elif form == "full":
        user, answer = [{"type": "image", "image": image}, {"type": "text", "text": prompts.full(depth)}], chain.render(it, full=True)
    elif form == "spec_short":
        user, answer = [{"type": "text", "text": prompts.spec_short(depth, it["spec"])}], chain.render(it)
    elif form == "spec_full":
        user, answer = [{"type": "text", "text": prompts.spec_full(depth, it["spec"])}], chain.render(it, full=True)
    else:
        raise ValueError(f"unknown form {form!r}")
    return [{"role": "user", "content": user}], answer


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

    def __init__(self, processor, pictures: dict, max_len: int):
        self.processor, self.pictures, self.max_len, self.prefix_len = processor, pictures, max_len, {}

    def prefix_length(self, key: tuple, prefix: str, image) -> int:
        if key not in self.prefix_len:
            kwargs = {"images": [image]} if image is not None else {}
            self.prefix_len[key] = int(self.processor(text=[prefix], return_tensors="pt", **kwargs)["input_ids"].shape[1])
        return self.prefix_len[key]

    def __call__(self, pairs: list) -> dict:
        texts, images, lengths = [], [], []
        for it, form in pairs:
            image = self.pictures.get(it["id"]) if it["kind"] == "image" else None
            messages, answer = conversation(it, form, image)
            prefix = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            texts.append(prefix + answer + TURN_END)
            if image is not None:
                images.append(image)
            lengths.append(self.prefix_length((it["id"], form), prefix, image))
        kwargs = {"images": images} if images else {}
        batch = self.processor(text=texts, padding=True, return_tensors="pt", **kwargs)
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


def run(model_id: str = "Qwen/Qwen3.5-9B", inputs="hub:sft/sft_v1.json", out_dir: str = "/content/train",
        rank: int = 32, alpha: int = 64, dropout: float = 0.05, lr: float = 2e-4, epochs: float = 1.0, batch: int = 8,
        grad_accum: int = 2, max_len: int = 2048, warmup: float = 0.03, n_items: int | None = None, ship_every: int = 100,
        repo: str = REPO, do_ship: bool = True, seed: int = 20261030, tag: str = "", init_adapter: str | None = None) -> dict:
    """Train the adapter; return {adapter_dir, path_in_repo, steps, s_per_step, train_loss, ...}. With init_adapter (see
    adapter_source) training continues from that trained adapter instead of a fresh LoRA; its rank and alpha must be the ones passed."""
    from .checks import peft_beside_torchao
    peft_check = peft_beside_torchao()
    assert peft_check["peft_lora_ok"], f"PEFT cannot build a LoRA here: {peft_check.get('error')}; {peft_check.get('advice', '')}"
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    short = model_id.split("/")[-1]
    path_in_repo = f"lora/{short}/r{rank}/{stamp}"
    run_dir = os.path.join(out_dir, f"{short}_r{rank}_{stamp}")
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    env = preflight()
    items = load_items(inputs)[:n_items]
    t0 = time.time()
    pictures = fetch_images(items, os.path.join(out_dir, "images"))
    items = [it for it in items if it["kind"] != "image" or it["id"] in pictures]
    pairs = examples_of(items)
    random.Random(seed).shuffle(pairs)
    n_img = sum(1 for it in items if it["kind"] == "image")
    forms = {}
    for _, form in pairs:
        forms[form] = forms.get(form, 0) + 1
    print(f"train {stamp}: {len(items)} items ({n_img} with a picture), {len(pairs)} examples by form {forms}; images fetched in "
          f"{time.time() - t0:.0f} s", flush=True)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    model = vlm_class().from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if init_adapter:
        from peft import PeftModel
        from .hub import token
        src, sub = adapter_source(init_adapter, repo)
        print(f"continuing from the adapter {src}" + (f" / {sub}" if sub else ""), flush=True)
        model = PeftModel.from_pretrained(model, src, is_trainable=True, token=token(), **({"subfolder": sub} if sub else {}))
        cfg = model.peft_config["default"]
        assert (cfg.r, cfg.lora_alpha) == (rank, alpha), f"the adapter is r{cfg.r} alpha {cfg.lora_alpha}; pass its own rank and alpha"
    else:
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=TARGETS,
                            task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, config)
    report = lora_report(model)
    print("LoRA modules by name:", report, flush=True)
    assert report, "no module matched the LoRA targets; read the model's module names"
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(pairs) / (batch * grad_accum))
    total = math.ceil(steps_per_epoch * epochs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999))        # plain Adam, no weight decay
    wanted = dict(output_dir=run_dir, per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
                  num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=math.ceil(total * warmup),
                  bf16=True, logging_steps=10, save_strategy="no", report_to=[], remove_unused_columns=False,
                  dataloader_num_workers=0, seed=seed, gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False})
    allowed = set(inspect.signature(TrainingArguments).parameters)
    dropped = [k for k in wanted if k not in allowed]
    if dropped:
        print("  this transformers' TrainingArguments lacks", dropped, "(left out)", flush=True)
    args = TrainingArguments(**{k: v for k, v in wanted.items() if k in allowed})
    log = []

    class Progress(TrainerCallback):
        def __init__(self):
            self.t0 = None

        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()
            print(f"  {total} steps of {batch} x {grad_accum} sequences; {epochs} epoch(s)", flush=True)

        def on_log(self, args, state, control, logs=None, **kw):
            spent = time.time() - self.t0
            done = max(state.global_step, 1)
            rec = {"step": state.global_step, "spent_s": round(spent), "left_s": round(spent / done * (total - done)),
                   "s_per_step": round(spent / done, 2), **{k: v for k, v in (logs or {}).items() if k in ("loss", "learning_rate")}}
            log.append(rec)
            print(f"  step {rec['step']}/{total}: loss {rec.get('loss', float('nan')):.4f}, {rec['spent_s']} s spent, "
                  f"{rec['left_s']} s left, {rec['s_per_step']} s/step", flush=True)
            if do_ship and ship_every and state.global_step and state.global_step % ship_every == 0 and state.global_step < total:
                model.save_pretrained(adapter_dir)
                ship(adapter_dir, f"{path_in_repo}/step_{state.global_step}", f"{short} r{rank} {stamp}: step {state.global_step}", repo)

    trainer = Trainer(model=model, args=args, train_dataset=Pairs(pairs), data_collator=Collator(processor, pictures, max_len),
                      optimizers=(optimizer, None), callbacks=[Progress()])
    result = trainer.train()
    model.save_pretrained(adapter_dir)
    summary = {"package": __version__, "model": model_id, "tag": tag, "rank": rank, "alpha": alpha, "dropout": dropout, "lr": lr,
               "epochs": epochs, "batch": batch, "grad_accum": grad_accum, "max_len": max_len, "items": len(items),
               "items_with_picture": n_img, "examples": len(pairs), "examples_by_form": forms, "steps": result.global_step,
               "train_loss": result.training_loss, "s_per_step": round((time.time() - t0) / max(result.global_step, 1), 2),
               "train_min": round((time.time() - t0) / 60, 1), "lora_modules": report, "targets": TARGETS, "inputs": inputs,
               "init_adapter": init_adapter, "path_in_repo": path_in_repo, "env": env, "log": log}
    with open(os.path.join(adapter_dir, "train_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\ntrained in {summary['train_min']} min: {summary['steps']} steps, loss {summary['train_loss']:.4f}, "
          f"{summary['s_per_step']} s/step", flush=True)
    if do_ship:
        url = ship(adapter_dir, f"{path_in_repo}/final", f"{short} r{rank} {stamp}: final ({summary['steps']} steps)", repo)
        print(f"  shipped: {url}", flush=True)
    summary["adapter_dir"] = adapter_dir
    del trainer, model, processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return summary
