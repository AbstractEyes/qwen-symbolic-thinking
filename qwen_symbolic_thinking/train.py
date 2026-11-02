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

from . import REPO, __version__, chain, family, prompts
from .hub import model_short, resolve_model, ship
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


def conversation(it: dict, form: str, image=None, system: str | None = None, mark: str | None = None) -> tuple:
    """(the messages, the answer text) of one item under one form: the user turn, after a system turn when a system prompt is given.
    With a picture marker (family.marker) the user content is one string."""
    depth = it["depth"]
    if form == "short":
        user, answer = family.content(prompts.short(depth), image, mark), chain.render(it)
    elif form == "full":
        user, answer = family.content(prompts.full(depth), image, mark), chain.render(it, full=True)
    elif form == "spec_short":
        user, answer = family.content(prompts.spec_short(depth, it["spec"]), None, mark), chain.render(it)
    elif form == "spec_full":
        user, answer = family.content(prompts.spec_full(depth, it["spec"]), None, mark), chain.render(it, full=True)
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
                 kind: str = ""):
        self.processor, self.pictures, self.max_len, self.prefix_len, self.system = processor, pictures, max_len, {}, system
        self.mark, self.kind = mark, kind   # the picture marker and the family key (family.marker, family.model_type)
        self.turn_end = turn_end(processor)
        self.collated = self.cut = self.longest = 0   # examples collated, examples longer than max_len (tail cut), the longest

    def prefix_length(self, key: tuple, prefix: str, image) -> int:
        if key not in self.prefix_len:
            self.prefix_len[key] = int(encode(self.processor, [prefix], [image])["input_ids"].shape[1])
        return self.prefix_len[key]

    def __call__(self, pairs: list) -> dict:
        texts, images, lengths = [], [], []
        for it, form in pairs:
            image = self.pictures.get(it["id"]) if it["kind"] == "image" else None
            messages, answer = conversation(it, form, image, self.system, self.mark)
            prefix = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                        **family.thinking_kwargs(self.processor, False))
            texts.append(prefix + answer + self.turn_end)
            images.append(image)                      # one slot per text: None for a specification row
            lengths.append(self.prefix_length((it["id"], form), prefix, image))
        batch = encode(self.processor, texts, family.fill_images(self.kind, images), padding=True)
        family.fix_padding(self.kind, batch, self.processor.tokenizer.pad_token_id)
        n, cut, longest = overrun(batch["attention_mask"], self.max_len)
        self.collated, self.cut, self.longest = self.collated + n, self.cut + cut, max(self.longest, longest)
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


def run(model_id: str = "Qwen/Qwen3.5-9B", inputs="hub:sft/sft_v1.json", out_dir: str = "/content/train",
        rank: int = 32, alpha: int = 64, dropout: float = 0.05, lr: float = 2e-4, epochs: float = 1.0, batch: int = 8,
        grad_accum: int = 2, max_len: int = 2048, warmup: float = 0.03, n_items: int | None = None, ship_every: int | None = None,
        repo: str = REPO, do_ship: bool = True, seed: int = 20261030, tag: str = "", init_adapter: str | None = None,
        system: str | None = None, trust_remote_code: bool = False, revision: str | None = None,
        targets: str | None = None, image_settings: dict | None = None) -> dict:
    """Train the adapter; return {adapter_dir, path_in_repo, steps, s_per_step, train_loss, ...}. With init_adapter (see
    adapter_source) training continues from that trained adapter instead of a fresh LoRA; its rank and alpha must be the ones passed.
    With system (see prompts.system) every example carries that system turn before its user turn. Rank 0 is a full fine-tune of
    the language model (fp32 weights under bf16 autocast; the vision tower frozen); adapter_dir then holds the whole model in bf16
    with its processor. ship_every: the adapter ships every so many steps (100 by default; a full model ships only at the end)."""
    full = rank == 0
    import torch
    from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments
    if not full:
        from .checks import peft_beside_torchao
        peft_check = peft_beside_torchao()
        assert peft_check["peft_lora_ok"], f"PEFT cannot build a LoRA here: {peft_check.get('error')}; {peft_check.get('advice', '')}"
        from peft import LoraConfig, TaskType, get_peft_model
    if ship_every is None:
        ship_every = 0 if full else 100
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    short = model_short(model_id)
    path_in_repo = f"full/{short}/{stamp}" if full else f"lora/{short}/r{rank}/{stamp}"
    run_dir = os.path.join(out_dir, f"{short}_{'full' if full else f'r{rank}'}_{stamp}")
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    env = preflight()
    src = resolve_model(model_id, repo)
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
    family.apply_image_settings(processor, image_settings)
    mark = family.marker(model)
    targets = targets or TARGETS
    if trust_remote_code or mark or image_settings:
        print(f"  {family.model_type(model)}" + (f": picture marker {mark}" if mark else "")
              + (f"; repository code at {revision}" if trust_remote_code else "")
              + (f"; image processor {image_settings}" if image_settings else ""), flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if family.input_grads(model):
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
        model = get_peft_model(model, config)
    if not full:
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
                name_base(adapter_dir, base_dir, model_id, repo)
                ship(adapter_dir, f"{path_in_repo}/step_{state.global_step}", f"{short} r{rank} {stamp}: step {state.global_step}", repo)

    collate = Collator(processor, pictures, max_len, system_text, mark, family.model_type(model))
    pic = next((p for p in pairs if p[1] in PICTURE_FORMS and pictures.get(p[0]["id"]) is not None), None)
    spc = next((p for p in pairs if p[1] in SPEC_FORMS), None)
    if pic and spc:   # a mixed batch encodes before the first step (a family whose processor takes pictures differently fails here, not at step 1)
        probe = collate([pic, spc])
        print(f"  a mixed batch (a picture row, a specification row) encodes: input_ids {tuple(probe['input_ids'].shape)}", flush=True)
        collate.collated = collate.cut = collate.longest = 0   # the counts are the training batches'
    trainer = Trainer(model=model, args=args, train_dataset=Pairs(pairs), data_collator=collate,
                      optimizers=(optimizer, None), callbacks=[Progress()])
    result = trainer.train()
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
               "items_with_picture": n_img, "examples": len(pairs), "examples_by_form": forms, "steps": result.global_step,
               "train_loss": result.training_loss, "s_per_step": round((time.time() - t0) / max(result.global_step, 1), 2),
               "train_min": round((time.time() - t0) / 60, 1), "lora_modules": report, "targets": targets, "inputs": inputs,
               "model_type": family.model_type(model), "trust_remote_code": trust_remote_code, "revision": revision,
               "image_settings": image_settings,
               "init_adapter": init_adapter, "system": system, "system_tokens": system_tokens, "path_in_repo": path_in_repo,
               "collated_examples": collate.collated, "examples_cut_at_max_len": collate.cut, "longest_example_tokens": collate.longest,
               "env": env, "log": log}
    print(f"  examples longer than max_len {max_len} (answer tail cut): {collate.cut} of {collate.collated} collated "
          f"(the longest {collate.longest} tokens)", flush=True)
    with open(os.path.join(adapter_dir, "train_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\ntrained in {summary['train_min']} min: {summary['steps']} steps, loss {summary['train_loss']:.4f}, "
          f"{summary['s_per_step']} s/step", flush=True)
    if do_ship:
        url = ship(adapter_dir, f"{path_in_repo}/final", f"{short} {'full' if full else f'r{rank}'} {stamp}: final ({summary['steps']} steps)", repo)
        print(f"  shipped: {url}", flush=True)
    summary["adapter_dir"] = adapter_dir
    del trainer, model, processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return summary
