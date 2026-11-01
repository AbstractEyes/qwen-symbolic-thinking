"""Does a picture reach the model inside the trainer's own batch? A diagnostic for a looker that trained blind: the same picture
rows go through train.Collator exactly as training builds them, and the loss of the true answers is measured against the loss
with the pictures rolled between the rows (each row paired with its neighbour's picture) and with the pictures blanked, under the
configurations that separate the screen's path from the trainer's: eval or train mode, left or right padding, the plain model or
the model wrapped as the trainer wraps it (use_cache off, gradient checkpointing, the input-gradient hook, a fresh LoRA, whose
zero-initialised B leaves the function unchanged), with and without the multimodal token types. A model that sees pays for the
wrong picture and for no picture; a path on which the rolled and blank losses equal the true one is blind on that path.

Usage (a queue spec):
    {"name": "D_pictures_g12b", "kind": "tool", "needs": "0.1.23",
     "tool": {"module": "picture_probe", "args": {"model_id": "google/gemma-4-12B-it", "n_rows": 3}}}
The result (the batch's anatomy and a table of configurations) lands in the run's marker and prints as markdown."""
from __future__ import annotations

import os
import time

from .. import REPO

TYPE_KEYS = ("mm_token_type_ids", "token_type_ids")
TEXT_KEYS = ("input_ids", "attention_mask", "labels")
SEES, BLIND = 0.2, 0.05          # nats: the price of the wrong picture that counts as seeing; the slack that counts as blind


def rolled(batch: dict):
    """The batch with every picture moved to the next row (the last to the first); None when the rows' picture token counts differ
    (a rolled picture would then not fit its row's placeholders). Qwen's processor flattens the patches of all pictures into one
    tensor with image_grid_thw beside; Gemma's stack one picture per leading index."""
    import torch
    b = dict(batch)
    if "image_grid_thw" in b:
        grid = b["image_grid_thw"]
        sizes = [int(x) for x in grid.prod(-1)]
        if len(sizes) < 2 or len(set(sizes)) != 1:
            return None
        parts = list(torch.split(b["pixel_values"], sizes))
        b["pixel_values"] = torch.cat(parts[-1:] + parts[:-1])
        b["image_grid_thw"] = torch.roll(grid, 1, 0)
        return b
    n = int(b["pixel_values"].shape[0])
    if n < 2:
        return None
    for k, v in list(b.items()):
        if k in TYPE_KEYS or k in TEXT_KEYS:
            continue
        if hasattr(v, "shape") and v.dim() >= 1 and int(v.shape[0]) == n:
            b[k] = torch.roll(v, 1, 0)
    return b


def blanked(batch: dict) -> dict:
    import torch
    return {**batch, "pixel_values": torch.zeros_like(batch["pixel_values"])}


def anatomy(batch: dict, image_token_id) -> dict:
    """What the batch holds: each key's shape and dtype; per row the real span, the image tokens and the trained span."""
    out = {"keys": {k: ([int(x) for x in v.shape] + [str(v.dtype).replace("torch.", "")]) if hasattr(v, "shape") else type(v).__name__
                    for k, v in batch.items()}, "rows": []}
    ids, labels, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
    for i in range(int(ids.shape[0])):
        real = att[i].nonzero().flatten()
        trained = (labels[i] != -100).nonzero().flatten()
        out["rows"].append({"length": int(att[i].sum()), "first_real": int(real[0]), "last_real": int(real[-1]),
                            "image_tokens": int((ids[i] == image_token_id).sum()) if image_token_id is not None else None,
                            "trained_tokens": int(len(trained)), "first_trained": int(trained[0]) if len(trained) else None,
                            "last_trained": int(trained[-1]) if len(trained) else None})
    return out


def loss_of(model, batch: dict, train_mode: bool, grad: bool, drop_types: bool = False) -> float:
    import torch
    b = {k: v for k, v in batch.items() if not (drop_types and k in TYPE_KEYS)}
    device = next(model.parameters()).device
    b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
    was = model.training
    model.train(train_mode)
    try:
        with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b)
        return float(out.loss.detach().float())
    finally:
        model.train(was)


def verdict(true: float, rolled_loss, blank: float) -> str:
    wrong = (rolled_loss - true) if rolled_loss is not None else None
    none = blank - true
    if none >= SEES and (wrong is None or wrong >= SEES):
        return "sees"
    if none <= BLIND and (wrong is None or wrong <= BLIND):
        return "blind"
    return "partial"


def run(model_id: str, n_rows: int = 3, inputs: str = "hub:sft/pictures_v2.json", form: str = "full", max_len: int = 2048,
        rank: int = 32, alpha: int = 64, dropout: float = 0.05, out_dir: str = "/content/runs", tag: str = "", repo: str = REPO) -> dict:
    import torch
    from transformers import AutoProcessor
    from ..screen import resolve_model, vlm_class
    from ..train import TARGETS, Collator, fetch_images, load_items
    t0 = time.time()
    src = resolve_model(model_id, repo)
    items = [it for it in load_items(inputs) if it["kind"] == "image"][: max(n_rows * 2, n_rows)]
    pictures = fetch_images(items, os.path.join(out_dir, "images"))
    items = [it for it in items if it["id"] in pictures][:n_rows]
    assert len(items) >= 2, f"{len(items)} picture rows fetched; the rolled picture needs two"
    pairs = [(it, form) for it in items]
    processor = AutoProcessor.from_pretrained(src)
    model = vlm_class().from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    image_token_id = getattr(model.config, "image_token_id", None)

    def batch_for(side: str) -> dict:
        processor.tokenizer.padding_side = side
        return Collator(processor, pictures, max_len)(pairs)

    left, right = batch_for("left"), batch_for("right")
    result = {"model": model_id, "rows": [it["id"] for it in items], "form": form, "image_token_id": image_token_id,
              "anatomy_right": anatomy(right, image_token_id), "anatomy_left": anatomy(left, image_token_id), "table": []}
    counts = {r["image_tokens"] for r in result["anatomy_right"]["rows"]}
    result["rolled_available"] = len(counts) == 1
    variants = {"true": lambda b: b, "rolled": rolled, "blank": blanked}

    def measure(name: str, m, batch: dict, train_mode: bool, grad: bool, drop_types: bool = False):
        row = {"config": name}
        for v, make in variants.items():
            b = make(batch)
            row[v] = None if b is None else loss_of(m, b, train_mode, grad, drop_types)
        row["verdict"] = verdict(row["true"], row["rolled"], row["blank"])
        result["table"].append(row)
        print(f"  {name}: true {row['true']:.3f} | rolled {row['rolled'] if row['rolled'] is None else round(row['rolled'], 3)} "
              f"| blank {row['blank']:.3f} -> {row['verdict']}", flush=True)

    print(f"picture probe on {model_id}: {len(items)} rows of the form {form}; image tokens per row {sorted(counts)}", flush=True)
    measure("plain, eval, left padding (the screen's path)", model, left, False, False)
    measure("plain, eval, right padding", model, right, False, False)
    measure("plain, train mode, right padding", model, right, True, False)
    measure("plain, train mode, right padding, no token types", model, right, True, False, drop_types=True)

    from peft import LoraConfig, TaskType, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    wrapped = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=TARGETS,
                                               task_type=TaskType.CAUSAL_LM))
    measure("wrapped, eval, right padding", wrapped, right, False, False)
    measure("wrapped, train mode, right padding, no grad", wrapped, right, True, False)
    measure("wrapped, train mode, right padding, grad (the trainer's path)", wrapped, right, True, True)
    measure("wrapped, train mode, right padding, grad, no token types", wrapped, right, True, True, drop_types=True)
    measure("wrapped, train mode, left padding, grad", wrapped, left, True, True)

    result["seconds"] = round(time.time() - t0, 1)
    print("\n| configuration | true | rolled | blank | verdict |\n|---|---:|---:|---:|---|")
    for r in result["table"]:
        rolled_s = "-" if r["rolled"] is None else f"{r['rolled']:.3f}"
        print(f"| {r['config']} | {r['true']:.3f} | {rolled_s} | {r['blank']:.3f} | {r['verdict']} |")
    print(f"\n{result['seconds']} s", flush=True)
    return result
