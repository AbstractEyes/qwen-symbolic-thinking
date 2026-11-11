"""Keeping the base model's behavior while a LoRA learns the rows.

A fine-tune on one fixed answer form moves more than the form: the form starts to appear where it was not asked for, and
the model's other answers drift away from the base model's. The terms here hold the adapted model near the base. Each is
read against the base model itself (the same weights with the adapter switched off), so no second copy of the model is kept,
and each is a mean over the answer positions of its micro-batch:

    general_kl   lambda_g * KL( p_base || p_tuned ) at the answer positions of general rows: requests that do not ask for the
                 row form (other questions about pictures, text-only requests), answered by the base model itself. The base
                 side runs with the adapter off and no gradient.
    target_kl    lambda_t * KL( p_base^-y || p_tuned^-y ) at the answer positions of the training rows, where p^-y is the
                 next-token distribution with the training token y removed and the rest renormalized. It keeps the base's
                 ranking of the alternatives at every position without opposing the cross-entropy on y (Xu et al., "Mask the
                 Target: A Plug-and-Play Regularizer Against LoRA Forgetting", arXiv 2605.29498). Positions where the base
                 already gives y more than 1 - 1e-4 are left out, as there.
    rehearsal    mu * CE on the general rows (the base's own answers as targets; no base pass at training time).

The training step with any term on: loss = CE(training rows) + general_kl * KL_g + target_kl * KL_t + rehearsal * CE_g.
With every weight at zero the plain Trainer runs, unchanged.

The general rows ride beside each micro-batch of training rows (PairedCollator: `anchor_batch` of them, drawn in a seeded
order and cycled). An item of the general set: {"kind": "anchor", "id", "prompt", "answer", "writer", and, for a picture,
"image_urls" or "image" (a locator, as for the training items)}.
"""
from __future__ import annotations

import random
from collections import Counter

DROP_ABOVE = 1.0 - 1e-4      # target-masked KL: positions where the base gives the target more than this are left out


# ---------------------------------------------------------------------------------------------------------- the terms
def answer_positions(labels):
    """(mask over the shifted positions, the targets there): position t predicts token t + 1, so the mask is
    labels[:, 1:] != -100 over logits[:, :-1]."""
    shifted = labels[:, 1:]
    mask = shifted != -100
    return mask, shifted[mask]


def _chunks(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(n, i + size))


def kl_full(z_base, z_tuned, chunk: int = 1024):
    """Mean over rows of KL(p_base || p_tuned) for logits [N, V] (the base side carries no gradient). Computed in fp32,
    chunk rows at a time (the vocabulary is large)."""
    import torch
    n = z_tuned.shape[0]
    if n == 0:
        return z_tuned.sum() * 0.0
    total = z_tuned.new_zeros((), dtype=torch.float32)
    for s in _chunks(n, chunk):
        lb = torch.log_softmax(z_base[s].float(), dim=-1)
        lt = torch.log_softmax(z_tuned[s].float(), dim=-1)
        total = total + (lb.exp() * (lb - lt)).sum()
    return total / n


def kl_target_masked(z_base, z_tuned, targets, chunk: int = 1024, drop_above: float = DROP_ABOVE):
    """Mean over the kept rows of KL(p_base^-y || p_tuned^-y): both distributions with the target token y removed and the
    rest renormalized. Rows where the base's own probability of y exceeds drop_above are left out. Returns (the mean, the
    rows kept)."""
    import torch
    n = z_tuned.shape[0]
    if n == 0:
        return z_tuned.sum() * 0.0, 0
    total = z_tuned.new_zeros((), dtype=torch.float32)
    kept = 0
    for s in _chunks(n, chunk):
        zb, zt, y = z_base[s].float(), z_tuned[s].float(), targets[s].view(-1, 1)
        keep = torch.log_softmax(zb, dim=-1).gather(1, y).squeeze(1).exp() <= drop_above
        if not bool(keep.any()):
            continue
        zb, zt, y = zb[keep], zt[keep], y[keep]
        neg = torch.finfo(zb.dtype).min
        lb = torch.log_softmax(zb.scatter(1, y, neg), dim=-1)      # the target's logit at the dtype's floor: weight 0
        lt = torch.log_softmax(zt.scatter(1, y, neg), dim=-1)
        pb = lb.exp()
        total = total + (pb * (lb - lt)).sum()
        kept += int(keep.sum())
    return (total / kept if kept else total * 0.0), kept


def ce_mean(z, targets, chunk: int = 1024):
    """Mean cross-entropy of logits [N, V] against targets [N], in fp32, chunk rows at a time."""
    import torch
    n = z.shape[0]
    if n == 0:
        return z.sum() * 0.0
    total = z.new_zeros((), dtype=torch.float32)
    for s in _chunks(n, chunk):
        total = total + torch.nn.functional.cross_entropy(z[s].float(), targets[s], reduction="sum")
    return total / n


def scale_adapter(model, factor: float) -> int:
    """Multiply every LoRA layer's scaling by factor, before a merge: the merged weights become W + factor * (alpha / r) B A,
    an interpolation between the base model (0) and the adapter as trained (1) (the delta form of weight interpolation:
    Wortsman et al., "Robust fine-tuning of zero-shot models", arXiv 2109.01903). Returns the number of layers scaled."""
    n = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and scaling:
            for name in list(scaling):
                scaling[name] = scaling[name] * factor
            n += 1
    if not n:
        raise ValueError("no LoRA layer to scale")
    return n


# ---------------------------------------------------------------------------------------------------- who wrote what
def check_writers(items: list, writers) -> dict:
    """Every item names its writer (`writer`: the model, or the people, whose text the item teaches). With a list of
    writers, an item whose writer is missing or not on the list stops the run before anything loads; returns the counts by
    writer. With writers None nothing is checked and {} comes back."""
    if writers is None:
        return {}
    allowed = set(writers)
    seen, refused = Counter(), Counter()
    for it in items:
        w = it.get("writer")
        (seen if w in allowed else refused)[w if w else "(none)"] += 1
    if refused:
        raise ValueError(f"{sum(refused.values())} items come from writers this run does not accept: {dict(refused)} "
                         f"(accepted: {sorted(allowed)})")
    return dict(seen)


# ------------------------------------------------------------------------------------------------------ the batches
class PairedCollator:
    """A training micro-batch with general rows beside it: the training rows' batch as the Trainer expects it, plus
    batch["anchor"] = the general rows' batch, built by a collator of the same kind (same template, picture handling and
    answer-only labels; its own instance, so the training rows' counts stay theirs). The general rows come in a seeded
    order and cycle."""

    def __init__(self, collate, anchors: list, anchor_batch: int, seed: int, anchor_collate=None):
        if not anchors:
            raise ValueError("no general rows to pair")
        self.collate, self.anchors, self.k = collate, list(anchors), int(anchor_batch)
        self.anchor_collate = anchor_collate or collate
        self.rng = random.Random(seed)
        self.order, self.at = [], 0

    def _next(self, k: int) -> list:
        out = []
        while len(out) < k:
            if self.at >= len(self.order):
                self.order = list(range(len(self.anchors)))
                self.rng.shuffle(self.order)
                self.at = 0
            out.append((self.anchors[self.order[self.at]], "anchor"))
            self.at += 1
        return out

    def __call__(self, pairs: list) -> dict:
        batch = self.collate(pairs)
        if self.k > 0:
            batch["anchor"] = self.anchor_collate(self._next(self.k))
        return batch


# ------------------------------------------------------------------------------------------------------ the trainer
def keep_trainer(general_kl: float = 0.0, target_kl: float = 0.0, rehearsal: float = 0.0, stats: dict | None = None,
                 chunk: int = 1024):
    """A Trainer class whose loss adds the terms above to the stock cross-entropy on the training rows. The training rows'
    loss is the stock one (computed by super().compute_loss, so its normalization is the plain Trainer's); the added terms
    are micro-batch means, divided by the accumulation count exactly when the stock loss comes back already scaled for it.
    stats, when given, collects the running sums of each term for the progress log."""
    from transformers import Trainer

    weights = {"general_kl": float(general_kl), "target_kl": float(target_kl), "rehearsal": float(rehearsal)}

    class KeepTrainer(Trainer):
        keep_weights = weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            anchor = inputs.pop("anchor", None)
            peft_model = self.accelerator.unwrap_model(model)
            if not hasattr(peft_model, "disable_adapter"):
                raise ValueError("these terms read the base model through the adapter's off switch: a LoRA only")
            base_side = None
            if weights["target_kl"] > 0:
                plain = {k: v for k, v in inputs.items() if k not in ("labels", "num_items_in_batch")}
                mask, targets = answer_positions(inputs["labels"])
                with torch.no_grad(), peft_model.disable_adapter():
                    zb = model(**plain).logits[:, :-1][mask]
                base_side = (mask, targets, zb)
            task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            extra = task_loss.new_zeros(())
            parts = {}
            if base_side is not None:
                mask, targets, zb = base_side
                kt, kept = kl_target_masked(zb, outputs.logits[:, :-1][mask], targets, chunk)
                extra = extra + weights["target_kl"] * kt
                parts["target_kl"] = float(kt.detach())
                parts["target_kl_kept"] = kept / max(int(targets.numel()), 1)
                del zb
            if anchor is not None and (weights["general_kl"] > 0 or weights["rehearsal"] > 0):
                a_plain = {k: v for k, v in anchor.items() if k not in ("labels", "num_items_in_batch")}
                a_mask, a_targets = answer_positions(anchor["labels"])
                zt = model(**a_plain).logits[:, :-1][a_mask]
                if weights["general_kl"] > 0:
                    with torch.no_grad(), peft_model.disable_adapter():
                        zb = model(**a_plain).logits[:, :-1][a_mask]
                    kg = kl_full(zb, zt, chunk)
                    extra = extra + weights["general_kl"] * kg
                    parts["general_kl"] = float(kg.detach())
                    del zb
                if weights["rehearsal"] > 0:
                    cr = ce_mean(zt, a_targets, chunk)
                    extra = extra + weights["rehearsal"] * cr
                    parts["rehearsal_ce"] = float(cr.detach())
            scaled = (self.loss_is_scaled_for_ga if getattr(self, "loss_is_scaled_for_ga", None) is not None else
                      (self.model_accepts_loss_kwargs and num_items_in_batch is not None) or self.compute_loss_func is not None)
            if scaled:
                extra = extra / getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
            if stats is not None:
                stats["n"] = stats.get("n", 0) + 1
                for k, v in parts.items():
                    stats[k] = stats.get(k, 0.0) + v
            loss = task_loss + extra
            return (loss, outputs) if return_outputs else loss

    return KeepTrainer
