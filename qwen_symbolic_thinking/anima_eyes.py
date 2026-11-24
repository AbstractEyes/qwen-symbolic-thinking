"""Anima's eyes: InternVL3.5-1B with Anima's own top layers put back (the "last-4 hybrid"), as a model family of the trainer.

Anima's text encoder is Qwen/Qwen3-0.6B-Base, byte for byte. OpenGVLab's InternVL3.5-1B is a Qwen3-0.6B retrained together with
a picture encoder (InternViT-300M and an MLP projector), so its language model has the 0.6B's shape (28 layers x 1,024, the same
vocabulary). Its last layers drifted from the Base during that training: the final states (after the last norm, what Anima
reads) agree with the Base's at 0.50 (centered cosine, 24 Anima-style prompts), and Anima loses a detailed caption's specifics.
With the Base's own layers 24-27 and final norm put back on InternVL's word table and lower 24 layers, the agreement is 0.81 and
Anima draws as with its own encoder (an A/B in ComfyUI against Anima's own encoder, 2026-11-24). The hybrid keeps 24 of
InternVL's picture-trained layers, its vision tower and its projector: a model that reads pictures and still fits Anima's slot.
Its output head (HEAD) is the Base's word table, the head the Base reads its own top layers with; InternVL's head, trained on
InternVL's top layers, looped on them (the choice is pinned below with the caption check that made it).

    from qwen_symbolic_thinking import anima_eyes
    folder = anima_eyes.build("/content/anima_eyes")           # the hybrid from the two pinned Hub sources, fingerprint-checked
    python -m qwen_symbolic_thinking.tools.anima_eyes_base     # (a queue tool) build it and ship it to the project repo

The pieces:
- build(): downloads OpenGVLab/InternVL3_5-1B-HF and Qwen/Qwen3-0.6B-Base at pinned commits, checks each weights file's sha256
  against the Hub's own record, composes the hybrid at the tensor level (no model code runs), checks the result's fingerprint
  against the one pinned here (its language model is the text-encoder file the ComfyUI A/B drew with), and writes a folder
  transformers loads with its own InternVLForConditionalGeneration (no repository code): the weights in bf16, InternVL's
  processor and chat template, and the config with tie_word_embeddings false and the record under the key "anima_eyes".
- The family (family.anima_eyes reads that key): Anima's slot (layers 24-27 and the final norm) stays frozen in a full fine-tune
  and carries no LoRA modules (lora_targets, is_slot), and every load checks the slot's fingerprint (check_slot).
- The slot term (SlotTerm): beside every micro-batch, a few Anima-style prompts (the training items' own wordings: their
  generation prompt, their tags, their caption), read as Anima reads them (the bare text, no chat template, no start token);
  the model's final states are held to Anima's encoder's (the Base's) on the same prompt:

      L_slot = (1 / T) * sum over the T real prompt tokens t of  || h_model(t) - h_anima(t) ||^2 / || h_anima(t) ||^2
      loss   = CE(training rows) + general_kl * KL_g + slot * L_slot

  (h: the final states after the last norm; the Base runs frozen, in bf16, without gradient). Reported beside it (SlotHeld):
  the agreement on 24 held Anima-style prompts never trained on, read as the numbers above were (the centered cosine with the
  Base's mean state removed, each prompt's first token read apart; the hybrid starts at 0.805), the first token's cosine, and
  the relative difference.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time

INTERNVL = {"repo": "OpenGVLab/InternVL3_5-1B-HF", "revision": "9191dbccf312b537016f041b25d61c72e7c5c9f3",
            "weights": "model.safetensors", "bytes": 2_121_890_856,
            "sha256": "11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39",
            "files": ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
                      "video_preprocessor_config.json", "chat_template.jinja", "added_tokens.json", "special_tokens_map.json",
                      "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"),
            "license": "Apache-2.0"}
BASE = {"repo": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "weights": "model.safetensors", "bytes": 1_192_135_096,
        "sha256": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        "files": ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
        "license": "Apache-2.0"}
FIRST_BASE_LAYER, N_LAYERS = 24, 28
HEADS = ("internvl", "base")
# The output head, chosen by a caption check (2026-11-24: 11 pictures, 12 asks, greedy, on the CPU): the Base's own word table,
# its tied head, which reads the Base's top layers as it was trained to. With it the hybrid described the pictures nearly as
# InternVL3.5-1B does (the same subject on 10 of 11) and ended 7 captions of 12 within 160 tokens; with InternVL's head
# (trained on InternVL's own top layers) it kept the gist but looped, ending 1 of 12. "internvl" builds the other (its
# fingerprint is pinned below too).
HEAD = "base"
NAME = "InternVL3_5-1B-anima-last4"          # the folder's name; the model's short name in run paths (hub.model_short)
FOLDER = f"base/{NAME}"                       # where the lane's first run ships it in the project repo
MODEL_ID = f"hub:{FOLDER}"                    # the model id every later run loads it by
CONFIG_KEY = "anima_eyes"                     # the record's key in the folder's config.json (read back by family.anima_eyes)
# The fingerprints of the composed hybrid (every tensor, bf16), by head, and of Anima's slot (layers 24-27 and the final
# norm, = the Base's own tensors), computed 2026-11-24 from the two pinned files; the hybrid's language model was checked then
# tensor for tensor against the last-4 text-encoder file the ComfyUI A/B used.
FINGERPRINTS = {"internvl": "f4df14201bbb11ef08a687944706ee270260613092fb87a4bf10d688bfc9b6a9",
                "base": "97021ac755a14cb94c95e961c80f777397ae03471823b14a2ebd39d622e7db9c"}
SLOT_FINGERPRINT = "4ff5a52c4d2393194e9c53cb73d9a5058642706385d4fc8e17d35120b38841cb"
# The 24 held Anima-style prompts the slot term is reported on (the prompts the agreement numbers above were read on; never
# trained on).
HELD_PROMPTS = (
    "solo, adult woman, long silver hair, red coat, rainy city street, night, neon lights, umbrella, masterpiece, best quality",
    "landscape, mountain lake, sunrise, mist, pine forest, reflections, no humans, scenery, highly detailed",
    "1man, adult, beard, leather jacket, motorcycle, desert highway, sunset, dust, cinematic lighting",
    "still life, ceramic teapot, two cups, wooden table, window light, steam, soft shadows, photorealistic",
    "solo, adult woman, short black hair, glasses, library, bookshelves, reading, warm light, cozy, sitting",
    "castle on a cliff, stormy sky, lightning, waves crashing, dark fantasy, dramatic, wide shot",
    "1man, elderly, fisherman, wool sweater, harbor, boats, morning fog, seagulls, detailed face",
    "red sports car, city at night, motion blur, wet asphalt, reflections, headlights, low angle",
    "solo, adult woman, kimono, fox mask, shrine, torii gate, autumn leaves, lanterns, evening",
    "cyberpunk alley, holographic signs, steam vents, crowd, rain, purple and teal lighting, wide angle",
    "white cat, sleeping, sunlit windowsill, potted plants, curtains, peaceful, close-up",
    "two adults, dancing, ballroom, chandelier, formal wear, golden light, motion, elegant",
    "A woman in a red coat walks along a rainy street at night, holding a black umbrella under neon signs.",
    "A quiet mountain lake at sunrise, with mist rising between the pine trees and the peaks reflected in the water.",
    "An old fisherman mends his net on the dock while fog rolls in over the harbor.",
    "A ceramic teapot and two cups sit on a wooden table in soft morning light from a window.",
    "A knight in dented armor kneels in a field of red flowers as the sun sets behind him.",
    "A crowded night market with paper lanterns, food stalls and steam rising into the cold air.",
    "A small red boat drifts on a calm sea under a sky full of stars.",
    "A man in a grey suit waits alone on a train platform as the snow begins to fall.",
    "A library with tall wooden shelves, a rolling ladder and a reading lamp glowing on a desk.",
    "A dragon made of glass coils around a lighthouse during a thunderstorm.",
    "A chef plates a dessert with careful hands in a bright restaurant kitchen.",
    "An astronaut sits on the edge of a crater, looking back at the Earth rising over the horizon.",
)
SLOT_FROM = ("prompt", "tags", "caption")     # the wordings of a training item the slot prompts are drawn from
SLOT_MAX_LEN = 256                            # tokens a slot prompt keeps (Anima's prompts are short; a caption fits)

# A language-model parameter or module name in any layout this family meets: transformers 5 (model.language_model.layers.N),
# the checkpoint files (language_model.model.layers.N), a PEFT wrapper (base_model.model.model.language_model.layers.N).
_LM_LAYER = re.compile(r"(?:^|\.)language_model(?:\.model)?\.layers\.(\d+)\.(.+)$")
_LM_NORM = re.compile(r"(?:^|\.)language_model(?:\.model)?\.norm\.weight$")
_BASE_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


# ------------------------------------------------------------------------------------------------------------ the slot
def slot_part(name: str, first: int = FIRST_BASE_LAYER) -> str | None:
    """The canonical name ('layers.24.mlp.down_proj.weight', 'norm.weight') of a parameter of Anima's slot (the language model's
    layers first..27 and its final norm), in any of the layouts above; None for every other parameter."""
    m = _LM_LAYER.search(name)
    if m and int(m.group(1)) >= first:
        return f"layers.{m.group(1)}.{m.group(2)}"
    if _LM_NORM.search(name):
        return "norm.weight"
    return None


def is_slot(name: str, first: int = FIRST_BASE_LAYER) -> bool:
    return slot_part(name, first) is not None


def lora_targets(first: int = FIRST_BASE_LAYER) -> str:
    """The LoRA targets of the hybrid: the language model's attention and MLP projections in layers 0..first-1 only (Anima's slot
    carries none; the vision tower and the projector are left alone, as for every family). A regex PEFT matches in full."""
    below = "|".join(str(i) for i in range(first))
    return (rf".*language_model(?:\.model)?\.layers\.({below})\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$")


def _bf16_bytes(t) -> bytes:
    import torch
    t = t.detach().to("cpu")
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.contiguous().view(torch.int16).numpy().tobytes()


def fingerprint(named) -> str:
    """sha256 over (name, shape, bf16 bytes) of every tensor, in name order. named: a dict name -> tensor."""
    h = hashlib.sha256()
    for k in sorted(named):
        t = named[k]
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(_bf16_bytes(t))
    return h.hexdigest()


def slot_fingerprint(named, first: int = FIRST_BASE_LAYER) -> str:
    """The fingerprint of Anima's slot alone, under its canonical names (the same in every layout and wrapper). The Base's
    own layers first..27 and final norm give the same value: the slot is the Base's, tensor for tensor."""
    slot = {}
    for k, v in named.items():
        part = slot_part(k, first)
        if part is not None:
            assert part not in slot, f"two tensors map to the slot's {part}"
            slot[part] = v
    per_layer = {}
    for p in slot:
        if p.startswith("layers."):
            per_layer[p.split(".")[1]] = per_layer.get(p.split(".")[1], 0) + 1
    assert "norm.weight" in slot and per_layer, "the slot needs its layers and the final norm"
    assert len(set(per_layer.values())) == 1, f"the slot's layers hold different numbers of tensors: {per_layer}"
    return fingerprint(slot)


def base_slot(base: dict, first: int = FIRST_BASE_LAYER) -> dict:
    """The Base's own slot tensors under their canonical names (keys of the Base's file: model.layers.N.*, model.norm.weight)."""
    out = {}
    for k, v in base.items():
        m = _BASE_LAYER.match(k)
        if m and int(m.group(1)) >= first:
            out[f"layers.{m.group(1)}.{m.group(2)}"] = v
        elif k == "model.norm.weight":
            out["norm.weight"] = v
    return out


# ---------------------------------------------------------------------------------------------------------- the build
def compose(internvl: dict, base: dict, head: str = HEAD, first: int = FIRST_BASE_LAYER) -> dict:
    """The hybrid's tensors, in the InternVL3_5-1B-HF file's own layout: InternVL's every tensor, with the language model's
    layers first..27 and final norm replaced by the Base's (keys model.layers.N.*, model.norm.weight of the Base's file), and
    with head "base" the output head replaced by the Base's word table (its head is tied to it). Shapes and dtypes must agree."""
    assert head in HEADS, f"head is one of {HEADS}, not {head!r}"
    out = dict(internvl)
    put = 0
    for part, v in base_slot(base, first).items():
        target = f"language_model.model.{part}"
        assert target in out, f"InternVL's file has no {target}"
        assert tuple(out[target].shape) == tuple(v.shape) and out[target].dtype == v.dtype, f"{target}: shape or dtype differs"
        out[target] = v
        put += 1
    assert put > 1, f"{put} slot tensors put back"
    if head == "base":
        h = "language_model.lm_head.weight"
        assert tuple(out[h].shape) == tuple(base["model.embed_tokens.weight"].shape)
        out[h] = base["model.embed_tokens.weight"]
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch_pinned(source: dict, cache_dir: str | None = None, local_dir: str | None = None) -> str:
    """The source's files at its pinned commit, as a local folder (a public repo: no token is sent); its weights file's size
    and sha256 are checked against the record pinned here (the Hub's own LFS record). local_dir: a folder already holding the
    files (a test, or a machine that has them): they are checked, not downloaded."""
    if local_dir is None:
        from huggingface_hub import hf_hub_download
        for f in source["files"] + (source["weights"],):
            p = hf_hub_download(source["repo"], f, revision=source["revision"], cache_dir=cache_dir, token=False)
        local_dir = os.path.dirname(p)
    w = os.path.join(local_dir, source["weights"])
    size = os.path.getsize(w)
    assert size == source["bytes"], f"{source['repo']}: {size:,} bytes, not {source['bytes']:,}"
    got = sha256_file(w)
    assert got == source["sha256"], f"{source['repo']}@{source['revision'][:7]}: sha256 {got} is not the pinned {source['sha256']}"
    return local_dir


def record(head: str = HEAD, first: int = FIRST_BASE_LAYER, fp: str | None = None, slot_fp: str | None = None) -> dict:
    from . import __version__
    return {"name": NAME, "first_base_layer": first, "head": head,
            "internvl": {k: INTERNVL[k] for k in ("repo", "revision", "sha256", "license")},
            "base": {k: BASE[k] for k in ("repo", "revision", "sha256", "license")},
            "fingerprint": fp, "slot_fingerprint": slot_fp, "made_by": f"qwen-symbolic-thinking {__version__}",
            "what": "InternVL3.5-1B's vision tower, projector, word table and language layers 0-23; Qwen3-0.6B-Base's "
                    "(Anima's text encoder's) layers 24-27 and final norm" + ("; the Base's word table as the output head"
                                                                              if head == "base" else "; InternVL's output head")}


def build(out_dir: str, head: str = HEAD, cache_dir: str | None = None, internvl_dir: str | None = None,
          base_dir: str | None = None, check: bool = True) -> str:
    """Write the hybrid's folder (out_dir/<NAME>) and return its path: model.safetensors (bf16, the InternVL3_5-1B-HF layout),
    InternVL's processor, tokenizer and chat template, generation_config.json, config.json (tie_word_embeddings false; the
    record under "anima_eyes") and anima_eyes.json (the same record). With check, the composed hybrid's fingerprint must equal
    the one pinned for its head (FINGERPRINTS) and its slot's the pinned SLOT_FINGERPRINT (a value still None is computed and
    printed, not checked: the pinning run)."""
    from safetensors.torch import load_file, save_file
    t0 = time.time()
    ivl = fetch_pinned(INTERNVL, cache_dir, internvl_dir)
    bas = fetch_pinned(BASE, cache_dir, base_dir)
    tensors = compose(load_file(os.path.join(ivl, INTERNVL["weights"])), load_file(os.path.join(bas, BASE["weights"])), head)
    fp = fingerprint(tensors)
    sfp = slot_fingerprint(tensors)
    if check:
        if FINGERPRINTS.get(head) is not None:
            assert fp == FINGERPRINTS[head], f"the composed hybrid's fingerprint {fp} is not the pinned {FINGERPRINTS[head]}"
        if SLOT_FINGERPRINT is not None:
            assert sfp == SLOT_FINGERPRINT, f"the slot's fingerprint {sfp} is not the pinned {SLOT_FINGERPRINT}"
    folder = os.path.join(out_dir, NAME)
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(folder, "model.safetensors"), metadata={"format": "pt"})
    for f in INTERNVL["files"]:
        if f != "config.json":
            shutil.copyfile(os.path.join(ivl, f), os.path.join(folder, f))
    with open(os.path.join(ivl, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    rec = record(head, FIRST_BASE_LAYER, fp, sfp)
    cfg["tie_word_embeddings"] = False
    cfg.setdefault("text_config", {})["tie_word_embeddings"] = False
    cfg["text_config"]["_name_or_path"] = f"{INTERNVL['repo']} layers 0-23 + {BASE['repo']} layers 24-27"
    cfg[CONFIG_KEY] = rec
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(folder, "anima_eyes.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({**rec, "seconds": round(time.time() - t0, 1)}, fh, indent=1)
        fh.write("\n")
    print(f"anima eyes: built {folder} in {time.time() - t0:.0f} s (head {head}; fingerprint {fp[:16]}...; slot {sfp[:16]}...)",
          flush=True)
    return folder


def check_slot(model, first: int = FIRST_BASE_LAYER, expected: str | None = None) -> str:
    """Anima's slot, read off a loaded model (any dtype, wrapped or not), must be the Base's: its fingerprint against the record's
    (or the pinned one). Returns the fingerprint; raises when it differs. A LoRA module on the slot is refused too."""
    named = {k: v for k, v in model.named_parameters()}
    hit = [k for k in named if is_slot(k, first) and ("lora_" in k)]
    assert not hit, f"LoRA modules on Anima's slot: {hit[:3]}"
    named = {k.replace(".base_layer.", "."): v for k, v in named.items()}
    got = slot_fingerprint(named, first)
    want = expected if expected is not None else (SLOT_FINGERPRINT if first == FIRST_BASE_LAYER else None)
    if want is not None:
        assert got == want, f"Anima's slot was changed: fingerprint {got} is not {want}"
    return got


# ------------------------------------------------------------------------------------------------------- the slot term
def slot_prompts(items: list, wordings: tuple = SLOT_FROM) -> list:
    """The Anima-style prompts of a training set: each item's wordings named in `wordings` (its generation prompt, its tags,
    its caption), stripped, repeats dropped (case-folded), in the items' order."""
    out, seen = [], set()
    for it in items:
        w = it.get("wordings") or {}
        for k in wordings:
            text = str(w.get(k) or "").strip()
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                out.append(text)
    return out


def encode_prompts(tokenizer, prompts: list, max_len: int = SLOT_MAX_LEN) -> dict:
    """Prompts as Anima's text path reads them: the bare text (no chat template, no start token), right-padded."""
    import torch
    ids = [tokenizer(p, add_special_tokens=False)["input_ids"][:max_len] for p in prompts]
    n = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((len(ids), n), pad, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return {"input_ids": input_ids, "attention_mask": mask}


class SlotCollator:
    """A training micro-batch with Anima-style prompts beside it: the inner collator's batch (the training rows, and the general
    rows when it pairs them), plus batch["slot"] = `k` prompts drawn in a seeded order and cycled, encoded as Anima reads them.
    advance() replays the draws of micro-batches already trained (a resumed run), the inner collator's first."""

    def __init__(self, inner, prompts: list, k: int, seed: int, tokenizer, max_len: int = SLOT_MAX_LEN):
        if not prompts:
            raise ValueError("no Anima-style prompts to pair")
        self.inner, self.prompts, self.k, self.tokenizer, self.max_len = inner, list(prompts), int(k), tokenizer, max_len
        self.rng = random.Random(seed + 7)
        self.order, self.at = [], 0
        self.counts = getattr(inner, "counts", None)

    def _next(self, k: int) -> list:
        out = []
        while len(out) < k:
            if self.at >= len(self.order):
                self.order = list(range(len(self.prompts)))
                self.rng.shuffle(self.order)
                self.at = 0
            out.append(self.prompts[self.order[self.at]])
            self.at += 1
        return out

    def __call__(self, pairs: list) -> dict:
        batch = self.inner(pairs)
        if self.k > 0:
            batch["slot"] = encode_prompts(self.tokenizer, self._next(self.k), self.max_len)
        return batch

    def advance(self, batches: int) -> None:
        if hasattr(self.inner, "advance"):
            self.inner.advance(batches)
        for _ in range(int(batches)):
            if self.k > 0:
                self._next(self.k)


def text_model(model):
    """The language model inside a model (unwrapped from PEFT and from the Trainer's wrappers): its forward returns the final
    states after the last norm (InternVL: model.model.language_model, a Qwen3Model)."""
    m = model
    for attr in ("module",):                                  # a DDP or accelerate wrapper
        m = getattr(m, attr, m)
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    inner = getattr(m, "model", m)
    lm = getattr(inner, "language_model", None)
    assert lm is not None, f"{type(m).__name__} has no language model to read final states from"
    return lm


def slot_loss(h_model, h_anima, mask) -> tuple:
    """(L_slot, the raw cosine's mean): the relative squared error per real token, ||h_model - h_anima||^2 / ||h_anima||^2,
    averaged over the real tokens (mask 1), in fp32; the teacher side carries no gradient."""
    import torch
    m = mask.bool()
    a = h_model.float()[m]
    b = h_anima.float()[m].detach()
    if a.shape[0] == 0:
        return h_model.sum() * 0.0, 0.0
    rel = ((a - b) ** 2).sum(-1) / (b ** 2).sum(-1).clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(a.detach(), b, dim=-1)
    return rel.mean(), float(cos.mean())


class SlotTerm:
    """The slot term's teacher (Anima's encoder, frozen) and its loss on a micro-batch's Anima-style prompts."""

    def __init__(self, teacher):
        self.teacher = teacher

    def loss(self, model, slot: dict) -> tuple:
        import torch
        dev = next(self.teacher.parameters()).device
        ids, mask = slot["input_ids"].to(dev), slot["attention_mask"].to(dev)
        with torch.no_grad():
            h_anima = self.teacher(input_ids=ids, attention_mask=mask).last_hidden_state
        h_model = text_model(model)(input_ids=ids, attention_mask=mask).last_hidden_state
        loss, cos = slot_loss(h_model, h_anima, mask)
        return loss, {"slot": float(loss.detach()), "slot_cos": cos}


def agreement(base_states: list, other_states: list) -> dict:
    """The read of two encoders' final states on the same prompts that the numbers above come from: the centered cosine (the
    Base's mean state over every prompt token after the first removed from both), its median, the raw cosine, the first token's
    cosine (read apart: the attention sink), and the relative difference's median."""
    import torch
    A = torch.cat([s[1:] for s in base_states]).float()
    B = torch.cat([s[1:] for s in other_states]).float()
    mu = A.mean(0, keepdim=True)
    cen = torch.nn.functional.cosine_similarity(A - mu, B - mu, dim=-1)
    raw = torch.nn.functional.cosine_similarity(A, B, dim=-1)
    rel = (B - A).norm(dim=-1) / A.norm(dim=-1).clamp_min(1e-6)
    first = torch.stack([torch.nn.functional.cosine_similarity(a[:1].float(), b[:1].float(), dim=-1)[0]
                         for a, b in zip(base_states, other_states)])
    return {"slot_held": round(float(cen.mean()), 4), "slot_held_median": round(float(cen.median()), 4),
            "slot_held_raw": round(float(raw.mean()), 4), "slot_held_first": round(float(first.mean()), 4),
            "slot_held_rel": round(float(rel.median()), 4)}


class SlotHeld:
    """The slot term's report on the held prompts: the teacher's states once, the model's at each call (eval mode, no gradient,
    the training mode put back)."""

    def __init__(self, teacher, tokenizer, prompts: tuple = HELD_PROMPTS):
        self.tokenizer, self.prompts = tokenizer, list(prompts)
        self.base_states = self._states(teacher)

    def _states(self, m) -> list:
        import torch
        dev = next(m.parameters()).device
        out = []
        with torch.no_grad():
            for p in self.prompts:
                ids = torch.tensor([self.tokenizer(p, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=dev)
                out.append(m(input_ids=ids).last_hidden_state[0].float().cpu())
        return out

    def __call__(self, model) -> dict:
        import torch
        lm = text_model(model)
        was = lm.training
        lm.eval()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                states = self._states(lm)
        finally:
            lm.train(was)
        return agreement(self.base_states, states)


def load_teacher(device: str = "cuda", cache_dir: str | None = None, local_dir: str | None = None):
    """Anima's encoder for the slot term: Qwen/Qwen3-0.6B-Base at its pinned commit (its weights' sha256 checked), loaded as a
    plain Qwen3Model in bf16, frozen, in eval mode."""
    import torch
    from transformers import AutoModel
    folder = fetch_pinned(BASE, cache_dir, local_dir)
    teacher = AutoModel.from_pretrained(folder, dtype=torch.bfloat16)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
