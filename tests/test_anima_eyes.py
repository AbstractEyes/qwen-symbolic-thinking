"""Anima's eyes (anima_eyes.py; family.anima_eyes, lora_targets, frozen; train's slot term and reference copy; keep_trainer's
slot and reference): the slot's names in every layout, the composition, the fingerprints, the slot term's loss, its collator
and its report, on tiny tensors and a tiny InternVL built on the CPU. The tests marked "local" read the real files (the two
pinned sources and the ComfyUI A/B's last-4 file) where a machine has them, and skip elsewhere."""
import copy
import os
import random
import re

import pytest

from qwen_symbolic_thinking import anima_eyes, family, keep, train

FIRST = anima_eyes.FIRST_BASE_LAYER


# ------------------------------------------------------------------------------------------------------- names and targets
def test_the_slot_is_named_alike_in_every_layout():
    for name, part in [("model.language_model.layers.24.self_attn.q_proj.weight", "layers.24.self_attn.q_proj.weight"),
                       ("language_model.model.layers.27.mlp.down_proj.weight", "layers.27.mlp.down_proj.weight"),
                       ("base_model.model.model.language_model.layers.25.input_layernorm.weight",
                        "layers.25.input_layernorm.weight"),
                       ("model.language_model.norm.weight", "norm.weight"),
                       ("language_model.model.norm.weight", "norm.weight")]:
        assert anima_eyes.slot_part(name) == part, name
    for name in ["model.language_model.layers.23.self_attn.q_proj.weight", "model.language_model.embed_tokens.weight",
                 "lm_head.weight", "language_model.lm_head.weight", "model.vision_tower.encoder.layer.24.attention.q_proj.weight",
                 "model.vision_tower.encoder.layer.0.layernorm_after.weight", "model.multi_modal_projector.layer_norm.weight",
                 "model.language_model.layers.3.post_attention_layernorm.weight"]:
        assert anima_eyes.slot_part(name) is None, name


def test_the_lora_targets_stop_below_the_slot():
    pat = re.compile(anima_eyes.lora_targets())
    for i in (0, 7, 12, 23):
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                     "mlp.up_proj", "mlp.down_proj"):
            assert pat.fullmatch(f"model.language_model.layers.{i}.{proj}"), (i, proj)
            assert pat.fullmatch(f"base_model.model.model.language_model.layers.{i}.{proj}"), (i, proj)
    for name in ["model.language_model.layers.24.self_attn.q_proj", "model.language_model.layers.27.mlp.down_proj",
                 "model.vision_tower.encoder.layer.0.attention.q_proj", "model.multi_modal_projector.linear_1", "lm_head",
                 "model.language_model.layers.3.self_attn.q_norm"]:
        assert not pat.fullmatch(name), name


# ------------------------------------------------------------------------------------------------ composition, fingerprints
def _fake(torch, n_layers=4, width=4, seed=0, layout="file"):
    """Tensors shaped like the two files: InternVL's (language_model.model.*, language_model.lm_head, vision_tower.*) or the
    Base's (model.*), in bf16."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    pre = "language_model.model." if layout == "file" else "model."
    for i in range(n_layers):
        for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight"):
            out[f"{pre}layers.{i}.{k}"] = torch.randn(width, width, generator=g).to(torch.bfloat16)
    out[f"{pre}norm.weight"] = torch.randn(width, generator=g).to(torch.bfloat16)
    out[f"{pre}embed_tokens.weight"] = torch.randn(6, width, generator=g).to(torch.bfloat16)
    if layout == "file":
        out["language_model.lm_head.weight"] = torch.randn(6, width, generator=g).to(torch.bfloat16)
        out["vision_tower.encoder.layer.0.attention.q_proj.weight"] = torch.randn(width, width, generator=g).to(torch.bfloat16)
    return out


def test_compose_puts_the_base_slot_back_and_nothing_else():
    torch = pytest.importorskip("torch")
    ivl, base = _fake(torch, seed=1), _fake(torch, seed=2, layout="base")
    hy = anima_eyes.compose(ivl, base, "internvl", first=2)
    assert set(hy) == set(ivl)
    for k in ivl:
        part = anima_eyes.slot_part(k, 2)
        want = base[f"model.{part}"] if part else ivl[k]
        assert torch.equal(hy[k], want), k
    hb = anima_eyes.compose(ivl, base, "base", first=2)
    assert torch.equal(hb["language_model.lm_head.weight"], base["model.embed_tokens.weight"])
    assert torch.equal(hb["language_model.model.embed_tokens.weight"], ivl["language_model.model.embed_tokens.weight"])
    with pytest.raises(AssertionError):
        anima_eyes.compose(ivl, base, "other", first=2)
    bad = dict(base)
    bad["model.layers.3.mlp.down_proj.weight"] = bad["model.layers.3.mlp.down_proj.weight"].float()
    with pytest.raises(AssertionError):
        anima_eyes.compose(ivl, bad, "internvl", first=2)


def test_the_slot_fingerprint_is_the_bases_in_every_layout_and_moves_with_any_change():
    torch = pytest.importorskip("torch")
    ivl, base = _fake(torch, seed=1), _fake(torch, seed=2, layout="base")
    hy = anima_eyes.compose(ivl, base, "internvl", first=2)
    fp = anima_eyes.slot_fingerprint(hy, 2)
    assert fp == anima_eyes.fingerprint(anima_eyes.base_slot(base, 2))
    as_model = {k.replace("language_model.model.", "model.language_model."): v.float() for k, v in hy.items()}
    assert anima_eyes.slot_fingerprint(as_model, 2) == fp                        # fp32 copies of bf16 values: the same bytes
    peft = {"base_model.model." + k: v for k, v in as_model.items()}
    assert anima_eyes.slot_fingerprint(peft, 2) == fp
    assert anima_eyes.fingerprint(dict(reversed(list(hy.items())))) == anima_eyes.fingerprint(hy)   # name order, not dict order
    moved = dict(hy)
    moved["language_model.model.layers.3.mlp.down_proj.weight"] = moved["language_model.model.layers.3.mlp.down_proj.weight"] * 2
    assert anima_eyes.slot_fingerprint(moved, 2) != fp
    below = dict(hy)
    below["language_model.model.layers.0.mlp.down_proj.weight"] = below["language_model.model.layers.0.mlp.down_proj.weight"] * 2
    assert anima_eyes.slot_fingerprint(below, 2) == fp and anima_eyes.fingerprint(below) != anima_eyes.fingerprint(hy)


# --------------------------------------------------------------------------------------------------------- the slot term
def test_the_slot_loss_is_the_relative_squared_error_on_real_tokens():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(3)
    b = torch.randn(2, 5, 8, generator=g)
    a = b + 0.1 * torch.randn(2, 5, 8, generator=g)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    loss, cos = anima_eyes.slot_loss(a, b, mask)
    m = mask.bool()
    want = (((a[m] - b[m]) ** 2).sum(-1) / (b[m] ** 2).sum(-1)).mean()
    assert torch.allclose(loss, want, atol=1e-6) and 0.9 < cos <= 1.0
    zero, one = anima_eyes.slot_loss(b.clone(), b, mask)
    assert float(zero) == pytest.approx(0.0, abs=1e-7) and one == pytest.approx(1.0, abs=1e-6)
    a2 = a.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)
    anima_eyes.slot_loss(a2, b2, mask)[0].backward()
    assert a2.grad is not None and float(a2.grad[0, 3:].abs().sum()) == 0.0 and b2.grad is None   # padding and teacher: no grad


def test_the_slot_prompts_come_from_the_items_wordings_once_each():
    items = [{"wordings": {"caption": "A cat.", "prompt": "cat, window", "tags": "cat, window"}},
             {"wordings": {"caption": "a CAT.", "prompt": "", "tags": "dog"}}, {"kind": "spec"}]
    assert anima_eyes.slot_prompts(items) == ["cat, window", "A cat.", "dog"]
    assert anima_eyes.slot_prompts(items, ("caption",)) == ["A cat."]


class _Tok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        assert add_special_tokens is False            # Anima's text path: the bare text, no start token
        return {"input_ids": [1 + ord(c) % 50 for c in text]}       # ids inside a tiny model's vocabulary


def test_the_slot_collator_draws_in_a_seeded_cycle_and_replays_on_resume():
    pytest.importorskip("torch")
    pool = [f"p{i}" for i in range(5)]

    def inner(pairs):
        return {"rows": list(pairs)}

    a = anima_eyes.SlotCollator(inner, pool, 2, seed=7, tokenizer=_Tok())
    drawn = [a._next(2) for _ in range(5)]
    flat = [p for d in drawn for p in d]
    assert sorted(flat[:5]) == sorted(pool) and sorted(flat[5:10]) == sorted(pool)     # every prompt once a cycle
    b = anima_eyes.SlotCollator(inner, pool, 2, seed=7, tokenizer=_Tok())
    b.advance(3)
    assert b._next(2) == drawn[3]
    batch = anima_eyes.SlotCollator(inner, ["ab", "abcd"], 2, seed=1, tokenizer=_Tok())(["x"])
    assert batch["rows"] == ["x"] and batch["slot"]["input_ids"].shape == (2, 4)
    assert sorted(batch["slot"]["attention_mask"].sum(1).tolist()) == [2, 4]
    with pytest.raises(ValueError):
        anima_eyes.SlotCollator(inner, [], 2, 1, _Tok())


def test_the_agreement_reads_one_on_identical_states_and_centers_on_the_bases_mean():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(4)
    base = [torch.randn(n, 6, generator=g) for n in (4, 5, 3)]
    same = anima_eyes.agreement(base, [s.clone() for s in base])
    assert same["slot_held"] == pytest.approx(1.0) and same["slot_held_first"] == pytest.approx(1.0)
    assert same["slot_held_rel"] == pytest.approx(0.0, abs=1e-6)
    shifted = anima_eyes.agreement(base, [s + 3.0 for s in base])              # a common shift: raw cosine falls
    A = torch.cat([s[1:] for s in base])
    mu = A.mean(0, keepdim=True)
    want = torch.nn.functional.cosine_similarity(A - mu, A + 3.0 - mu, dim=-1).mean()
    assert shifted["slot_held"] == pytest.approx(round(float(want), 4))


# ----------------------------------------------------------------------------------------------- a tiny InternVL, CPU
HF = r"X:\anima_qwen_eyes\hf_home\hub\models--OpenGVLab--InternVL3_5-1B-HF\snapshots\9191dbccf312b537016f041b25d61c72e7c5c9f3"
BASE_DIR = r"X:\anima_qwen_eyes\hf_home\hub\models--Qwen--Qwen3-0.6B-Base\snapshots\da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
BASE_FILE = r"E:\cache\anima_local\data\models\split_files\text_encoders\qwen_3_06b_base.safetensors"
LAST4_FILE = r"X:\anima_qwen_eyes\comfy\text_encoders\qwen_3_06b_hybrid_internvl_layers_base_last4.safetensors"


def tiny_internvl(first=2, record=True):
    """An InternVL of 4 language layers (width 32) and 1 vision layer, random, with the anima_eyes record for a slot of the
    layers from `first` up; built from transformers' own config class, no download."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from transformers import InternVLConfig, InternVLForConditionalGeneration, Qwen3Config
    text = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    cfg = InternVLConfig(text_config=text.to_dict(), vision_config={"hidden_size": 32, "intermediate_size": 64,
                                                                    "num_hidden_layers": 1, "num_attention_heads": 4,
                                                                    "image_size": [28, 28], "patch_size": [14, 14]},
                         image_token_id=63, tie_word_embeddings=False)
    if record:
        cfg.anima_eyes = {"name": "tiny", "first_base_layer": first, "head": "internvl", "slot_fingerprint": None}
    torch.manual_seed(0)
    return InternVLForConditionalGeneration(cfg), tf


def test_the_family_reads_the_record_and_keeps_the_slot_out_of_training():
    torch = pytest.importorskip("torch")
    model, _ = tiny_internvl()
    rec = family.anima_eyes(model)
    assert rec["first_base_layer"] == 2 and family.anima_eyes(tiny_internvl(record=False)[0]) is None
    pat = re.compile(family.lora_targets(model))
    assert family.lora_targets(tiny_internvl(record=False)[0]) is None
    names = [n for n, _ in model.named_modules()]
    hit = [n for n in names if pat.fullmatch(n)]
    assert hit and all(int(n.split(".layers.")[1].split(".")[0]) < 2 for n in hit) and len(hit) == 2 * 7
    frozen = [n for n, _ in model.named_parameters() if family.frozen(model, n)]
    assert frozen and all(anima_eyes.is_slot(n, 2) for n in frozen)
    assert any(n.endswith("language_model.norm.weight") for n in frozen)
    assert not any("vision_tower" in n or "embed_tokens" in n or "lm_head" in n for n in frozen)
    fp = anima_eyes.check_slot(model, 2)
    note = family.prepare_model(model)
    assert note.startswith("Anima's eyes") and fp[:16] in note
    model.config.anima_eyes["slot_fingerprint"] = fp
    with torch.no_grad():
        model.model.language_model.layers[3].mlp.down_proj.weight.mul_(2)
    with pytest.raises(AssertionError, match="slot was changed"):
        family.prepare_model(model)
    tied, _ = tiny_internvl()                          # a head tied to the word table at load would have replaced the head
    tied.lm_head.weight = tied.model.language_model.embed_tokens.weight
    with pytest.raises(ValueError, match="tied to the word table"):
        family.prepare_model(tied)


def test_a_lora_on_the_tiny_hybrid_has_no_module_on_the_slot():
    pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    model, _ = tiny_internvl()
    lora = peft.get_peft_model(model, peft.LoraConfig(r=2, lora_alpha=4, target_modules=family.lora_targets(model),
                                                      task_type=peft.TaskType.CAUSAL_LM))
    report = train.lora_report(lora)
    assert sum(report.values()) == 14 and set(report) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                         "down_proj"}
    anima_eyes.check_slot(lora, 2)                     # no LoRA module on the slot; its fingerprint reads through the wrapper
    assert not [n for n, p in lora.named_parameters() if p.requires_grad and anima_eyes.is_slot(n, 2)]


def test_the_slot_term_moves_the_layers_below_the_slot_only():
    torch = pytest.importorskip("torch")
    model, _ = tiny_internvl()
    from transformers import Qwen3Model
    teacher = Qwen3Model(model.config.text_config)
    teacher.load_state_dict(model.model.language_model.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    term = anima_eyes.SlotTerm(teacher)
    slot = {"input_ids": torch.tensor([[5, 6, 7, 0], [8, 9, 10, 11]]), "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
    loss, parts = term.loss(model, slot)
    assert float(loss) == pytest.approx(0.0, abs=1e-10) and parts["slot_cos"] == pytest.approx(1.0, abs=1e-5)
    for n, p in model.named_parameters():                  # the full fine-tune's settings
        p.requires_grad = train.trains_in_full(n) and not family.frozen(model, n)
    with torch.no_grad():
        model.model.language_model.layers[0].mlp.up_proj.weight.add_(0.05)
    loss, parts = term.loss(model, slot)
    assert float(loss) > 0 and parts["slot"] == pytest.approx(float(loss))
    loss.backward()
    lm = model.model.language_model
    assert lm.layers[0].mlp.up_proj.weight.grad is not None and float(lm.layers[0].mlp.up_proj.weight.grad.abs().sum()) > 0
    assert lm.layers[2].mlp.up_proj.weight.grad is None and lm.norm.weight.grad is None
    held = anima_eyes.SlotHeld(teacher, _Tok(), prompts=("abc", "abcdef"))
    lm.train()
    read = held(model)
    assert set(read) == {"slot_held", "slot_held_median", "slot_held_raw", "slot_held_first", "slot_held_rel"} and lm.training


def test_keep_trainer_reads_the_base_on_a_reference_copy_for_a_full_finetune(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import TrainingArguments
    model, _ = tiny_internvl()
    reference = copy.deepcopy(model).eval()
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=1, report_to=[], use_cpu=True,
                             remove_unused_columns=False)
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    lab = torch.tensor([[-100, -100, 7, 8, 9]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab,
             "anchor": {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}}
    stats = {}
    trainer = keep.keep_trainer(general_kl=1.0, stats=stats, reference=reference)(model=model, args=args)
    loss = trainer.compute_loss(model, copy.deepcopy(batch))
    assert torch.isfinite(loss) and stats["general_kl"] == pytest.approx(0.0, abs=1e-6)    # the copy: no drift yet
    with pytest.raises(ValueError, match="neither is here"):
        keep.keep_trainer(general_kl=1.0)(model=model, args=args).compute_loss(model, copy.deepcopy(batch))
    with pytest.raises(ValueError, match="teacher"):
        keep.keep_trainer(slot=1.0)


def test_a_full_finetune_trains_two_steps_with_the_reference_and_the_slot_term(tmp_path):
    """The Trainer loop itself (CPU, tiny shapes): the general rows' KL on the reference copy and the slot term ride beside the
    training rows through gradient accumulation; the slot stays the Base's, the layers below it move."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3Model, TrainingArguments
    model, _ = tiny_internvl()
    fp0 = anima_eyes.check_slot(model, 2)
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad = False
    teacher = Qwen3Model(model.config.text_config)
    teacher.load_state_dict(model.model.language_model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        p.requires_grad = train.trains_in_full(n) and not family.frozen(model, n)
    before = model.model.language_model.layers[0].mlp.up_proj.weight.detach().clone()
    g = torch.Generator().manual_seed(5)

    def row():
        ids = torch.randint(1, 60, (6,), generator=g)
        lab = ids.clone()
        lab[:3] = -100
        return ids, lab

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return i

    def inner(pairs):
        rows = [row() for _ in pairs]
        ids = torch.stack([r[0] for r in rows])
        lab = torch.stack([r[1] for r in rows])
        a_ids, a_lab = row()
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab,
                "anchor": {"input_ids": a_ids[None], "attention_mask": torch.ones_like(a_ids[None]), "labels": a_lab[None]}}

    collate = anima_eyes.SlotCollator(inner, ["abc", "abcdefg", "ab"], 2, seed=1, tokenizer=_Tok())
    stats = {}
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=1, gradient_accumulation_steps=2,
                             max_steps=2, learning_rate=1e-2, report_to=[], use_cpu=True, remove_unused_columns=False,
                             save_strategy="no", logging_steps=1)
    trainer_class = keep.keep_trainer(general_kl=1.0, stats=stats, slot=1.0, slot_term=anima_eyes.SlotTerm(teacher),
                                      reference=reference)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    trainer = trainer_class(model=model, args=args, train_dataset=Rows(), data_collator=collate, optimizers=(optimizer, None))
    result = trainer.train()
    assert result.global_step == 2 and stats.get("n", 0) >= 4 and "slot" in stats and "general_kl" in stats
    assert not torch.equal(model.model.language_model.layers[0].mlp.up_proj.weight, before)
    assert anima_eyes.check_slot(model, 2) == fp0


# ------------------------------------------------------------------------------------- local: the real files, where present
local = pytest.mark.skipif(not (os.path.isdir(HF) and os.path.isdir(BASE_DIR) and os.path.isfile(BASE_FILE)),
                           reason="the pinned sources are not on this machine")


@local
def test_local_the_composed_language_model_is_the_comfyui_last4_file():
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    from safetensors.torch import load_file
    hy = anima_eyes.compose(load_file(os.path.join(HF, "model.safetensors")), load_file(BASE_FILE), "internvl")
    if os.path.isfile(LAST4_FILE):
        with safe_open(LAST4_FILE, "pt") as f:
            keys = list(f.keys())
            assert len(keys) == 310
            for k in keys:
                assert torch.equal(hy["language_model." + k], f.get_tensor(k)), k
    assert anima_eyes.slot_fingerprint(hy) == anima_eyes.fingerprint(anima_eyes.base_slot(load_file(BASE_FILE)))
    for head in anima_eyes.HEADS:
        want = anima_eyes.FINGERPRINTS.get(head)
        if want is not None:
            got = anima_eyes.fingerprint(hy if head == "internvl" else anima_eyes.compose(
                load_file(os.path.join(HF, "model.safetensors")), load_file(BASE_FILE), "base"))
            assert got == want, head
    if anima_eyes.SLOT_FINGERPRINT is not None:
        assert anima_eyes.slot_fingerprint(hy) == anima_eyes.SLOT_FINGERPRINT


@local
def test_local_the_processor_speaks_anima_s_tokens_and_ends_its_turn():
    pytest.importorskip("transformers")
    from transformers import AutoProcessor, AutoTokenizer
    from qwen_symbolic_thinking.template import turn_end
    proc = AutoProcessor.from_pretrained(HF)
    base_tok = AutoTokenizer.from_pretrained(BASE_DIR)
    for p in anima_eyes.HELD_PROMPTS:                 # the slot prompts tokenize the same for the model and the Base
        assert proc.tokenizer(p, add_special_tokens=False)["input_ids"] == base_tok(p, add_special_tokens=False)["input_ids"]
    assert turn_end(proc) == "<|im_end|>\n" and family.thinking_kwargs(proc, False) == {}


@local
def test_local_internvl_s_processor_pads_on_its_tokenizer_s_side():
    pytest.importorskip("transformers")
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(HF)
    proc.tokenizer.padding_side = "right"
    assert proc(text=["a", "a b c d"], return_tensors="pt", padding=True)["attention_mask"][0, -1] == 1   # its own default: left
    assert family.follow_padding_side(proc) and not family.follow_padding_side(proc)
    assert proc(text=["a", "a b c d"], return_tensors="pt", padding=True)["attention_mask"][0].tolist()[0] == 1
    assert proc(text=["a", "a b c d"], return_tensors="pt", padding=True)["attention_mask"][0].tolist()[-1] == 0
    proc.tokenizer.padding_side = "left"
    assert proc(text=["a", "a b c d"], return_tensors="pt", padding=True)["attention_mask"][0].tolist()[0] == 0
    assert type(proc).__name__ == "InternVLProcessor"


@local
def test_local_the_collator_masks_the_prompt_and_the_picture():
    torch = pytest.importorskip("torch")
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(HF)
    proc.tokenizer.padding_side = "right"                  # as train.run, then the family's own preparation
    family.prepare_processor(proc, "internvl")
    item = {"kind": "image", "id": "p1", "depth": "simple", "thoughts": [{"step": "see", "text": "A red square."}],
            "wordings": {"caption": "A red square on white."}, "questions": [], "spec": {}}
    spec = {"kind": "spec", "id": "s1", "depth": "simple", "spec": {"thing": {"name": "cube"}},
            "thoughts": [{"step": "see", "text": "A cube."}], "wordings": {"caption": "A cube."}, "questions": []}
    pics = {"p1": Image.new("RGB", (64, 64), (200, 30, 30))}
    collate = train.Collator(proc, pics, 2048, kind="internvl")
    batch = collate([(item, "short"), (spec, "spec_short")])
    ids, lab = batch["input_ids"], batch["labels"]
    img = proc.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    assert int((ids[0] == img).sum()) == 256 and int((ids[1] == img).sum()) == 0     # one 448-pixel tile: 256 picture tokens
    assert bool((lab[0][ids[0] == img] == -100).all())
    for row, it in ((0, item), (1, spec)):                  # the labels are the answer and the turn end, nothing else
        taught = proc.tokenizer.decode(lab[row][lab[row] != -100])
        assert taught == "see: " + it["thoughts"][0]["text"] + "\ncaption: " + it["wordings"]["caption"] + "<|im_end|>\n", taught
    assert "pixel_values" in batch and batch["pixel_values"].shape[0] == 1
