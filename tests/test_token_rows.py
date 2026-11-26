"""Token rows trained beside a LoRA (train.run token_rows: PEFT's trainable token rows, here on an untied output head): the spec
and its resolution ("turn_end" -> the id of the chat template's turn end), and on tiny models built on the CPU: the rows train
with the pairs and nothing else does; they save, reload and merge; the adapter's off switch reads the original rows (the keeping
terms' base side); a scale interpolates them with the pairs (the keep read, a merge at a strength); the slot of Anima's eyes
stays free of them."""
import copy
import json
import os

import pytest

from qwen_symbolic_thinking import anima_eyes, checks, family, keep, train
from qwen_symbolic_thinking.tools import keep_read as kr

ROW = 5                        # the head row the tiny models train (a token the rows below end on)


# ------------------------------------------------------------------------------------------------- no torch needed
class Tok:
    """A tokenizer over a fixed vocabulary: text is split into its tokens, longest first."""

    def __init__(self, vocab: dict, eos: str):
        self.vocab, self.eos_token_id = vocab, vocab[eos]
        self.back = {v: k for k, v in vocab.items()}

    def __call__(self, text, add_special_tokens=False):
        ids, rest = [], text
        while rest:
            for tok in sorted(self.vocab, key=len, reverse=True):
                if rest.startswith(tok):
                    ids.append(self.vocab[tok])
                    rest = rest[len(tok):]
                    break
            else:
                raise ValueError(rest)
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(self.back[i] for i in ids)

    def __len__(self):
        return max(self.vocab.values()) + 1


class Proc:
    """Renders as Qwen's template does (<|im_start|>role\\ntext<|im_end|>\\n a turn) over the given vocabulary."""

    def __init__(self, vocab):
        self.tokenizer = Tok(vocab, "<|endoftext|>")

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n" + "".join(c["text"] for c in m["content"]) + "<|im_end|>\n"
                      for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


QWEN = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "\n": 3, "q": 4, "ANSWERMARKER": 5, "user": 6, "assistant": 7,
        "x": 30}


class Model:
    def __init__(self, eos):
        self.generation_config = type("G", (), {"eos_token_id": eos})()
        self.config = self.generation_config


def test_the_spec_is_checked_before_anything_loads():
    assert train.check_token_rows_spec(None) is None and train.check_token_rows_spec({}) is None
    assert train.check_token_rows_spec({"lm_head": ["turn_end", 7]}) == {"lm_head": ["turn_end", 7]}
    for bad in ({"lm_head": []}, {"lm_head": "turn_end"}, {"lm_head": [-1]}, {"lm_head": ["eos"]}, {"lm_head": [True]},
                {"": [1]}, ["lm_head"], {"lm_head": 5}, {"lm_head": [1.5]}):
        with pytest.raises(ValueError):
            train.check_token_rows_spec(bad)


def test_turn_end_resolves_to_the_template_s_own_token():
    p = Proc(QWEN)
    assert train.resolve_token_rows({"lm_head": ["turn_end"]}, p, Model(0)) == {"lm_head": [2]}
    assert train.resolve_token_rows({"lm_head": ["turn_end", 30, 2]}, p) == {"lm_head": [2, 30]}   # in order, once each
    assert train.resolve_token_rows(None, p) is None
    with pytest.raises(ValueError, match="past the tokenizer"):
        train.resolve_token_rows({"lm_head": [31]}, p)


def test_a_turn_end_of_several_tokens_or_one_generation_does_not_stop_on_is_refused():
    pieces = {k: v for k, v in QWEN.items() if k != "<|im_end|>"} | {"<|im_": 20, "end|>": 21}
    with pytest.raises(ValueError, match="2 tokens"):
        train.resolve_token_rows({"lm_head": ["turn_end"]}, Proc(pieces))
    merged = dict(QWEN) | {"<|im_end|>\n": 22}           # the turn end and its newline one token: generation stops on 22
    with pytest.raises(ValueError, match="not among the ids generation stops on"):
        train.resolve_token_rows({"lm_head": ["turn_end"]}, Proc(merged), Model(0))


def test_run_refuses_token_rows_on_a_full_finetune_and_a_bad_spec():
    with pytest.raises(ValueError, match="full fine-tune"):
        train.run(rank=0, token_rows={"lm_head": ["turn_end"]})
    with pytest.raises(ValueError, match="neither a token id"):
        train.run(token_rows={"lm_head": ["end"]})


# ------------------------------------------------------------------------------------- tiny models on the CPU
def tiny_llama(vocab=32, seed=0):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=False)
    return transformers.LlamaForCausalLM(cfg).eval()


def with_rows(model, rows=None, seed=None):
    pytest.importorskip("peft")
    return train.build_lora(model, 4, 8, 0.0, ["q_proj", "v_proj"], rows or {"lm_head": [ROW]}, seed=seed)


def batch(torch):
    ids = torch.tensor([[9, 10, 11, 12, ROW], [13, 14, 15, ROW, ROW]])
    labels = ids.clone()
    labels[:, :2] = -100
    return ids, labels


def trained(steps=6):
    """A tiny untied model with a LoRA and its head row ROW, trained a few plain Adam steps on rows ending in ROW."""
    torch = pytest.importorskip("torch")
    model = with_rows(tiny_llama())
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-2, betas=(0.9, 0.999))
    ids, labels = batch(torch)
    for _ in range(steps):
        model(input_ids=ids, labels=labels).loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model.eval()


def logits(model, torch):
    with torch.no_grad():
        return model(input_ids=batch(torch)[0]).logits.float()


def head_row_layer(model):
    return [m for m in model.modules() if keep.is_token_layer(m)]


def test_peft_trains_token_rows_here():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    out = checks.peft_token_rows()
    assert out["peft_token_rows_ok"], out


def test_the_row_trains_with_the_pairs_and_nothing_else():
    torch = pytest.importorskip("torch")
    start = tiny_llama()
    before = {n: p.detach().clone() for n, p in start.named_parameters()}
    model = with_rows(tiny_llama())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    rows = [n for n in names if "trainable_tokens" in n]
    assert rows == ["base_model.model.lm_head.token_adapter.trainable_tokens_delta.default"]
    assert all("lora_" in n for n in names if n not in rows) and len(names) == 1 + 2 * 2 * 2     # 2 layers x q, v x A, B
    assert train.token_rows_report(model) == {"lm_head": {"ids": [ROW], "numbers": 16, "dtype": "float32"}}
    assert train.lora_report(model) == {"q_proj": 2, "v_proj": 2}
    assert train.check_token_rows(model, {"lm_head": [ROW]})["lm_head"]["ids"] == [ROW]
    end_before = logits(model, torch)[:, -2, ROW]
    t = trained()
    layer = head_row_layer(t)[0]
    assert not torch.equal(layer.trainable_tokens_delta["default"].float(), layer.trainable_tokens_original["default"].float())
    assert bool((logits(t, torch)[:, -2, ROW] > end_before).all())          # the row learned to end the rows
    for n, p in t.named_parameters():                                          # every base weight as it started
        if "lora_" in n or "trainable_tokens" in n:
            continue
        key = n.replace("base_model.model.", "").replace(".base_layer.", ".").replace("lm_head.token_adapter.", "lm_head.")
        assert torch.equal(p, before[key]), n
    assert any(float(p.detach().abs().sum()) > 0 for n, p in t.named_parameters() if "lora_B" in n)   # the pairs moved too


def test_check_token_rows_refuses_rows_nobody_asked_for_or_a_module_not_wrapped():
    pytest.importorskip("torch")
    model = with_rows(tiny_llama())
    with pytest.raises(ValueError, match="asked"):
        train.check_token_rows(model, {"lm_head": [ROW + 1]})
    with pytest.raises(ValueError, match="0 layers"):
        train.check_token_rows(model, {"embed_tokens": [ROW]})
    with pytest.raises(ValueError, match="nobody asked"):
        train.check_token_rows(model, {})


def test_the_rows_save_reload_and_merge(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    t = trained()
    want = logits(t, torch)
    t.save_pretrained(str(tmp_path))
    local = str(tmp_path / "local base")
    cfg_path = tmp_path / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["trainable_token_indices"] == {"lm_head": [ROW]}
    cfg["base_model_name_or_path"] = local
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert train.name_base(str(tmp_path), local, "org/model") == ["adapter_config.json"]     # as a step ship names its base
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["base_model_name_or_path"] == "org/model" and cfg["trainable_token_indices"] == {"lm_head": [ROW]}
    from safetensors import safe_open
    with safe_open(str(tmp_path / "adapter_model.safetensors"), "pt") as f:
        assert [k for k in f.keys() if "trainable_tokens" in k] == ["base_model.model.lm_head.token_adapter.trainable_tokens_delta"]
    again = peft.PeftModel.from_pretrained(tiny_llama(), str(tmp_path)).eval()
    assert torch.equal(logits(again, torch), want)                          # the reads load it this way, unmerged
    head0 = tiny_llama().lm_head.weight.detach()
    merged = peft.PeftModel.from_pretrained(tiny_llama(), str(tmp_path)).merge_and_unload().eval()
    changed = (merged.lm_head.weight.detach() - head0).abs().sum(dim=1).nonzero().flatten().tolist()
    assert changed == [ROW] and torch.allclose(logits(merged, torch), want, atol=1e-5)
    assert torch.equal(merged.lm_head.weight[ROW].detach(), head_row_layer(t)[0].trainable_tokens_delta["default"].detach()[0])


def test_the_off_switch_reads_the_original_rows():
    torch = pytest.importorskip("torch")
    t = trained()
    with t.disable_adapter():
        off = logits(t, torch)
    assert torch.equal(off, logits(tiny_llama(), torch)) and not torch.equal(off, logits(t, torch))


def test_the_keeping_term_reads_the_moved_row_against_the_original(tmp_path):
    """general_kl's base side is the adapter's off switch: with the LoRA pairs at zero, the head's row alone moved, the term
    reads the row's change; with the row at its original it reads 0."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = with_rows(tiny_llama())
    ids, labels = batch(torch)
    task = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    stats = {}
    cls = keep.keep_trainer(general_kl=1.0, stats=stats)
    args = transformers.TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=2, report_to=[], use_cpu=True)
    kt = cls(model=model, args=args)
    with torch.no_grad():
        kt.compute_loss(model, {**task, "anchor": dict(task)})
    assert stats["general_kl"] == pytest.approx(0.0, abs=1e-7)
    layer = head_row_layer(model)[0]
    with torch.no_grad():
        layer.trainable_tokens_delta["default"].add_(0.5)
    stats.clear()
    with torch.no_grad():
        kt.compute_loss(model, {**task, "anchor": dict(task)})
    assert stats["general_kl"] > 1e-4


def test_a_scale_interpolates_the_rows_with_the_pairs(tmp_path):
    """tools.keep_read's scales: at s the row is original + s (trained - original) while every LoRA layer's scaling is s times
    its trained value; at 1 the trained model comes back bit for bit, at 0 the off switch's. keep.scale_adapter, before a
    merge (screen.load_model at an adapter_scale), gives the same model merged."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    t = trained()
    t.save_pretrained(str(tmp_path))
    m = peft.PeftModel.from_pretrained(tiny_llama(), str(tmp_path)).eval()
    full = logits(m, torch)
    layers = kr.lora_layers(m)
    rows = [x for x, _ in layers if keep.is_token_layer(x)]
    assert len(layers) == 4 + 1 and len(rows) == 1
    orig = rows[0].trainable_tokens_original["default"].float()
    tuned = rows[0].trainable_tokens_delta["default"].detach().float().clone()
    kr.set_scale(layers, 0.5)
    assert torch.allclose(rows[0].trainable_tokens_delta["default"].float(), orig + 0.5 * (tuned - orig), atol=1e-7)
    assert all(x.scaling["default"] == pytest.approx(0.5 * 8 / 4) for x, _ in layers if not keep.is_token_layer(x))
    half = logits(m, torch)
    assert not torch.equal(half, full)
    kr.set_scale(layers, 0.0)
    zero = logits(m, torch)
    with m.disable_adapter():
        assert torch.equal(zero, logits(m, torch))
    kr.set_scale(layers, 1.0)
    assert torch.equal(logits(m, torch), full) and torch.equal(rows[0].trainable_tokens_delta["default"].float(), tuned)
    scaled = peft.PeftModel.from_pretrained(tiny_llama(), str(tmp_path)).eval()
    assert keep.scale_adapter(scaled, 0.5) == 4 + 1
    merged = scaled.merge_and_unload().eval()
    head0 = tiny_llama().lm_head.weight.detach()
    assert (merged.lm_head.weight.detach() - head0).abs().sum(dim=1).nonzero().flatten().tolist() == [ROW]
    assert torch.allclose(merged.lm_head.weight[ROW].detach(), (orig + 0.5 * (tuned - orig))[0], atol=1e-7)
    assert torch.allclose(logits(merged, torch), half, atol=1e-5)


def tiny_internvl(first=2):
    """An InternVL of 4 language layers (width 32) with an untied head and the anima_eyes record for a slot of the layers from
    `first` up (as test_anima_eyes builds it)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import InternVLConfig, InternVLForConditionalGeneration, Qwen3Config
    text = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    cfg = InternVLConfig(text_config=text.to_dict(), vision_config={"hidden_size": 32, "intermediate_size": 64,
                                                                    "num_hidden_layers": 1, "num_attention_heads": 4,
                                                                    "image_size": [28, 28], "patch_size": [14, 14]},
                         image_token_id=63, tie_word_embeddings=False)
    cfg.anima_eyes = {"name": "tiny", "first_base_layer": first, "head": "base", "slot_fingerprint": None}
    torch.manual_seed(0)
    return InternVLForConditionalGeneration(cfg)


def test_the_slot_of_anima_s_eyes_stays_free_of_the_rows():
    """The hybrid's LoRA with its head's row: the row is the head's, not the slot's; the slot carries no module, trains nothing
    and keeps its fingerprint through a training step; the head's other rows stay the base's."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    model = tiny_internvl()
    fp = anima_eyes.check_slot(model, 2)
    head0 = model.lm_head.weight.detach().clone()
    lora = train.build_lora(model, 2, 4, 0.0, family.lora_targets(model), {"lm_head": [7]})
    row_names = [n for n, p in lora.named_parameters() if p.requires_grad and "trainable_tokens" in n]
    assert row_names and not any(anima_eyes.is_slot(n, 2) for n in row_names)
    assert not [n for n, p in lora.named_parameters() if p.requires_grad and anima_eyes.is_slot(n, 2)]
    assert sum(train.lora_report(lora).values()) == 2 * 7 and train.token_rows_report(lora)["lm_head"]["ids"] == [7]
    assert anima_eyes.check_slot(lora, 2) == fp
    opt = torch.optim.Adam([p for p in lora.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.tensor([[5, 6, 7, 8, 7]])
    lora.train()
    lora(input_ids=ids, labels=ids).loss.backward()
    assert float(next(p for n, p in lora.named_parameters() if "trainable_tokens" in n).grad.abs().sum()) > 0
    opt.step()
    assert anima_eyes.check_slot(lora, 2) == fp
    merged = lora.merge_and_unload()
    assert (merged.lm_head.weight.detach() - head0).abs().sum(dim=1).nonzero().flatten().tolist() == [7]
    assert anima_eyes.check_slot(merged, 2) == fp


def test_an_adapter_continued_from_carries_its_rows_and_they_must_be_asked(tmp_path):
    """init_adapter: an adapter saved with its rows loads them trainable again (PEFT reads them from its config); train.run
    compares them with the rows a run asks (_rows_of_config)."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    trained().save_pretrained(str(tmp_path))
    again = peft.PeftModel.from_pretrained(tiny_llama(), str(tmp_path), is_trainable=True)
    assert [n for n, p in again.named_parameters() if p.requires_grad and "trainable_tokens" in n]
    assert train._rows_of_config(again.peft_config["default"]) == {"lm_head": [ROW]}
    plain = peft.get_peft_model(tiny_llama(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
    assert train._rows_of_config(plain.peft_config["default"]) == {}
    assert copy.deepcopy(train.token_rows_report(again))["lm_head"]["ids"] == [ROW] and os.path.isdir(str(tmp_path))
    assert torch.is_tensor(head_row_layer(again)[0].trainable_tokens_delta["default"])
