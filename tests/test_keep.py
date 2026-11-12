"""Keeping the base model's behavior (keep.py): the loss terms, the writer check, the paired batches, the adapter scale, the
general rows' conversation, and the trainer's loss on a tiny model (where torch, transformers and peft are installed)."""
import inspect

import pytest

from qwen_symbolic_thinking import keep, train


# ------------------------------------------------------------------------------------------------- no torch needed
def test_check_writers_none_checks_nothing():
    assert keep.check_writers([{"id": "a"}], None) == {}


def test_check_writers_counts_the_accepted_and_refuses_the_rest():
    items = [{"id": "a", "writer": "gpt-6-luna"}, {"id": "b", "writer": "deepseek-flash"}, {"id": "c", "writer": "gpt-6-luna"}]
    assert keep.check_writers(items, ["gpt-6-luna", "deepseek-flash"]) == {"gpt-6-luna": 2, "deepseek-flash": 1}
    with pytest.raises(ValueError, match=r"\(none\)"):
        keep.check_writers(items + [{"id": "d"}], ["gpt-6-luna", "deepseek-flash"])
    with pytest.raises(ValueError, match="other-model"):
        keep.check_writers([{"id": "e", "writer": "other-model"}], ["gpt-6-luna"])


def test_paired_collator_cycles_the_general_rows_in_a_seeded_order():
    def collate(pairs):
        return {"ids": [it["id"] for it, _ in pairs], "forms": [form for _, form in pairs]}
    anchors = [{"id": f"anc-{i}", "kind": "anchor"} for i in range(5)]
    pc = keep.PairedCollator(collate, anchors, 2, seed=7)
    out = [pc([({"id": "t"}, "full")]) for _ in range(5)]
    drawn = [a for o in out for a in o["anchor"]["ids"]]
    names = sorted(a["id"] for a in anchors)
    assert sorted(drawn[:5]) == names and sorted(drawn[5:]) == names      # ten draws = two whole passes over five
    assert all(f == "anchor" for o in out for f in o["anchor"]["forms"]) and all(o["ids"] == ["t"] for o in out)
    again = keep.PairedCollator(collate, anchors, 2, seed=7)
    assert [a for o in (again([({"id": "t"}, "full")]) for _ in range(5)) for a in o["anchor"]["ids"]] == drawn
    with pytest.raises(ValueError):
        keep.PairedCollator(collate, [], 2, seed=7)


def test_paired_collator_builds_the_general_rows_with_its_own_collator():
    b = keep.PairedCollator(lambda p: {"who": "task"}, [{"id": "a"}], 1, seed=1,
                            anchor_collate=lambda p: {"who": "general"})([({"id": "t"}, "full")])
    assert b["who"] == "task" and b["anchor"]["who"] == "general"


def test_scale_adapter_multiplies_every_scaling_and_refuses_a_model_without_one():
    class Layer:
        def __init__(self):
            self.scaling = {"default": 2.0}

    class Model:
        def __init__(self):
            self.layers = [Layer(), Layer()]

        def modules(self):
            return [self] + self.layers

    m = Model()
    assert keep.scale_adapter(m, 0.5) == 2 and all(layer.scaling["default"] == 1.0 for layer in m.layers)

    class Bare:
        def modules(self):
            return [self]

    with pytest.raises(ValueError):
        keep.scale_adapter(Bare(), 0.5)


def test_a_general_row_is_its_request_and_the_base_answer():
    it = {"kind": "anchor", "id": "anc-1", "prompt": "What is on the table?", "answer": "A teapot and two cups."}
    messages, answer = train.conversation(it, "anchor")
    assert answer == "A teapot and two cups."
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "What is on the table?"}]}]
    messages, _ = train.conversation(it, "anchor", image="IMG", mark="<image>")
    assert messages[0]["content"] == "<image>\nWhat is on the table?"
    messages, _ = train.conversation(it, "anchor", image="IMG")
    assert messages[0]["content"][0] == {"type": "image", "image": "IMG"}


def test_run_takes_the_keeping_options_off_by_default():
    params = inspect.signature(train.run).parameters
    for name, default in (("general_kl", 0.0), ("target_kl", 0.0), ("rehearsal", 0.0), ("anchors", None), ("anchor_batch", 4),
                          ("writers", None), ("seed_adapter", False)):
        assert params[name].default == default, name


def test_run_refuses_the_terms_on_a_full_finetune_and_the_general_terms_without_rows():
    with pytest.raises(ValueError, match="LoRA only"):
        train.run(rank=0, target_kl=1.0)
    with pytest.raises(ValueError, match="pass anchors"):
        train.run(general_kl=1.0)
    with pytest.raises(ValueError, match="pass anchors"):
        train.run(rehearsal=0.5)


# ----------------------------------------------------------------------------------------------------- the terms
def test_answer_positions_shift_by_one():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[-100, -100, 5, 6, -100], [-100, 7, 8, 9, 3]])
    mask, targets = keep.answer_positions(labels)
    assert mask.tolist() == [[False, True, True, False], [True, True, True, True]]
    assert targets.tolist() == [5, 6, 7, 8, 9, 3]


def test_kl_full_matches_its_definition_and_is_zero_on_equal_logits():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    zb, zt = torch.randn(6, 11, generator=g), torch.randn(6, 11, generator=g)
    pb = zb.softmax(-1)
    want = (pb * (pb.log() - zt.log_softmax(-1))).sum(-1).mean()
    assert torch.allclose(keep.kl_full(zb, zt, chunk=4), want, atol=1e-6)
    assert float(keep.kl_full(zb, zb.clone())) == pytest.approx(0.0, abs=1e-7)


def test_target_masked_kl_does_not_see_the_target_and_matches_its_definition():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(1)
    zb = torch.randn(5, 9, generator=g)
    y = torch.tensor([0, 3, 8, 2, 2])
    zt = zb.clone()
    zt[torch.arange(5), y] += 4.0                    # only the target's logit moves: the renormalized rest is unchanged
    kt, kept = keep.kl_target_masked(zb, zt, y)
    assert kept == 5 and float(kt) == pytest.approx(0.0, abs=1e-6)
    zt2 = torch.randn(5, 9, generator=g)
    kt2, _ = keep.kl_target_masked(zb, zt2, y, chunk=2)

    def masked(z, t):
        z = z.clone()
        z[torch.arange(len(t)), t] = float("-inf")
        return z.log_softmax(-1)

    lb, lt = masked(zb, y), masked(zt2, y)
    pb = lb.exp()
    want = torch.where(pb > 0, pb * (lb - lt), torch.zeros_like(pb)).sum(-1).mean()
    assert torch.allclose(kt2, want, atol=1e-5)


def test_target_masked_kl_leaves_out_sure_positions_and_sends_no_gradient_to_the_target():
    torch = pytest.importorskip("torch")
    zb = torch.zeros(3, 6)
    zb[0, 2] = 30.0                                  # the base gives token 2 all but certainty at row 0
    y = torch.tensor([2, 1, 4])
    zt = torch.randn(3, 6, generator=torch.Generator().manual_seed(3)).requires_grad_(True)
    kt, kept = keep.kl_target_masked(zb, zt, y)
    assert kept == 2
    kt.backward()
    assert float(zt.grad[1, 1]) == 0.0 and float(zt.grad[2, 4]) == 0.0
    assert float(zt.grad[0].abs().sum()) == 0.0      # the left-out row sends nothing
    assert float(zt.grad[1].abs().sum()) > 0.0


def test_ce_mean_matches_torch():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(2)
    z, y = torch.randn(7, 13, generator=g), torch.randint(0, 13, (7,), generator=g)
    assert torch.allclose(keep.ce_mean(z, y, chunk=3), torch.nn.functional.cross_entropy(z, y), atol=1e-6)


# ------------------------------------------------------------------------------------- the trainer on a tiny model
def _tiny(torch, transformers, peft):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=4, max_position_embeddings=64)
    base = transformers.LlamaForCausalLM(cfg)
    model = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                                                      task_type=peft.TaskType.CAUSAL_LM))
    ids = torch.randint(3, 64, (2, 12), generator=torch.Generator().manual_seed(5))
    labels = ids.clone()
    labels[:, :5] = -100
    task = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    a_ids = torch.randint(3, 64, (2, 10), generator=torch.Generator().manual_seed(6))
    a_labels = a_ids.clone()
    a_labels[:, :4] = -100
    anchor = {"input_ids": a_ids, "attention_mask": torch.ones_like(a_ids), "labels": a_labels}
    return model, task, anchor


def _trainer(transformers, cls, model, tmp_path):
    args = transformers.TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=2, report_to=[], use_cpu=True)
    return cls(model=model, args=args)


def test_keep_trainer_adds_nothing_at_the_start_and_reads_the_moved_adapter(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    model, task, anchor = _tiny(torch, transformers, peft)
    model.eval()
    stock = _trainer(transformers, transformers.Trainer, model, tmp_path)
    with torch.no_grad():
        plain = float(stock.compute_loss(model, dict(task)))
    stats = {}
    cls = keep.keep_trainer(general_kl=1.0, target_kl=1.0, rehearsal=0.0, stats=stats)
    kt = _trainer(transformers, cls, model, tmp_path)
    with torch.no_grad():
        start = float(kt.compute_loss(model, {**task, "anchor": anchor}))
    assert start == pytest.approx(plain, abs=1e-5)                       # B = 0 at the start: tuned = base, both KL terms 0
    assert stats["n"] == 1 and stats["general_kl"] == pytest.approx(0.0, abs=1e-6) and stats["target_kl"] == pytest.approx(0.0, abs=1e-6)
    assert dict(cls.keep_empty) == {}                                    # both terms had answer positions to act on
    with torch.no_grad():                                                # general rows with no answer position: counted
        kt.compute_loss(model, {**task, "anchor": {**anchor, "labels": torch.full_like(anchor["labels"], -100)}})
    assert dict(cls.keep_empty) == {"general_kl": 1}
    stats.clear()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.5)
    stats.clear()
    with torch.no_grad():
        moved_plain = float(stock.compute_loss(model, dict(task)))
        moved = float(kt.compute_loss(model, {**task, "anchor": anchor}))
    assert stats["general_kl"] > 0 and stats["target_kl"] > 0
    # no token count passed: the stock loss is a micro-batch mean, the Trainer divides the whole loss by the accumulation count
    assert moved == pytest.approx(moved_plain + stats["general_kl"] + stats["target_kl"], rel=1e-4)
    if kt.model_accepts_loss_kwargs:
        # a token count passed: the stock loss comes back scaled for accumulation, so the added terms are divided here
        n_items = int((task["labels"][:, 1:] != -100).sum())
        kt.current_gradient_accumulation_steps = stock.current_gradient_accumulation_steps = 2
        stats.clear()
        with torch.no_grad():
            scaled_plain = float(stock.compute_loss(model, dict(task), num_items_in_batch=n_items))
            scaled = float(kt.compute_loss(model, {**task, "anchor": anchor}, num_items_in_batch=n_items))
        assert scaled == pytest.approx(scaled_plain + (stats["general_kl"] + stats["target_kl"]) / 2, rel=1e-4)
    cls2 = keep.keep_trainer(general_kl=0.0, target_kl=0.0, rehearsal=0.5, stats=stats)
    stats.clear()
    with torch.no_grad():
        rehearsed = float(_trainer(transformers, cls2, model, tmp_path).compute_loss(model, {**task, "anchor": anchor}))
    assert rehearsed == pytest.approx(moved_plain + 0.5 * stats["rehearsal_ce"], rel=1e-4)
