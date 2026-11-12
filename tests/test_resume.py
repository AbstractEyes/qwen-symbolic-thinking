"""Resuming a long training on a fresh machine (resume.py): the counts, the schedule, the stores, the lane's pickup, and the
resumed run on a tiny model, which must give the uninterrupted run's loss log, data order, general rows and final adapter."""
import inspect
import json
import os
import time

import pytest

from qwen_symbolic_thinking import keep, queue, resume, train


# ------------------------------------------------------------------------------------------------- no torch needed
def test_consumed_counts_the_micro_batches_trained():
    # 10 examples in batches of 2: 5 micro-batches an epoch, in steps of 2, 2 and 1
    assert [resume.consumed(s, 10, 2, 2) for s in range(7)] == [0, 2, 4, 5, 7, 9, 10]
    assert [resume.consumed(s, 7, 2, 1) for s in range(9)] == list(range(9))
    assert resume.consumed(4, 16, 2, 2) == 8                                  # 8 micro-batches, 4 steps: one epoch exactly


def test_counts_keep_each_batch_and_restore_a_prefix():
    c = resume.Counts()
    for x in ((2, 0, 30), (2, 1, 70), (1, 0, 20)):
        c.add(*x)
    assert (c.collated, c.cut, c.longest) == (5, 1, 70)
    assert c.upto(2) == [[2, 0, 30], [2, 1, 70]]
    with pytest.raises(ValueError):
        c.upto(4)
    c.restore(json.loads(json.dumps(c.upto(2))))
    assert (c.collated, c.cut, c.longest) == (4, 1, 70) and c.trace == [(2, 0, 30), (2, 1, 70)]
    c.clear()
    assert (c.collated, c.cut, c.longest) == (0, 0, 0)


def test_fingerprint_reads_the_order_and_the_forms():
    a, b = {"id": "a"}, {"id": "b"}
    fp = resume.fingerprint([(a, "full"), (b, "short")])
    assert fp == resume.fingerprint([({"id": "a", "other": 1}, "full"), (b, "short")])
    assert fp != resume.fingerprint([(b, "short"), (a, "full")])
    assert fp != resume.fingerprint([(a, "short"), (b, "short")])
    assert fp != resume.fingerprint([(a, "full")])


def test_a_resumed_run_must_have_the_saved_settings_and_examples():
    resume.check_settings({"lr": 2e-4, "inputs": ["x.json"]}, {"lr": 2e-4, "inputs": ("x.json",)})   # tuples read back as lists
    with pytest.raises(ValueError, match="lr"):
        resume.check_settings({"lr": 2e-4, "inputs": ["x.json"]}, {"lr": 1e-4, "inputs": ["x.json"]})
    with pytest.raises(ValueError, match="seed"):
        resume.check_settings({"lr": 2e-4}, {"lr": 2e-4, "seed": 1})
    saved = {"n_examples": 4, "examples": "abc", "general_rows": "g1"}
    resume.check_examples(saved, "abc", "g1", 4)
    with pytest.raises(ValueError, match="training examples"):
        resume.check_examples(saved, "abc", "g1", 3)
    with pytest.raises(ValueError, match="training examples"):
        resume.check_examples(saved, "abd", "g1", 4)
    with pytest.raises(ValueError, match="general rows"):
        resume.check_examples(saved, "abc", None, 4)


def test_the_schedule_saves_on_the_clock_and_once_before_the_age_limit():
    now, age = [0.0], [1.0]
    s = resume.Schedule(every_h=2.0, before_h=23.0, retry_min=15.0, clock=lambda: now[0], age=lambda: age[0])
    s.start()
    assert s.due(10, 10, 100) is None                         # nothing due yet
    now[0] = 2 * 3600.0
    assert s.due(15, 10, 100) is None                         # due by the clock, but not a log step
    assert s.due(100, 10, 100) is None                        # never at the last step
    assert s.due(20, 10, 100) == "schedule"
    s.saved("schedule")
    now[0] += 60.0
    age[0] = 23.2
    assert s.due(30, 10, 100) == "age"
    s.saved("age")
    assert s.due(40, 10, 100) is None                         # the age save happens once
    now[0] += 2 * 3600.0
    assert s.due(50, 10, 100) == "schedule"                   # the clock keeps going past the age limit
    s.failed()                                                # a save that fails: tried again 15 min later, not at once
    assert s.due(60, 10, 100) is None
    now[0] += 15 * 60.0
    assert s.due(70, 10, 100) == "schedule"


def test_the_schedule_has_no_age_save_on_a_machine_already_past_the_limit_or_without_an_age():
    now = [0.0]
    old = resume.Schedule(every_h=2.0, before_h=23.0, clock=lambda: now[0], age=lambda: 40.0)
    old.start()
    blind = resume.Schedule(every_h=2.0, before_h=23.0, clock=lambda: now[0], age=lambda: None)
    blind.start()
    now[0] = 3600.0
    assert old.due(10, 10, 100) is None and blind.due(10, 10, 100) is None
    steps = resume.Schedule(at_steps=[20])
    assert steps.due(20, 10, 100) == "step" and steps.due(10, 10, 100) is None and steps.due(25, 5, 100) is None


def test_the_folder_store_replaces_reads_back_and_finishes(tmp_path):
    ckpt = tmp_path / "checkpoint-30"
    ckpt.mkdir()
    (ckpt / resume.STATE).write_text(json.dumps({"step": 30}), encoding="utf-8")
    (ckpt / "optimizer.pt").write_bytes(b"x")
    (ckpt / "README.md").write_text("card", encoding="utf-8")
    (ckpt / "training_args.bin").write_bytes(b"pickle")
    store = resume.FolderStore(str(tmp_path / "store"))
    assert resume.load(store, "R1", str(tmp_path / "dl")) is None
    store.put(str(ckpt), "R1")
    (ckpt / resume.STATE).write_text(json.dumps({"step": 40}), encoding="utf-8")
    store.put(str(ckpt), "R1")                                # a later save replaces the earlier one
    found = resume.load(store, "R1", str(tmp_path / "dl"))
    assert found["step"] == 40 and sorted(os.listdir(found["dir"])) == ["optimizer.pt", resume.STATE]
    store.finish("R1", {"finished": True, "summary": {"steps": 50}})
    again = resume.load(store, "R1", str(tmp_path / "dl2"))
    assert again["finished"] and again["summary"] == {"steps": 50} and os.listdir(again["dir"]) == [resume.STATE]


def test_the_lane_runs_again_a_resumable_training_whose_machine_ended():
    spec = {"name": "R", "kind": "train", "train": {"resume": True}}

    def marker(status):
        return lambda path, repo: None if status is None else {"status": status}
    assert queue.resumes(spec, "o/r", read=marker("started"))
    assert not queue.resumes(spec, "o/r", read=marker("done")) and not queue.resumes(spec, "o/r", read=marker("failed"))
    assert not queue.resumes(spec, "o/r", read=marker(None))                     # a marker that cannot be read: not this poll
    asked = []
    assert not queue.resumes({"name": "P", "kind": "train", "train": {}}, "o/r", read=lambda p, r: asked.append(p))
    assert not queue.resumes({"name": "T", "kind": "tool", "tool": {}}, "o/r", read=lambda p, r: asked.append(p))
    assert asked == []                                                          # no read without resume on


def test_run_takes_resume_off_by_default_and_refuses_it_without_a_tag_or_on_a_full_finetune():
    params = inspect.signature(train.run).parameters
    assert params["resume"].default is False and params["resume_every_h"].default == 2.0 and params["resume_before_h"].default == 23.0
    with pytest.raises(ValueError, match="pass tag"):
        train.run(resume=True)
    with pytest.raises(ValueError, match="full fine-tune"):
        train.run(resume=True, tag="x", rank=0)


def test_run_returns_a_finished_run_without_training_and_refuses_other_settings(tmp_path, monkeypatch):
    store = resume.FolderStore(str(tmp_path / "store"))
    monkeypatch.setattr(train.saved_state, "HubStore", lambda repo: store)
    monkeypatch.setattr(train, "fetch_folder", lambda path, repo: f"/fetched/{path}")

    def settings_of(**kw):          # the settings train.run records for these arguments
        params = inspect.signature(train.run).parameters
        args = {k: p.default for k, p in params.items()}
        args.update(kw)
        keys = ("model_id", "inputs", "rank", "alpha", "dropout", "lr", "epochs", "batch", "grad_accum", "max_len", "warmup",
                "n_items", "seed", "init_adapter", "system", "trust_remote_code", "revision", "targets", "image_settings", "keys",
                "max_steps", "anchors", "anchor_batch", "general_kl", "target_kl", "rehearsal", "writers", "seed_adapter")
        return resume.normalized({k: args[k] for k in keys})

    store.finish("R9", {"finished": True, "stamp": "20261112_000000_R9", "path_in_repo": "lora/Qwen3.5-9B/r32/20261112_000000_R9",
                        "settings": settings_of(model_id="Qwen/Qwen3.5-9B", epochs=2), "summary": {"steps": 2102, "shipped": True}})
    out = train.run(model_id="Qwen/Qwen3.5-9B", epochs=2, tag="R9", resume=True, out_dir=str(tmp_path / "out"))
    assert out["steps"] == 2102 and out["adapter_dir"] == "/fetched/lora/Qwen3.5-9B/r32/20261112_000000_R9/final"
    with pytest.raises(ValueError, match="epochs"):
        train.run(model_id="Qwen/Qwen3.5-9B", epochs=3, tag="R9", resume=True, out_dir=str(tmp_path / "out"))


# ------------------------------------------------------------------------------------- a resumed run on a tiny model
VOCAB = 64
N_EXAMPLES = 22                 # 11 micro-batches of 2 an epoch, in steps of 2 x 2 (the epoch's last step takes one)
BATCH, ACCUM, EPOCHS = 2, 2, 2
STEPS_PER_EPOCH = 6
TOTAL = STEPS_PER_EPOCH * EPOCHS


def _items():
    import random
    rng = random.Random(3)
    items = [{"id": f"ex-{i}", "kind": "image", "ids": [rng.randrange(3, VOCAB) for _ in range(6 + i % 5)]} for i in range(N_EXAMPLES)]
    anchors = [{"id": f"anc-{i}", "kind": "anchor", "ids": [rng.randrange(3, VOCAB) for _ in range(5 + i % 3)]} for i in range(7)]
    pairs = [(it, "short") for it in items]
    random.Random(20261030).shuffle(pairs)
    return pairs, anchors


class TinyCollator:
    """train.Collator's part that a resume reads (counts batch by batch), on token lists: the first two tokens are the prompt
    (no loss), the rest the answer; .seen lists the ids of every batch it built, in order."""

    def __init__(self, torch, max_len=9):
        self.torch, self.max_len, self.counts, self.seen = torch, max_len, resume.Counts(), []

    def __call__(self, pairs):
        torch = self.torch
        rows = [it["ids"] for it, _ in pairs]
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        labels = ids.clone()
        labels[mask == 0] = -100
        labels[:, :2] = -100
        self.counts.add(len(rows), sum(1 for r in rows if len(r) > self.max_len), width)
        self.seen.append([it["id"] for it, _ in pairs])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _machine(tmp_path, name, store, tag, at_steps=(), stop_at=None, logging_steps=1):
    """One machine's run of the same training, as train.run wires it: a fresh model (the same starting adapter), the paired
    collators, the keeping trainer, the progress log, and with a store the saved state (found and taken back, saved at
    at_steps). stop_at: the machine ends after that step (training stops there; nothing after it is saved)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    pairs, anchors = _items()
    found = resume.load(store, tag, str(tmp_path / name / "state")) if store is not None else None
    if found is not None:
        resume.check_examples(found, resume.fingerprint(pairs), resume.fingerprint([(a, "anchor") for a in anchors]), len(pairs))
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    model = peft.get_peft_model(transformers.LlamaForCausalLM(cfg),
                                peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
                                                task_type=peft.TaskType.CAUSAL_LM))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2, betas=(0.9, 0.999))
    args = transformers.TrainingArguments(output_dir=str(tmp_path / name / "run"), per_device_train_batch_size=BATCH,
                                          gradient_accumulation_steps=ACCUM, num_train_epochs=EPOCHS, learning_rate=1e-2,
                                          lr_scheduler_type="cosine", warmup_steps=2, logging_steps=logging_steps,
                                          save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
                                          seed=20261030, use_cpu=True, disable_tqdm=True)
    collate, anchor_collate = TinyCollator(torch), TinyCollator(torch)
    data = keep.PairedCollator(collate, anchors, 2, 11, anchor_collate)
    log, keep_stats, spent_before = train.carried(found)
    cls = keep.keep_trainer(general_kl=1.0, stats=keep_stats)
    progress_cb = train.progress(TOTAL, "tiny", log, keep_stats, None, spent_before)
    callbacks = [progress_cb]
    rs = None
    if store is not None:
        rs = train.RunState(found, tag=tag, stamp="20261112_000000_" + tag, path_in_repo="lora/tiny/r4/x", settings={},
                            pairs=pairs, examples_print=resume.fingerprint(pairs),
                            general_print=resume.fingerprint([(a, "anchor") for a in anchors]), batch=BATCH, grad_accum=ACCUM,
                            total=TOTAL, collate=collate, anchor_collate=anchor_collate, anchor_batch=2, data_collator=data,
                            log=log, keep_stats=keep_stats, keep_empty=cls.keep_empty, progress=progress_cb, t0=time.time())
        rs.saver = resume.saver(store, tag, resume.Schedule(at_steps=at_steps), rs.record)
        callbacks.append(rs.saver)

    class Stop(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if stop_at is not None and state.global_step >= stop_at:
                control.should_training_stop = True
            return control

    trainer = cls(model=model, args=args, train_dataset=train.Pairs(pairs), data_collator=data, optimizers=(optimizer, None),
                  callbacks=callbacks + [Stop()])
    if rs is not None:
        rs.trainer = trainer
    result = trainer.train(resume_from_checkpoint=found["dir"] if found else None)
    loss = rs.loss_sum() / max(result.global_step, 1) if found is not None else result.training_loss
    adapter = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    return {"log": log, "adapter": adapter, "collate": collate, "anchor_collate": anchor_collate, "found": found,
            "steps": result.global_step, "loss": loss, "rs": rs, "keep_empty": dict(cls.keep_empty)}


def _same_log(a, b):
    assert [r["step"] for r in a] == [r["step"] for r in b]
    for x, y in zip(a, b):
        assert x["loss"] == pytest.approx(y["loss"], rel=1e-6, abs=1e-7), (x, y)
        assert x["learning_rate"] == pytest.approx(y["learning_rate"], rel=1e-9), (x, y)
        assert x.get("general_kl", 0.0) == pytest.approx(y.get("general_kl", 0.0), rel=1e-5, abs=1e-7), (x, y)


@pytest.mark.parametrize("plan", [
    [(3, 5)],                 # saved in the first epoch, the machine ends two steps later
    [(8, 10)],                # saved in the second epoch
    [(6, 7)],                 # saved at the first epoch's end
    [(2, 4), (9, 11)],        # twice: a resumed run that is itself cut off and resumed
], ids=["first-epoch", "second-epoch", "epoch-end", "twice"])
def test_a_resumed_run_follows_the_uninterrupted_run(tmp_path, plan):
    torch = pytest.importorskip("torch")
    whole = _machine(tmp_path, "whole", None, "R")
    assert whole["steps"] == TOTAL and len(whole["log"]) == TOTAL
    store = resume.FolderStore(str(tmp_path / "store"))
    task_seen, anchor_seen, steps_before = [], [], 0
    for k, (save_at, end_at) in enumerate(plan):
        cut = _machine(tmp_path, f"cut{k}", store, "R", at_steps=[save_at], stop_at=end_at)
        assert cut["steps"] == end_at and [s["step"] for s in cut["rs"].saver.saves] == [save_at]
        n = resume.consumed(save_at, N_EXAMPLES, BATCH, ACCUM) - resume.consumed(steps_before, N_EXAMPLES, BATCH, ACCUM)
        task_seen += cut["collate"].seen[:n]        # the batches it trained up to its save (it collated one more ahead)
        anchor_seen += cut["anchor_collate"].seen[:n]
        steps_before = save_at
    last = _machine(tmp_path, "last", store, "R")
    assert last["found"]["step"] == plan[-1][0] and last["steps"] == TOTAL
    _same_log(last["log"], whole["log"])
    assert task_seen + last["collate"].seen == whole["collate"].seen                  # the data order
    assert anchor_seen + last["anchor_collate"].seen == whole["anchor_collate"].seen  # the general rows beside each batch
    assert last["collate"].counts.trace == whole["collate"].counts.trace              # the counts, batch by batch
    assert last["anchor_collate"].counts.trace == whole["anchor_collate"].counts.trace
    assert last["keep_empty"] == whole["keep_empty"]
    assert last["loss"] == pytest.approx(whole["loss"], rel=1e-6)                   # the whole run's mean loss
    assert set(last["adapter"]) == set(whole["adapter"])
    for name, w in whole["adapter"].items():                                          # the final adapter, to fp32 tolerance
        assert torch.allclose(last["adapter"][name], w, rtol=1e-5, atol=1e-6), name
    assert all(float(w.abs().sum()) > 0 for n, w in whole["adapter"].items() if "lora_B" in n)   # trained (B starts at zero)
    record = last["rs"].resumed
    assert [r["step"] for r in record] == [save for save, _ in plan]


def test_a_save_between_log_lines_keeps_the_log_the_same(tmp_path):
    whole = _machine(tmp_path, "whole", None, "L", logging_steps=2)
    store = resume.FolderStore(str(tmp_path / "store"))
    cut = _machine(tmp_path, "cut", store, "L", at_steps=[3, 4], stop_at=5, logging_steps=2)
    assert [s["step"] for s in cut["rs"].saver.saves] == [4]                         # step 3 is not a log step
    last = _machine(tmp_path, "last", store, "L", logging_steps=2)
    _same_log(last["log"], whole["log"])
    assert last["loss"] == pytest.approx(whole["loss"], rel=1e-6)
