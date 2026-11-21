"""The progress file (tools.progress): the training writes runs/progress/<run>.json at its first log step, then every N minutes,
and at its end, on a thread of its own, best effort; the reader prints one line a run. The uploader is stubbed (no network); the
callback is driven by a tiny model on the CPU."""
import inspect
import json
import threading
import time

import pytest

from qwen_symbolic_thinking import train
from qwen_symbolic_thinking.tools import progress


def _rec(step, spent, left=None, loss=1.0, gkl=0.01):
    return {"step": step, "spent_s": spent, "left_s": left if left is not None else 10 * (100 - step), "s_per_step": 8.1,
            "loss": loss, "general_kl": gkl}


def _writer(sent, now, tmp_path=None, **kw):
    return progress.ProgressWriter("V12NC_x", 100, "owner/proj", every_min=10, clock=lambda: now[0],
                                   upload=lambda rec, repo, path: sent.append((rec, repo, path)),
                                   local_dir=str(tmp_path) if tmp_path else None, **kw)


def test_it_writes_at_the_first_log_step_then_every_n_minutes_and_at_the_end(tmp_path):
    now, sent = [1_000_000.0], []
    w = _writer(sent, now, tmp_path, peak=lambda: 41.7, saved=lambda: 20, shipped=lambda: None)
    log = [_rec(10, 81)]
    assert w.maybe(log)
    w.thread.join(2)
    rec, repo, path = sent[0]
    assert (repo, path) == ("owner/proj", "runs/progress/V12NC_x.json")
    assert rec["run"] == "V12NC_x" and rec["state"] == "training" and (rec["step"], rec["of"]) == (10, 100)
    assert rec["loss"] == 1.0 and rec["general_kl"] == 0.01 and rec["s_per_step"] == 8.1 and rec["steady_s_per_step"] is None
    assert rec["peak_gb"] == 41.7 and rec["saved_step"] == 20 and rec["shipped_step"] is None and rec["left_s"] == 900
    assert rec["time"].startswith("1970-01-12") and rec["eta"] and rec["package"] and "_spent_s" not in rec
    log.append(_rec(20, 162))
    now[0] += 60
    assert not w.maybe(log) and len(sent) == 1                    # not due yet: ten minutes after the last write
    now[0] += 600
    log.append(_rec(90, 729))
    assert w.maybe(log)
    w.thread.join(2)
    assert sent[1][0]["steady_s_per_step"] == round((729 - 81) / (90 - 10), 3) and sent[1][0]["step"] == 90
    assert w.finish(log) and sent[2][0]["state"] == "trained" and w.writes == 3 and w.failures == 0
    local = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert local["state"] == "trained" and local["step"] == 90


def test_a_write_that_fails_is_printed_and_skipped(capsys):
    now = [0.0]

    def broken(rec, repo, path):
        raise ConnectionError("the Hub is away")
    w = progress.ProgressWriter("R", 10, "owner/proj", every_min=10, clock=lambda: now[0], upload=broken,
                                peak=lambda: 1 / 0)          # a reading that fails leaves its field empty
    assert w.maybe([_rec(1, 5)])
    w.thread.join(2)
    assert w.failures == 1 and w.writes == 0
    assert "progress file of R not written (ConnectionError: the Hub is away); skipped" in capsys.readouterr().out
    assert w.record([_rec(1, 5)])["peak_gb"] is None


def test_a_slow_write_never_holds_the_training():
    now, gate = [0.0], threading.Event()
    w = progress.ProgressWriter("R", 10, "owner/proj", every_min=10, clock=lambda: now[0],
                                upload=lambda rec, repo, path: gate.wait(10))
    t0 = time.perf_counter()
    assert w.maybe([_rec(1, 5)])
    assert time.perf_counter() - t0 < 0.5                        # the step goes on while the upload runs
    now[0] += 3600
    assert not w.maybe([_rec(2, 10)]) and w.skipped == 1         # the previous upload still runs: this write is skipped
    t0 = time.perf_counter()
    assert not w.finish([_rec(3, 15)], wait_s=0.2)               # the end waits a moment at most, then leaves it be
    assert time.perf_counter() - t0 < 1.0
    gate.set()
    w.thread.join(2)
    assert w.writes == 1


def test_the_reader_prints_one_line_a_run(monkeypatch, capsys):
    recs = {"runs/progress/A.json": {"run": "A", "state": "training", "step": 810, "of": 21815, "loss": 1.032, "general_kl": 0.0301,
                                     "s_per_step": 8.04, "steady_s_per_step": 8.01, "left_s": 169000, "eta": "2026-11-23T11:00:00+00:00",
                                     "peak_gb": 41.7, "saved_step": None, "shipped_step": None, "time": "2026-11-21T12:13:00+00:00"},
            "runs/progress/B.json": None}
    listing = ["runs/progress/A.json", "runs/progress/B.json", "runs/done/A.json", "x.json"]
    out = progress.read("owner/proj", list_files=lambda repo: listing, fetch=lambda repo, path: recs.get(path))
    assert [r["run"] for r in out] == ["A", "B"] and out[1] == {"run": "B", "state": None}
    line = progress.line(out[0], now=1_000_000_000_000.0)
    assert line.startswith("A: training, step 810 of 21815 (3.7%); loss 1.0320; general_kl 0.03010; 8.04 s a step (steady 8.01)")
    assert "peak 41.7 GB" in line and "saved at step -, shipped step -" in line and "about 46.9 h left" in line
    assert progress.line(out[1]) == "B: no progress file"
    assert progress.read("owner/proj", ["A"], fetch=lambda repo, path: recs.get(path))[0]["step"] == 810
    monkeypatch.setattr(progress, "_list_files", lambda repo: listing)
    monkeypatch.setattr(progress, "_fetch", lambda repo, path: recs.get(path))
    progress.main(["owner/proj"])
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("A: training, step 810") and printed[1] == "B: no progress file"


def test_train_writes_no_progress_file_unless_asked():
    params = inspect.signature(train.run).parameters
    assert params["progress_every_min"].default is None
    source = inspect.getsource(train.run)
    assert "if progress_every_min:" in source and '"progress_every_min"' not in source.split("settings = ")[1].split(")")[0]


def test_the_callback_follows_the_step_log_of_a_real_training(tmp_path):
    """train.progress's callback appends each log step's record; the progress file's callback, after it, writes at the first
    log step, then when due, and once at the end."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                   num_key_value_heads=2, max_position_embeddings=16)
    model = transformers.LlamaForCausalLM(cfg)
    rows = [torch.randint(3, 32, (6,)) for _ in range(12)]

    def collate(batch):
        ids = torch.stack(batch)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    args = transformers.TrainingArguments(output_dir=str(tmp_path / "run"), per_device_train_batch_size=2, num_train_epochs=1,
                                          learning_rate=1e-3, logging_steps=2, save_strategy="no", report_to=[],
                                          use_cpu=True, disable_tqdm=True, remove_unused_columns=False)
    log, sent, now = [], [], [0.0]
    writer = progress.ProgressWriter("tiny", 6, "owner/proj", every_min=10, clock=lambda: now[0],
                                     upload=lambda rec, repo, path: sent.append(rec))
    trainer = transformers.Trainer(model=model, args=args, train_dataset=rows, data_collator=collate,
                                   callbacks=[train.progress(6, "tiny", log, {}), progress.progress_callback(writer, log)])
    trainer.train()
    if writer.thread is not None:
        writer.thread.join(2)
    assert [r["step"] for r in log] == [2, 4, 6]
    assert [(r["state"], r["step"]) for r in sent] == [("training", 2), ("trained", 6)]   # the first log step, then the end
