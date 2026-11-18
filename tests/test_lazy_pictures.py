"""Pictures held on disk (extract.fetch_images(lazy=True), train.run(lazy_pictures=True)): the same pictures load and fail, for
the same reasons, as when they are decoded into memory, and the same pairs collate to the same tensors; rows that share a picture
share its file. Every locator kind is read offline (local files, a local parquet shard, local archives; downloads monkeypatched).
Also the run's switches of this release: gradient checkpointing, and the rates and memory in the summary. No network, no GPU."""
import inspect
import io
import json
import os
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from qwen_symbolic_thinking import chain, extract, queue, resume, train

MAX_SIDE = 40          # smaller than the test pictures, so every picture is bounded on its way in


def picture(seed: int, size=(64, 48), fmt="JPEG") -> bytes:
    """A noisy picture (so a JPEG is larger than the 1,000 bytes a url answer must reach)."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


GARBAGE = b"this is not a picture " * 100


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """Rows of every locator kind over local stand-ins, and the monkeypatched downloads that serve them."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    src = tmp_path / "src"
    src.mkdir()
    # urls
    served = {"http://pics.invalid/u1.jpg": picture(1), "http://pics.invalid/u2.png": picture(2, (30, 70), "PNG"),
              "http://pics.invalid/bad.jpg": GARBAGE}

    def get(url, timeout=None, headers=None, **kw):
        if url in served:
            return SimpleNamespace(status_code=200, content=served[url])
        return SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(requests, "get", get)
    # a parquet shard (two rows a group) and a dataset split keyed by a column
    cells = [{"bytes": picture(10 + i), "path": None} for i in range(5)]
    cells[4] = {"bytes": GARBAGE, "path": None}
    pq.write_table(pa.table({"image": cells}), str(src / "shard.parquet"), row_group_size=2)
    pq.write_table(pa.table({"key": ["k0", "k1", "k2"], "image": [{"bytes": picture(20 + i), "path": None} for i in range(3)]}),
                   str(src / "train-0.parquet"))
    # a file, a zip on the Hub, and archives at a url (a tar and a zip)
    (src / "one.jpg").write_bytes(picture(30, (50, 90)))
    with zipfile.ZipFile(src / "hub.zip", "w") as zf:
        zf.writestr("m/a.jpg", picture(31))
        zf.writestr("m/b.jpg", picture(32))
    members = {"x/1.jpg": picture(40), "x/2.jpg": picture(41, (90, 40)), "x/bad.jpg": GARBAGE}
    for name, b in members.items():
        (src / "tar_in" / os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
        (src / "tar_in" / name).write_bytes(b)
    with tarfile.open(src / "pics.tar", "w") as tf:
        for name in members:
            tf.add(str(src / "tar_in" / name), arcname=name)
    with zipfile.ZipFile(src / "pics.zip", "w") as zf:
        zf.writestr("z/1.jpg", picture(50))
        zf.writestr("z/bad.jpg", GARBAGE)
    hub_files = {"shard.parquet": src / "shard.parquet", "data/train-0.parquet": src / "train-0.parquet",
                 "one.jpg": src / "one.jpg", "hub.zip": src / "hub.zip"}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, **kw: str(hub_files[path]))

    class Api:
        def __init__(self, token=None):
            pass

        def dataset_info(self, repo):
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename="data/train-0.parquet")])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    shard = {"kind": "hf_parquet", "repo": "o/r", "repo_type": "model", "path": "shard.parquet", "column": "image"}
    split = {"kind": "hf_dataset_row", "repo": "o/d", "split": "train", "key_column": "key", "image_column": "image"}
    tar = {"kind": "url_archive", "archive_url": "http://archive.invalid/pics.tar"}
    zipped = {"kind": "url_archive", "archive_url": "http://archive.invalid/pics.zip"}
    hubzip = {"kind": "zip_member", "repo": "o/z", "path": "hub.zip"}
    rows = [
        {"sample_id": "url-a", "image": {"kind": "url", "urls": ["http://pics.invalid/u1.jpg"]}},
        {"sample_id": "url-b", "image": {"kind": "url", "urls": ["http://pics.invalid/u1.jpg"]}},       # shares url-a's picture
        {"sample_id": "url-c", "image": {"kind": "url", "urls": ["http://pics.invalid/missing.jpg", "http://pics.invalid/u2.png"]}},
        {"sample_id": "url-404", "image": {"kind": "url", "urls": ["http://pics.invalid/missing.jpg"]}},
        {"sample_id": "url-bad", "image": {"kind": "url", "urls": ["http://pics.invalid/bad.jpg"]}},
        {"id": "item-urls", "image_urls": ["http://pics.invalid/u2.png"]},                             # a training item's form
        {"sample_id": "pq-a", "image": dict(shard, row_index=1)},
        {"sample_id": "pq-b", "image": dict(shard, row_index=1)},                                      # shares pq-a's picture
        {"sample_id": "pq-c", "image": dict(shard, row_index=3)},
        {"sample_id": "pq-bad", "image": dict(shard, row_index=4)},
        {"sample_id": "pq-out", "image": dict(shard, row_index=99)},
        {"sample_id": "ds-a", "image": dict(split, key="k1")},
        {"sample_id": "ds-b", "image": dict(split, key="k1")},
        {"sample_id": "ds-none", "image": dict(split, key="k9")},
        {"sample_id": "file-a", "image": {"kind": "hf_file", "repo": "o/f", "path": "one.jpg"}},
        {"sample_id": "zip-a", "image": dict(hubzip, member="m/a.jpg")},
        {"sample_id": "zip-b", "image": dict(hubzip, member="m/b.jpg")},
        {"sample_id": "zip-none", "image": dict(hubzip, member="m/none.jpg")},
        {"sample_id": "tar-1", "image": dict(tar, member="x/1.jpg")},
        {"sample_id": "tar-2", "image": dict(tar, member="x/2.jpg")},
        {"sample_id": "tar-2b", "image": dict(tar, member="x/2.jpg")},
        {"sample_id": "tar-bad", "image": dict(tar, member="x/bad.jpg")},
        {"sample_id": "tar-none", "image": dict(tar, member="x/none.jpg")},
        {"sample_id": "azip-1", "image": dict(zipped, member="z/1.jpg")},
        {"sample_id": "azip-bad", "image": dict(zipped, member="z/bad.jpg")},
        {"sample_id": "pending"},
    ]
    return SimpleNamespace(rows=rows, src=src)


def fetch_both(sources, tmp_path):
    """The rows fetched eagerly and lazily, each into its own cache folder (an archive at a url is found there, downloaded)."""
    out = {}
    for mode in ("eager", "lazy"):
        cache = tmp_path / f"cache_{mode}"
        cache.mkdir()
        for name in ("pics.tar", "pics.zip"):
            (cache / name).write_bytes((sources.src / name).read_bytes())
        failures = {}
        got = extract.fetch_images(sources.rows, str(cache), workers=4, failures=failures, max_side=MAX_SIDE, lazy=mode == "lazy")
        out[mode] = (got, failures, cache)
    return out


def test_lazy_pictures_load_and_fail_as_the_eager_ones_for_every_kind(sources, tmp_path):
    import numpy as np
    both = fetch_both(sources, tmp_path)
    (eager, e_fail, _), (lazy, l_fail, cache) = both["eager"], both["lazy"]
    assert isinstance(eager, dict) and isinstance(lazy, extract.LazyPictures)
    assert sorted(eager) == sorted(lazy) == sorted(["url-a", "url-b", "url-c", "item-urls", "pq-a", "pq-b", "pq-c", "ds-a", "ds-b",
                                                     "file-a", "zip-a", "zip-b", "tar-1", "tar-2", "tar-2b", "azip-1"])
    assert e_fail == l_fail                                           # the same pictures fail, for the same reasons
    assert l_fail["url-404"] == "http 404" and l_fail["url-bad"].startswith("undecodable")
    assert l_fail["pq-out"] == "row index outside the shard" and l_fail["pq-bad"].startswith("undecodable")
    assert l_fail["ds-none"] == "key not in the split's files" and l_fail["zip-none"].startswith("zip member")
    assert l_fail["tar-bad"].startswith("archive member") and l_fail["tar-none"].startswith("archive member")
    assert l_fail["azip-bad"].startswith("archive member") and l_fail["pending"] == "no locator"
    for k in eager:
        a, b = eager[k], lazy[k]
        assert a.mode == b.mode == "RGB" and a.size == b.size and max(a.size) <= MAX_SIDE, k
        assert np.array_equal(np.asarray(a), np.asarray(b)), k        # the same pixels
    # the lazy mapping holds files, never pictures; rows that share a picture share its file
    assert all(isinstance(p, str) and os.path.isfile(p) for p in lazy.paths.values())
    for a, b in (("url-a", "url-b"), ("pq-a", "pq-b"), ("ds-a", "ds-b"), ("tar-2", "tar-2b")):
        assert lazy.paths[a] == lazy.paths[b], (a, b)
    assert len(set(lazy.paths.values())) == len(lazy.paths) - 4
    assert lazy.paths["tar-1"].startswith(str(cache)) and "pics.tar_x" in lazy.paths["tar-1"]   # read where the tar was unpacked
    assert all(p.startswith(str(cache)) for p in lazy.paths.values())
    assert lazy["url-a"] is not lazy["url-a"]                         # a fresh picture at every read


def test_the_lazy_mapping_reads_files_only_when_a_picture_is_read(tmp_path, monkeypatch):
    p1, p2 = tmp_path / "1.img", tmp_path / "2.img"
    p1.write_bytes(picture(1))
    p2.write_bytes(picture(2))
    reads = []
    real = extract._image_from_file
    monkeypatch.setattr(extract, "_image_from_file", lambda path, max_side=extract.MAX_SIDE: reads.append(path) or real(path, max_side))
    lazy = extract.LazyPictures(MAX_SIDE, threads=2)
    lazy.update({"a": str(p1), "b": str(p2)})
    assert "a" in lazy and "z" not in lazy and len(lazy) == 2 and sorted(lazy) == ["a", "b"] and reads == []
    other = extract.LazyPictures(MAX_SIDE)
    other["c"] = str(p1)
    lazy.update(other)
    assert lazy.paths["c"] == str(p1) and reads == []
    got = lazy.many(["a", "b", "a", "missing", "c"])
    assert sorted(got) == ["a", "b", "c"] and len(reads) == 3          # each picture decoded once; a missing one is left out
    assert lazy.get("missing") is None and got["a"].size == (MAX_SIDE, 30)
    with pytest.raises(TypeError):
        lazy["d"] = got["a"]                                           # a lazy mapping holds files, not pictures
    import pickle
    assert pickle.loads(pickle.dumps(lazy)).paths == lazy.paths       # its thread pool does not travel


# ------------------------------------------------------------------------------------------ the batches are the same
class Tokens:
    pad_token_id = 0
    eos_token = "<e>"


class Processor:
    """The part of a vision-language processor the Collator reads: a chat template, a tokenizer, and pictures turned into
    tensors exactly (each picture's RGB bytes in order, its size, and a run of placeholder tokens whose length follows the size)."""
    chat_template = "{{ enable_thinking }}"

    def __init__(self):
        self.tokenizer = Tokens()

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        out = ""
        for m in messages:
            c = m["content"]
            body = c if isinstance(c, str) else "".join("<img>" if p["type"] == "image" else p["text"] for p in c)
            out += f"<{m['role']}>{body}<end>\n"
        return out + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, images=None, return_tensors="pt", padding=False, **kw):
        import numpy as np
        import torch
        images, rows, k = list(images or []), [], 0
        assert sum(t.count("<img>") for t in text) == len(images)
        for t in text:
            ids = []
            for i, part in enumerate(t.split("<img>")):
                if i:
                    w, h = images[k].size
                    ids += [5] * (3 + (w * h) % 11)
                    k += 1
                ids += [ord(c) % 997 + 10 for c in part]
            rows.append(ids)
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for i, r in enumerate(rows):
            input_ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        out = {"input_ids": input_ids, "attention_mask": mask}
        if images:
            out["pixel_values"] = torch.cat([torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).reshape(-1).float() for im in images])
            out["image_sizes"] = torch.tensor([list(im.size) for im in images])
        return out


ROW = ("see: a red barn stands in a field\nsee: a white fence runs along the front\nmood: a calm morning\n"
       "caption: a red barn behind a white fence in a quiet field\nprompt: red barn, white fence, morning light\n"
       "question: barn color -> red [chain]\nquestion: time of day -> morning [chain]")


def test_eager_and_lazy_pictures_collate_to_the_same_batches(sources, tmp_path):
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_available(), "the tests run on the CPU (CUDA_VISIBLE_DEVICES=-1)"
    both = fetch_both(sources, tmp_path)
    eager, lazy = both["eager"][0], both["lazy"][0]
    parsed = chain.parse(ROW)
    items = [{"id": k, "kind": "image", "depth": "medium", "spec": {}, "thoughts": parsed["thoughts"], "wordings": parsed["wordings"],
              "questions": parsed["questions"], "forms": ["short", "full"]} for k in sorted(eager)]
    anchors = [{"id": k, "kind": "anchor", "prompt": f"What is shown in {k}?", "answer": f"A picture called {k}."}
               for k in ("tar-1", "pq-c", "no-picture")]
    pairs = train.examples_of(items) + [(a, "anchor") for a in anchors]
    import random
    random.Random(7).shuffle(pairs)
    assert any(p[0]["id"] == "no-picture" for p in pairs)
    proc = Processor()
    a, b = train.Collator(proc, eager, max_len=400), train.Collator(proc, lazy, max_len=400)   # some examples run past it
    for i in range(0, len(pairs), 4):
        x, y = a(pairs[i:i + 4]), b(pairs[i:i + 4])
        assert sorted(x) == sorted(y) and "pixel_values" in x
        for k in x:
            assert torch.equal(x[k], y[k]), (i, k)
        assert (x["labels"] != -100).any()
    assert a.counts.trace == b.counts.trace and a.collated == len(pairs) and 0 < a.cut < a.collated   # the cut rows read the same


# ------------------------------------------------------------------------------------------ the cache's writes
def full_disk_on_first_write(monkeypatch):
    """os.replace of a .part file fails once with a full disk (errno 28); every other call is the real one."""
    real, hits = os.replace, []

    def replace(src, dst, *a, **kw):
        if str(src).endswith(".part") and not hits:
            hits.append(dst)
            raise OSError(28, "No space left on device")
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(extract.os, "replace", replace)
    return hits


@pytest.mark.parametrize("kind,lazy", [("hf_parquet", True), ("url", True), ("url", False)])
def test_a_picture_file_that_cannot_be_written_stops_the_fetch(sources, tmp_path, monkeypatch, kind, lazy):
    rows = [r for r in sources.rows if (r.get("image") or {}).get("kind") == kind or (kind == "url" and "image_urls" in r)]
    hits = full_disk_on_first_write(monkeypatch)
    cache = tmp_path / "cache"
    failures = {}
    with pytest.raises(extract.PictureStoreError, match="No space left on device") as err:
        extract.fetch_images(rows, str(cache), workers=2, failures=failures, max_side=MAX_SIDE, lazy=lazy)
    assert hits and str(hits[0]) in str(err.value) and err.value.__cause__.errno == 28   # names the file and the OS error
    assert not [p for p in os.listdir(cache) if p.endswith(".part")]                   # no partial file left behind


def test_a_picture_that_does_not_decode_is_still_dropped_with_its_reason(sources, tmp_path):
    rows = [r for r in sources.rows if r.get("sample_id") in ("pq-bad", "url-bad", "pq-c")]
    failures = {}
    got = extract.fetch_images(rows, str(tmp_path / "c"), workers=2, failures=failures, max_side=MAX_SIDE, lazy=True)
    assert sorted(got) == ["pq-c"] and sorted(failures) == ["pq-bad", "url-bad"]
    assert all(v.startswith("undecodable") for v in failures.values())


@pytest.mark.parametrize("lazy", [False, True])
def test_a_cut_short_cached_download_is_fetched_again_and_a_whole_one_is_not(tmp_path, monkeypatch, lazy):
    import numpy as np
    import requests
    whole = picture(7)
    asked = []

    def get(url, timeout=None, headers=None, **kw):
        asked.append(url)
        return SimpleNamespace(status_code=200, content=whole) if url.endswith("ok.jpg") else SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(requests, "get", get)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = [{"sample_id": "a", "image": {"kind": "url", "urls": ["http://pics.invalid/ok.jpg"]}},
            {"sample_id": "b", "image": {"kind": "url", "urls": ["http://pics.invalid/gone.jpg"]}}]
    path_of = {r["sample_id"]: (extract._stored_name(str(cache), ["url", r["image"]["urls"]]) if lazy
                                else str(cache / (r["sample_id"] + ".img"))) for r in rows}
    for p in path_of.values():                     # a write cut short by a killed process, on the next start's disk
        with open(p, "wb") as fh:
            fh.write(whole[: len(whole) // 3])
    failures = {}
    got = extract.fetch_images(rows, str(cache), workers=2, failures=failures, max_side=MAX_SIDE, lazy=lazy)
    assert sorted(got) == ["a"] and failures == {"b": "http 404"}            # fetched again once; a second failure is ordinary
    assert sorted(asked) == ["http://pics.invalid/gone.jpg", "http://pics.invalid/ok.jpg"]
    assert open(path_of["a"], "rb").read() == whole and not os.path.exists(path_of["b"])
    expected = extract._image_from_bytes(whole, MAX_SIDE)
    assert np.array_equal(np.asarray(got["a"]), np.asarray(expected))
    asked.clear()
    again = extract.fetch_images(rows[:1], str(cache), workers=2, max_side=MAX_SIDE, lazy=lazy)
    assert sorted(again) == ["a"] and asked == []                            # a whole cached file is not fetched again
    assert not [p for p in os.listdir(cache) if p.endswith(".part")]


# ------------------------------------------------------------------------------------------ the run's switches
def test_run_takes_the_switches_of_this_release_with_the_earlier_behavior_by_default():
    params = inspect.signature(train.run).parameters
    assert params["lazy_pictures"].default is False and params["picture_workers"].default == 16
    assert params["grad_checkpointing"].default is True
    assert inspect.signature(extract.fetch_images).parameters["lazy"].default is False


def test_a_saved_state_made_without_the_switches_still_loads(tmp_path, monkeypatch):
    """The switches are not among the settings a resumed run must match: a state saved by an earlier package (whose settings
    never named them) is taken back by a run that sets them, whatever their values."""
    store = resume.FolderStore(str(tmp_path / "store"))
    monkeypatch.setattr(train.saved_state, "HubStore", lambda repo: store)
    monkeypatch.setattr(train, "fetch_folder", lambda path, repo: f"/fetched/{path}")
    params = inspect.signature(train.run).parameters
    keys = ("model_id", "inputs", "rank", "alpha", "dropout", "lr", "epochs", "batch", "grad_accum", "max_len", "warmup",
            "n_items", "seed", "init_adapter", "system", "trust_remote_code", "revision", "targets", "image_settings", "keys",
            "max_steps", "anchors", "anchor_batch", "general_kl", "target_kl", "rehearsal", "writers", "seed_adapter")
    saved = resume.normalized({k: params[k].default for k in keys})
    store.finish("R", {"finished": True, "stamp": "s", "path_in_repo": "lora/m/r32/s", "settings": saved,
                       "summary": {"steps": 7, "shipped": True}})
    for lazy, workers, gc in ((True, 64, False), (False, 16, True), (True, 8, True)):
        out = train.run(tag="R", resume=True, out_dir=str(tmp_path / "out"), lazy_pictures=lazy, picture_workers=workers,
                        grad_checkpointing=gc)
        assert out["steps"] == 7 and out["adapter_dir"] == "/fetched/lora/m/r32/s/final"


def test_the_fingerprints_read_ids_and_forms_never_pictures():
    class Item(dict):
        def __getitem__(self, k):
            if k not in ("id",):
                raise AssertionError(f"the fingerprint read {k!r}")
            return dict.__getitem__(self, k)

    pairs = [(Item(id="a", image=object()), "short"), (Item(id="b"), "full")]
    assert resume.fingerprint(pairs) == resume.fingerprint([({"id": "a"}, "short"), ({"id": "b"}, "full")])


def test_gradient_checkpointing_off_reaches_the_model_and_the_trainer():
    on = train.trainer_arguments("d", 8, 2, 1.0, 2e-4, 100, 0.03, 1, None, True)
    off = train.trainer_arguments("d", 8, 2, 1.0, 2e-4, 100, 0.03, 1, 30, False)
    assert on["gradient_checkpointing"] is True and on["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert off["gradient_checkpointing"] is False and "gradient_checkpointing_kwargs" not in off and off["max_steps"] == 30
    assert {k: v for k, v in on.items() if not k.startswith("gradient_checkpointing")} == \
           {k: v for k, v in off.items() if not k.startswith("gradient_checkpointing") and k != "max_steps"}
    assert on["warmup_steps"] == 3 and "max_steps" not in on

    class Model:
        def __init__(self, kind):
            self.config, self.calls = SimpleNamespace(model_type=kind), []

        def gradient_checkpointing_enable(self, **kw):
            self.calls.append(("checkpointing", kw))

        def enable_input_require_grads(self):
            self.calls.append(("input_grads",))

    m = Model("qwen3_5")
    assert train.prepare_checkpointing(m, True) is True
    assert m.calls == [("checkpointing", {"gradient_checkpointing_kwargs": {"use_reentrant": False}}), ("input_grads",)]
    m = Model("qwen3_5")
    assert train.prepare_checkpointing(m, False) is True and m.calls == []
    m = Model("molmo2")
    assert train.prepare_checkpointing(m, True) is False and [c[0] for c in m.calls] == ["checkpointing"]


def test_the_summary_reads_memory_and_the_training_phase_rate(monkeypatch):
    torch = pytest.importorskip("torch")
    assert not torch.cuda.is_available(), "the tests run on the CPU (CUDA_VISIBLE_DEVICES=-1)"
    assert train.peak_gpu_gb(torch) is None
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, max_memory_allocated=lambda: 61_234_000_000))
    assert train.peak_gpu_gb(fake) == 61.23
    pytest.importorskip("transformers")
    clock = [1000.0]
    monkeypatch.setattr(train.time, "time", lambda: clock[0])
    cb = train.progress(100, "x", [], {})
    assert cb.phase_s_per_step(10) is None                             # before the training began
    cb.on_train_begin(None, SimpleNamespace(global_step=40), None)    # a resumed machine starts at step 40
    clock[0] += 120.0
    assert cb.phase_s_per_step(70) == 4.0                              # 120 s over the 30 steps this machine trained


def test_a_train_spec_without_a_screen_leaves_its_marker_only(monkeypatch, tmp_path):
    from qwen_symbolic_thinking import screen
    asked = {}
    monkeypatch.setattr(queue, "_mark", lambda name, payload, repo, out_dir: None)
    monkeypatch.setattr(train, "run", lambda **kw: asked.update(kw) or {"steps": 30, "log": [1], "env": {}, "adapter_dir": "d"})
    monkeypatch.setattr(screen, "run", lambda **kw: (_ for _ in ()).throw(AssertionError("no screen was asked for")))
    spec = {"name": "S_smoke", "kind": "train", "screen": False, "train": {"max_steps": 30, "do_ship": False}}
    out = queue.run_spec(spec, "lane", "o/r", str(tmp_path))
    assert out["status"] == "done" and out["screen"] is None and out["train"] == {"steps": 30, "adapter_dir": "d"}
    assert asked == {"max_steps": 30, "do_ship": False, "tag": "S_smoke", "repo": "o/r"}
    screened = []
    monkeypatch.setattr(screen, "run", lambda **kw: screened.append(kw) or {"ok": 1})
    out = queue.run_spec({**spec, "screen": {"forms": ["short"]}}, "lane", "o/r", str(tmp_path))
    assert out["screen"] == {"ok": 1} and screened[0]["forms"] == ("short",)
    json.dumps(out)
