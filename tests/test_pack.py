"""pack.py on a stand-in storage bucket: the picture lists, the seed lookup, the shards and the manifest (read back through
extract.fetch_images, the way a training run reads them), every picture that is not packed counted by reason; tools.build_rows
with a pack; the queue's waits_for; the project repo from QST_REPO. No network."""
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen_symbolic_thinking import pack, queue


def jpeg(color=(200, 30, 30), size=(8, 6)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeBucket:
    """The three calls pack.py makes, over a dict {remote path: bytes}."""

    def __init__(self, files: dict):
        self.files, self.downloads = files, []

    def list_bucket_tree(self, bucket_id, prefix="", recursive=True):
        return [SimpleNamespace(path=p, type="file") for p in sorted(self.files) if p.startswith(prefix)]

    def download_bucket_files(self, bucket_id, pairs, raise_on_missing_files=True):
        for remote, local in pairs:
            self.downloads.append(remote)
            if remote in self.files:
                with open(local, "wb") as fh:
                    fh.write(self.files[remote])
            elif raise_on_missing_files:
                raise FileNotFoundError(remote)


def lists_file(lines) -> bytes:
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode("utf-8")


@pytest.fixture
def bucket():
    return FakeBucket({
        "prod/control/pictures/a.jsonl": lists_file([
            {"image": "ffhq/101.jpg", "urls": ["store:pictures/ffhq/512/101.jpg"]},
            {"image": "ffhq/102.jpg", "urls": ["store:pictures/ffhq/512/102.jpg"]},              # listed, not in the bucket
            {"image": "deepfashion/MEN-Pants-id_1.jpg", "urls": ["store:pictures/deepfashion/512/MEN-Pants-id_1.jpg"]},
            {"image": "deepfashion/MEN-Bad-id_2.jpg", "urls": ["store:pictures/deepfashion/512/MEN-Bad-id_2.jpg"]},
            {"image": "bench/384/ffhq/103.jpg", "urls": ["store:pictures/ffhq/384/103.jpg"]},   # another purpose: never used
            {"image": "ffhq/104.jpg", "urls": ["ftp://nowhere/104.jpg"]}]),
        "prod/control/pictures/b.jsonl": lists_file([{"image": "ffhq/101.jpg", "urls": ["store:pictures/ffhq/1024/101.jpg"]}]),
        "prod/pictures/ffhq/1024/101.jpg": jpeg((10, 200, 10), (16, 12)),
        "prod/pictures/deepfashion/512/MEN-Pants-id_1.jpg": jpeg(),
        "prod/pictures/deepfashion/512/MEN-Bad-id_2.jpg": b"not a picture",
        "prod/pictures/ffhq/384/103.jpg": jpeg(),
    })


DIRS = {"ffhq": ["ffhq"], "deepfashion_mm": ["deepfashion"]}


def test_bucket_names_and_links():
    assert pack.parse_bucket("owner/name") == ("owner/name", "")
    assert pack.parse_bucket("owner/name/prod/x") == ("owner/name", "prod/x")
    with pytest.raises(ValueError):
        pack.parse_bucket("name")
    assert pack.first_link(["ftp://x", "store:a/b.jpg", "https://y"]) == "store:a/b.jpg"
    assert pack.first_link(["ftp://x"]) is None


def test_lists_are_keyed_by_folder_and_stem_and_a_later_line_wins(bucket, tmp_path):
    table = pack.read_lists(bucket, "owner/store", "prod", workdir=str(tmp_path / "lists"))
    assert table[("ffhq", "101")] == ["store:pictures/ffhq/1024/101.jpg"]          # b.jsonl, read after a.jsonl
    assert ("bench", "103") in table and ("ffhq", "103") not in table
    assert pack.find("ffhq:101", table, DIRS) == ["store:pictures/ffhq/1024/101.jpg"]
    assert pack.find("deepfashion_mm:MEN-Pants-id_1", table, DIRS)[0].endswith("MEN-Pants-id_1.jpg")
    assert pack.find("ffhq:103", table, DIRS) is None and pack.find("coco:1", table, DIRS) is None


def test_pack_writes_shards_and_a_manifest_that_fetch_images_reads(bucket, tmp_path, monkeypatch):
    seeds = ["ffhq:101", "ffhq:102", "ffhq:103", "ffhq:104", "deepfashion_mm:MEN-Pants-id_1", "deepfashion_mm:MEN-Bad-id_2",
             "ffhq:101"]
    out = pack.pack_staged(seeds, "owner/store/prod", dirs=DIRS, out_dir=str(tmp_path / "pack"), tag="t", repo="owner/twin",
                           shard=1, do_ship=False, api=bucket)
    assert (out["asked"], out["packed"], out["shards"]) == (6, 2, 2)
    assert out["not_packed"] == {"missing from the bucket": 1, "in no picture list": 1, "no readable link": 1, "not a picture": 1}
    assert out["by_source"] == {"deepfashion_mm": 1, "ffhq": 1} and out["path_in_repo"] == "pictures/t"
    rows = [json.loads(l) for l in open(out["manifest"], encoding="utf-8")]
    assert [(r["source"], r["source_id"]) for r in rows] == [("deepfashion_mm", "MEN-Pants-id_1"), ("ffhq", "101")]
    loc = rows[1]["image"]
    assert loc == {"kind": "hf_parquet", "repo": "owner/twin", "repo_type": "model", "path": "pictures/t/shard_0001.parquet",
                   "row_index": 0, "column": "image"}
    assert rows[1]["width"] == 16 and rows[1]["height"] == 12
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(tmp_path / "pack" / "packed" / "shard_0001.parquet"))
    assert tbl.column_names == ["id", "source", "source_id", "image", "width", "height"] and tbl.column("id").to_pylist() == ["ffhq:101"]
    import huggingface_hub
    from qwen_symbolic_thinking import extract
    shards = tmp_path / "pack" / "packed"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, **kw: str(shards / os.path.basename(path)))
    got = extract.fetch_images([{"sample_id": r["sample_id"], "image": r["image"]} for r in rows], str(tmp_path / "cache"))
    assert sorted(got) == sorted(r["sample_id"] for r in rows) and got[rows[1]["sample_id"]].size == (16, 12)


def test_rows_that_share_a_packed_picture_each_get_it(bucket, tmp_path, monkeypatch):
    out = pack.pack_staged(["ffhq:101", "deepfashion_mm:MEN-Pants-id_1"], "owner/store/prod", dirs=DIRS,
                           out_dir=str(tmp_path / "pack"), tag="t", repo="owner/twin", do_ship=False, api=bucket)
    rows = [json.loads(l) for l in open(out["manifest"], encoding="utf-8")]
    import huggingface_hub
    from qwen_symbolic_thinking import extract
    shards = tmp_path / "pack" / "packed"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, **kw: str(shards / os.path.basename(path)))
    face = rows[1]["image"]                  # two rows written on one picture, another picture, and a row past the shard's end
    items = [{"sample_id": "a", "image": face}, {"sample_id": "b", "image": dict(face)}, {"sample_id": "c", "image": rows[0]["image"]},
             {"sample_id": "d", "image": dict(face, row_index=99)}]
    failures = {}
    got = extract.fetch_images(items, str(tmp_path / "cache"), failures=failures)
    assert sorted(got) == ["a", "b", "c"] and got["a"].size == got["b"].size == (16, 12) and got["a"] is not got["b"]
    assert failures == {"d": "row index outside the shard"}


def api_row(rid, seed_id, license_="CC BY-NC 4.0"):
    return {"id": rid, "series": "s", "depth": "medium", "rank": 1, "domain": "photograph", "genre": "portrait",
            "spec": {"thing": {"name": "person"}}, "thoughts": [{"step": "see", "text": "A person."}],
            "wordings": {"caption": "A person."}, "questions": [], "held": False, "row_id": f"row-{rid}",
            "source": {"kind": "seeded", "seed_id": seed_id, "license": license_, "model": "m", "seed_dataset": "d"},
            "writer": {"model": "model-a", "image_size": [512, 512]},
            "source_label": {"dataset": "d", "item_id": seed_id, "license": license_}}


def write_rows(tmp_path):
    folder = tmp_path / "rows" / "accepted" / "route-a"
    folder.mkdir(parents=True)
    rows_ = [api_row("1", "ffhq:101"), api_row("2", "deepfashion_mm:MEN-Pants-id_1"), api_row("3", "ffhq:102"),
             api_row("4", "coco:100", "CC BY 4.0")]
    (folder / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    (tmp_path / "screen.json").write_text(json.dumps({"items": []}), encoding="utf-8")


def test_build_rows_packs_first_and_the_packed_pictures_join_the_build(tmp_path, bucket, monkeypatch):
    from functools import partial
    from qwen_symbolic_thinking.tools import build_rows

    write_rows(tmp_path)
    monkeypatch.setattr(build_rows, "pack_staged", partial(build_rows.pack_staged, api=bucket))
    summary = build_rows.run(str(tmp_path / "rows"), tag="v_nc", repo="owner/twin", do_ship=False, out_dir=str(tmp_path / "out"),
                             screen=str(tmp_path / "screen.json"), licenses=None, hold_in_view=[],
                             pack={"bucket": "owner/store/prod", "dirs": DIRS})
    built = json.loads((tmp_path / "out" / "v_nc.json").read_text(encoding="utf-8"))
    ids = {it["id"]: it for it in built["items"]}
    assert set(ids) == {"row-1", "row-2", "row-4"} and summary["counts"]["dropped"] == {"no_locator": 1}
    assert ids["row-1"]["image"]["kind"] == "hf_parquet" and ids["row-1"]["image"]["repo"] == "owner/twin"
    assert ids["row-4"]["image_urls"][0].endswith("000000000100.jpg")              # COCO keeps its own links
    assert summary["pack"]["packed"] == 2 and summary["pack"]["asked"] == 3
    assert built["filters"]["manifest"] == ["owner/twin:pictures/v_nc/manifest.jsonl"]
    assert built["filters"]["pack"]["not_packed"] == {"missing from the bucket": 1}


def test_the_staged_copy_wins_over_an_older_manifests_picture_of_the_same_number(tmp_path, bucket, monkeypatch):
    from functools import partial
    from qwen_symbolic_thinking.tools import build_rows

    write_rows(tmp_path)
    older = tmp_path / "older.jsonl"         # another copy of the dataset, with a different picture under number 101
    older.write_text(json.dumps({"sample_id": "x-1", "split": "main", "source": "ffhq", "source_id": "101",
                                 "image": {"kind": "hf_parquet", "repo": "owner/other", "repo_type": "dataset",
                                           "path": "data/ffhq.parquet", "row_index": 7, "column": "image"}}) + "\n",
                     encoding="utf-8")
    monkeypatch.setattr(build_rows, "pack_staged", partial(build_rows.pack_staged, api=bucket))
    summary = build_rows.run(str(tmp_path / "rows"), tag="v_nc", repo="owner/twin", do_ship=False, out_dir=str(tmp_path / "out"),
                             screen=str(tmp_path / "screen.json"), licenses=None, hold_in_view=[], manifest=str(older),
                             pack={"bucket": "owner/store/prod", "dirs": DIRS})
    built = json.loads((tmp_path / "out" / "v_nc.json").read_text(encoding="utf-8"))
    ids = {it["id"]: it for it in built["items"]}
    assert summary["pack"]["asked"] == 3 and summary["pack"]["packed"] == 2      # 101 is packed although the older copy has it
    assert ids["row-1"]["image"]["repo"] == "owner/twin" and ids["row-1"]["image"]["path"].startswith("pictures/v_nc/")
    assert built["filters"]["manifest"] == [str(older), "owner/twin:pictures/v_nc/manifest.jsonl"]


def test_a_spec_waits_for_its_inputs_without_burning_its_name(monkeypatch):
    spec = {"name": "N0", "kind": "train", "waits_for": ["sft/v_nc.json", "/anchors/a/anchors.json"]}
    assert queue.missing_inputs(spec, {"sft/v_nc.json"}) == ["/anchors/a/anchors.json"]
    assert queue.missing_inputs(spec, {"sft/v_nc.json", "anchors/a/anchors.json"}) == []
    assert queue.missing_inputs({"name": "x"}, set()) == []
    polls = {"n": 0}
    files = [set(), {"sft/v_nc.json", "anchors/a/anchors.json"}]

    def repo_files(repo):
        polls["n"] += 1
        return files[min(polls["n"] - 1, 1)] | ({"runs/done/N0.json"} if polls["n"] > 2 else set())

    ran = []
    monkeypatch.setattr(queue, "_fresh", lambda path, repo: [spec])
    monkeypatch.setattr(queue, "_repo_files", repo_files)
    monkeypatch.setattr(queue, "_run_marked", lambda s, lane, repo, out: ran.append(s["name"]) or {"status": "done"})
    monkeypatch.setattr(queue.time, "sleep", lambda s: None)
    out = queue.run_queue("lane", repo="owner/twin", idle_minutes=0, poll_s=0)
    assert out == ["N0"] and ran == ["N0"] and polls["n"] >= 3


def test_a_spec_that_waits_too_long_ends_the_lane(monkeypatch):
    spec = {"name": "N0", "kind": "train", "waits_for": ["sft/never.json"]}
    clock = {"t": 0.0}
    monkeypatch.setattr(queue, "_fresh", lambda path, repo: [spec])
    monkeypatch.setattr(queue, "_repo_files", lambda repo: set())
    monkeypatch.setattr(queue.time, "time", lambda: clock["t"])
    monkeypatch.setattr(queue.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 3600))
    assert queue.run_queue("lane", repo="owner/twin", poll_s=1, wait_hours=2) == []


def test_the_project_repo_follows_qst_repo():
    code = "import qwen_symbolic_thinking as q, qwen_symbolic_thinking.queue as qu; print(q.REPO); print(qu.run_queue.__defaults__[0])"
    env = dict(os.environ, QST_REPO="owner/twin")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], env={**env, "PYTHONPATH": root + os.pathsep + env.get("PYTHONPATH", "")},
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["owner/twin", "owner/twin"]
    env.pop("QST_REPO")
    out = subprocess.run([sys.executable, "-c", code], env={**env, "PYTHONPATH": root + os.pathsep + env.get("PYTHONPATH", "")},
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "AbstractPhil/qwen-symbolic-thinking-image-training"
