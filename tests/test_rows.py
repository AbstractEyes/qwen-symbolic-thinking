"""rows.build_api_rows on a tiny fixture folder: both source kinds, every filter and its count, the held-out drops (the screen
set, the manifest's held split, the museum draw), the picture locators (COCO, DOCCI, the manifest join), the items' keys and a
reproducible, sorted output. No network."""
import json

import pytest

from qwen_symbolic_thinking import rows


def row(rid, seed_id=None, kind="seeded", writer="model-a", license_="CC BY 4.0", row_id=None, **extra):
    src = {"kind": kind, "model": "m", "seed_dataset": "d", "seed_id": seed_id, "license": license_}
    out = {"id": rid, "series": "s", "depth": "medium", "rank": 1, "domain": "photograph", "genre": "street",
           "spec": {"thing": {"name": "dog"}}, "thoughts": [{"step": "see", "text": "A dog."}], "wordings": {"caption": "A dog."},
           "questions": [], "held": False, "source": src, "row_id": row_id or f"row-{rid}",
           "writer": {"provider": "p", "model": writer, "route": "r", "effort": "high", "date": "2026-11-10", "task_id": "t",
                      "rules_version": "1", "package_version": "1"},
           "source_label": {"dataset": "d", "item_id": seed_id, "license": license_}}
    out.update(extra)
    return out


MANIFEST = [
    {"sample_id": "nga-7", "split": "probe", "source": "nga", "source_id": "7", "image": {"kind": "url", "urls": ["http://x/7.jpg"]}},
    {"sample_id": "nga-8", "split": "main", "source": "nga", "source_id": "8", "image": {"kind": "url", "urls": ["http://x/8.jpg"]}},
    {"sample_id": "nga-9", "split": "train", "source": "nga", "source_id": "9", "image": {"kind": "url", "urls": ["http://x/9.jpg"]}},
    {"sample_id": "nga-10", "split": "train", "source": "nga", "source_id": "10", "image": {"kind": "pending"}},
]
SCREEN = {"items": [{"id": "s1", "seed_id": "coco:1", "kind": "image"}, {"id": "s2", "seed_id": None, "kind": "spec"}],
          "examples": {"simple": [{"id": "e1", "seed_id": "coco:2"}]}}


@pytest.fixture
def fixture(tmp_path):
    folder = tmp_path / "dataset" / "accepted"
    (folder / "route-a").mkdir(parents=True)
    (folder / "route-b").mkdir()
    a = [row("1", "coco:100"),
         row("2", "localized_narratives:200"),
         row("3", "docci:test_000123"),
         row("4", "nga:9"),
         row("5", "nga:77"),                                        # no locator: the manifest has no such picture
         row("6", "nga:10"),                                        # the manifest's locator is still pending
         row("7", "coco:1"),                                        # a screen picture
         row("8", "coco:2"),                                        # a worked example of the screen
         row("9", "nga:7"),                                         # the manifest's held split
         row("10", "nga:8"),                                        # the museum read's draw
         row("11", None, kind="synthetic", license_="Apache-2.0"),
         row("12", "coco:300", writer="model-c"),                   # a writer the run does not accept
         row("13", "coco:301", license_="CC BY-NC 4.0"),            # a license it does not accept
         row("14", "coco:302", kind="mystery"),
         row("15", "coco:303", row_id="row-1")]                     # a repeated row id
    b = [row("16", "coco:100", writer="model-b", row_id="row-16"),  # the same picture, another writer: its own item
         {**row("17", "coco:304"), "thoughts": None}]
    del b[1]["thoughts"]                                             # a row without its thoughts
    (folder / "route-a" / "t1.jsonl").write_text("\n".join(json.dumps(r) for r in a) + "\n{not json\n\n", encoding="utf-8")
    (folder / "route-b" / "t2.jsonl").write_text("\n".join(json.dumps(r) for r in b) + "\n", encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in MANIFEST) + "\n", encoding="utf-8")
    (tmp_path / "screen.json").write_text(json.dumps(SCREEN), encoding="utf-8")
    return tmp_path


def build(fixture, **kw):
    args = dict(manifest=str(fixture / "manifest.jsonl"), screen=str(fixture / "screen.json"), writers=("model-a", "model-b"),
                museum={"manifest": str(fixture / "manifest.jsonl"), "split": "main", "sources": ["nga"], "per_source": 1})
    args.update(kw)
    return rows.build_api_rows(str(fixture / "dataset"), str(fixture / "out" / "rows.json"), **args)


def test_every_filter_is_counted_and_the_kept_items_are_the_expected_rows(fixture):
    out = build(fixture)
    ids = [it["id"] for it in out["items"]]
    assert ids == sorted(ids) == ["row-1", "row-11", "row-16", "row-2", "row-3", "row-4"]
    assert out["counts"]["dropped"] == {"unreadable": 1, "no_locator": 2, "held_out_picture": 4, "writer": 1, "license": 1,
                                        "unknown_source_kind": 1, "duplicate_row_id": 1, "incomplete_row": 1}
    assert out["counts"]["by_writer"] == {"model-a": 5, "model-b": 1}
    assert out["counts"]["by_kind"] == {"image": 5, "spec": 1}
    assert out["counts"]["by_source"] == {"coco": 2, "localized_narratives": 1, "docci": 1, "nga": 1, "synthetic": 1}
    assert out["counts"]["by_license"] == {"CC BY 4.0": 5, "Apache-2.0": 1}
    assert out["version"] == 3 and out["accepted"] == str(fixture / "dataset") and out["built"]


def test_items_carry_the_writer_route_license_row_id_forms_and_locators(fixture):
    items = {it["id"]: it for it in build(fixture)["items"]}
    coco = items["row-1"]
    assert (coco["kind"], coco["writer"], coco["route"], coco["license"], coco["row_id"]) == ("image", "model-a", "route-a", "CC BY 4.0", "row-1")
    assert coco["forms"] == ["full", "short"] and coco["seed_id"] == "coco:100"
    assert coco["image_urls"] == ["http://images.cocodataset.org/val2017/000000000100.jpg",
                                  "http://images.cocodataset.org/train2017/000000000100.jpg"]
    assert items["row-2"]["image_urls"][0].endswith("000000000200.jpg")
    docci = items["row-3"]
    assert docci["image_urls"] == [] and docci["image"] == {
        "kind": "url_archive", "archive_url": "https://storage.googleapis.com/docci/data/docci_images.tar.gz",
        "member": "images/test_000123.jpg", "example_id": "test_000123"}
    assert items["row-4"]["image"] == {"kind": "url", "urls": ["http://x/9.jpg"]} and items["row-4"]["image_urls"] == []
    spec = items["row-11"]
    assert (spec["kind"], spec["forms"], spec["seed_id"], spec["image_urls"], spec["license"]) == ("spec", ["spec_full"], None, [], "Apache-2.0")
    assert items["row-16"]["seed_id"] == "coco:100" and items["row-16"]["writer"] == "model-b" and items["row-16"]["route"] == "route-b"

def test_the_held_out_pictures_come_back_when_their_filters_are_off(fixture):
    ids = lambda out: {it["id"] for it in out["items"]}
    assert "row-10" in ids(build(fixture, museum=None)) and "row-9" not in ids(build(fixture, museum=None))
    assert "row-9" in ids(build(fixture, held_manifest_splits=()))
    both = ids(build(fixture, screen=None))
    assert {"row-7", "row-8"} <= both


def test_writer_and_license_filters_can_be_set_or_lifted(fixture):
    free = build(fixture, writers=None, licenses=None)
    assert {"row-12", "row-13"} <= {it["id"] for it in free["items"]} and free["counts"]["dropped"].get("writer") is None
    narrow = build(fixture, writers=("model-b",))
    assert [it["id"] for it in narrow["items"]] == ["row-16"]
    assert narrow["counts"]["dropped"]["writer"] == 16


def test_the_build_is_reproducible_and_written_as_json(fixture):
    first = build(fixture)
    text = (fixture / "out" / "rows.json").read_text(encoding="utf-8")
    second = build(fixture)
    assert [it["id"] for it in first["items"]] == [it["id"] for it in second["items"]]
    assert json.loads(text)["items"] == json.loads((fixture / "out" / "rows.json").read_text(encoding="utf-8"))["items"]
    assert json.loads(text)["version"] == 3


def test_a_folder_without_accepted_and_a_hub_dataset_are_read(fixture, monkeypatch):
    assert rows.accepted_dir(str(fixture / "dataset" / "accepted")) == str(fixture / "dataset" / "accepted")
    assert rows.accepted_dir(str(fixture / "dataset")) == str(fixture / "dataset" / "accepted")
    pytest.importorskip("huggingface_hub")
    import huggingface_hub
    calls = []

    def fake(repo, **kw):
        calls.append((repo, kw))
        return str(fixture / "dataset")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    got = rows.read_accepted("hub:owner/dataset")
    assert calls[0][0] == "owner/dataset" and calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["allow_patterns"] == ["accepted/*/*.jsonl"]
    assert len(got) == 17 and {route for route, _ in got} == {"route-a", "route-b"}


def test_locator_and_seed_ids():
    index = rows.manifest_index(MANIFEST)
    assert set(index) == {("nga", "7"), ("nga", "8"), ("nga", "9")}      # the pending locator is not one
    assert rows.locator("nga:9", index)[1]["urls"] == ["http://x/9.jpg"]
    assert rows.locator("nga:10", index) is None and rows.locator("zzz:1", index) is None and rows.locator("coco", index) is None
    assert rows.seed_id_of(MANIFEST[0]) == "nga:7" and rows.seed_id_of({"source": "nga"}) is None


def in_view_fixture(fixture, extra):
    folder = fixture / "dataset" / "accepted" / "route-c"
    folder.mkdir()
    (folder / "t3.jsonl").write_text("\n".join(json.dumps(r) for r in extra) + "\n", encoding="utf-8")


def with_view(rid, seed_id, size, **kw):
    r = row(rid, seed_id, **kw)
    r["writer"]["image_size"] = size
    return r


def test_a_picture_in_view_takes_the_pictures_own_license(fixture):
    in_view_fixture(fixture, [with_view("20", "coco:400", [640, 480]),            # in view, picture license given and listed
                              with_view("21", "coco:401", [640, 480]),            # in view, picture license given, not listed
                              with_view("22", "coco:402", [640, 480]),            # in view, no picture license given
                              with_view("23", "coco:403", None),                  # not in view: the annotations' license
                              with_view("24", "coco:404", ""),                    # empty size: not in view
                              with_view("25", "docci:d1", [640, 480])])           # in view, but docci is not held to it
    licenses = {"coco:400": "CC BY 4.0", "coco:401": "CC BY-NC 2.0"}
    out = build(fixture, picture_licenses=licenses)
    items = {it["id"]: it for it in out["items"]}
    assert items["row-20"]["license"] == "CC BY 4.0" and "row-21" not in items and "row-22" not in items
    assert items["row-23"]["license"] == "CC BY 4.0" and items["row-24"]["license"] == "CC BY 4.0"
    assert items["row-25"]["license"] == "CC BY 4.0"
    dropped = out["counts"]["dropped"]
    assert dropped[rows.NO_PICTURE_LICENSE] == 1 and dropped["license"] == 2      # row 21 and the earlier NC row
    assert out["counts"]["hold_in_view"] == ["coco"] and out["counts"]["picture_licenses"] == {"entries": 2}


def test_a_seeded_draw_of_pictures_leaves_the_training_set_for_a_file_of_its_own(fixture):
    held_path = fixture / "out" / "held.json"
    out = build(fixture, hold_out=2, hold_seed=7, held_path=str(held_path))
    held = json.loads(held_path.read_text(encoding="utf-8"))
    drawn = {it["seed_id"] for it in held["items"]}
    assert len(drawn) == 2 and held["pictures"] == 2 and held["hold_seed"] == 7 and held["version"] == 3
    assert all(it["kind"] == "image" for it in held["items"])
    kept = {it["seed_id"] for it in out["items"] if it["kind"] == "image"}
    assert not drawn & kept and len(kept) == 2                 # four pictures before the draw; every item of a drawn one leaves
    assert out["counts"]["dropped"]["held_out_drawn"] == len(held["items"])
    assert (out["filters"]["hold_out"], out["filters"]["hold_seed"], out["filters"]["held_drawn"]) == (2, 7, 2)
    again = build(fixture, hold_out=2, hold_seed=7, held_path=str(held_path))
    assert [it["id"] for it in again["items"]] == [it["id"] for it in out["items"]]     # the draw is fixed by the seed
    everything = build(fixture, hold_out=99, hold_seed=7, held_path=str(held_path))     # more than there are: all of them
    assert [it["kind"] for it in everything["items"]] == ["spec"]
    plain = build(fixture)
    assert "held_out_drawn" not in plain["counts"]["dropped"] and plain["filters"]["held_drawn"] == 0
    with pytest.raises(ValueError, match="held_path"):
        build(fixture, hold_out=1)


def test_screen_files_and_manifests_can_be_lists(fixture):
    other = fixture / "other_held.json"
    other.write_text(json.dumps({"items": [{"id": "h1", "kind": "image", "seed_id": "docci:test_000123"}]}), encoding="utf-8")
    out = build(fixture, screen=[str(fixture / "screen.json"), str(other)])
    assert "row-3" not in {it["id"] for it in out["items"]} and out["counts"]["dropped"]["held_out_picture"] == 5
    extra = fixture / "extra.jsonl"
    extra.write_text(json.dumps({"sample_id": "x-1", "split": "packed", "source": "nga", "source_id": "77",
                                 "image": {"kind": "hf_parquet", "repo": "o/r", "path": "p.parquet", "row_index": 0}}) + "\n",
                     encoding="utf-8")
    two = build(fixture, manifest=[str(fixture / "manifest.jsonl"), str(extra)])
    items = {it["id"]: it for it in two["items"]}
    assert items["row-5"]["image"]["kind"] == "hf_parquet" and two["counts"]["dropped"]["no_locator"] == 1   # nga:10 still pending
    assert "row-9" not in items                                                    # the first manifest's probe split still held
    assert rows.seeds_without_locator(str(fixture / "dataset"), str(fixture / "manifest.jsonl"), ["nga"]) == ["nga:10", "nga:77"]
    assert rows.seeds_without_locator(str(fixture / "dataset"), [str(fixture / "manifest.jsonl"), str(extra)]) == ["nga:10"]


def test_the_queue_tool_ships_the_held_pictures_beside_the_training_set(fixture):
    from qwen_symbolic_thinking.tools import build_rows
    folder = fixture / "ship"
    summary = build_rows.run(str(fixture / "dataset"), tag="v_rows", do_ship=False, out_dir=str(folder),
                             manifest=str(fixture / "manifest.jsonl"), screen=str(fixture / "screen.json"),
                             writers=["model-a", "model-b"], hold_out=1, hold_seed=3)
    assert summary["path_in_repo"] == "sft/v_rows.json" and summary["held_path_in_repo"] == "sft/v_rows_held.json"
    assert (folder / "v_rows.json").exists() and (folder / "v_rows_held.json").exists()
    assert summary["filters"]["held_drawn"] == 1
    assert build_rows.run(str(fixture / "dataset"), tag="v2", do_ship=False, out_dir=str(folder / "b"),
                          screen=str(fixture / "screen.json"))["held_path_in_repo"] is None


def test_picture_licenses_from_a_file_and_other_held_sources(fixture):
    in_view_fixture(fixture, [with_view("30", "coco:410", [1, 1]), with_view("31", "docci:d2", [1, 1])])
    path = fixture / "licenses.json"
    path.write_text(json.dumps({"coco:410": "Apache-2.0", "docci:d2": "CC BY-NC 4.0"}), encoding="utf-8")
    out = build(fixture, picture_licenses=str(path), hold_in_view=("coco", "docci"))
    items = {it["id"]: it for it in out["items"]}
    assert items["row-30"]["license"] == "Apache-2.0" and "row-31" not in items    # the picture's NC license is not listed
    assert out["counts"]["picture_licenses"] == str(path) and out["counts"]["hold_in_view"] == ["coco", "docci"]
    none = build(fixture, hold_in_view=())                                          # nothing held to it: the label's license
    assert {"row-30", "row-31"} <= {it["id"] for it in none["items"]}
