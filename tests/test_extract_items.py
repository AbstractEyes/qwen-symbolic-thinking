"""The extraction's rows as training items, and the locator reading of training items (no network, no model)."""
import json

from qwen_symbolic_thinking import chain, extract

GOOD = ("see: a red barn stands in a field\nsee: a white fence runs along the front\nmood: a calm morning\n"
        "caption: a red barn behind a white fence in a quiet field\nprompt: red barn, white fence, morning light\n"
        "question: barn colour -> red [chain]\nquestion: time of day -> morning [chain]\nquestion: location -> unknown [unknown]")


def _row(sample_id, source, hinted, backed, said, rejected=None, kind="url"):
    parsed = chain.parse(GOOD)
    image = {"kind": "url", "urls": ["http://example.invalid/a.jpg"]} if kind == "url" else {"kind": "hf_parquet", "repo": "x/y", "path": "p.parquet", "row_index": 3}
    return {"sample_id": sample_id, "source": source, "group": "photo", "depth": "medium", "text": GOOD, "thoughts": parsed["thoughts"],
            "wordings": parsed["wordings"], "questions": parsed["questions"], "rejected": rejected, "hinted": hinted, "image": image,
            "scores": {"hint_backed": backed, "hint_said": said, "full_ok": True}}


def test_training_items_gates_and_round_trip(tmp_path):
    rows = [_row("qti-000001", "coco", True, 0.45, 0.30), _row("qti-000002", "coco", True, 0.20, 0.30),
            _row("qti-000003", "trdg_words", False, None, None), _row("qti-000004", "nga", True, 0.50, 0.40, rejected="gloss", kind="hf_parquet"),
            _row("qti-000005", "ffhq", True, 0.40, 0.20, kind="hf_parquet")]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    data = extract.training_items(str(p), out_path=str(tmp_path / "items.json"))
    ids = sorted(it["id"] for it in data["items"])
    assert ids == ["qti-000001", "qti-000005"], (ids, data["left_out"])
    assert data["left_out"] == {"under the grounding floors": 1, "no captions": 1, "rejected by the audit": 1}
    it = next(i for i in data["items"] if i["id"] == "qti-000005")
    assert it["forms"] == ["full"] and it["image"]["kind"] == "hf_parquet" and it["image_urls"] == []
    # the item renders back to the row the model wrote
    assert chain.parse(chain.render(it, full=True))["thoughts"] == it["thoughts"]
    assert json.load(open(tmp_path / "items.json", encoding="utf-8"))["by_source"] == {"coco": 1, "ffhq": 1}


def test_training_items_either_rule_caps_and_exclusions(tmp_path):
    rows = [_row("qti-000001", "coco", True, 0.10, 0.30), _row("qti-000002", "coco", True, 0.40, 0.05),
            _row("qti-000003", "coco", True, 0.10, 0.10), _row("qti-000004", "instagram", True, 0.90, 0.90),
            _row("qti-000005", "ffhq", True, 0.40, 0.05), _row("qti-000006", "ffhq", True, 0.40, 0.20)]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    data = extract.training_items(str(p), min_backed=0.25, min_said=0.15, grounding="either", exclude_sources=("instagram",), per_source_max=1)
    assert data["by_source"] == {"coco": 1, "ffhq": 1}
    assert data["left_out"] == {"under the grounding floors": 1, "source left out": 1, "over the source's cap": 2}
    both = extract.training_items(str(p), min_backed=0.25, min_said=0.15, grounding="both", exclude_sources=("instagram",))
    assert [it["id"] for it in both["items"]] == ["qti-000006"]          # the one row over both floors


def test_locator_row_reads_training_items():
    r = extract.locator_row({"id": "row-1", "image_urls": ["http://example.invalid/a.jpg", "http://example.invalid/b.jpg"]})
    assert r["sample_id"] == "row-1" and r["image"] == {"kind": "url", "urls": ["http://example.invalid/a.jpg", "http://example.invalid/b.jpg"]}
    r2 = extract.locator_row({"id": "row-2", "image": {"kind": "hf_file", "repo": "x/y", "path": "a.png"}})
    assert r2["sample_id"] == "row-2" and r2["image"]["kind"] == "hf_file"
    m = {"sample_id": "qti-1", "image": {"kind": "url", "urls": ["u"]}}
    assert extract.locator_row(m) is m
