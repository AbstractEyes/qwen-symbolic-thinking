"""The extraction module without a model: manifest loading, the grounding hint, the selection and the audit of a parsed row."""
import json

from qwen_symbolic_thinking import chain, extract
from qwen_symbolic_thinking.synth import audit


def _manifest(tmp_path, rows):
    p = tmp_path / "manifest.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def test_load_manifest_local(tmp_path):
    rows = [{"sample_id": "qti-000001", "source": "coco", "group": "photo", "depth_plan": "closer",
             "image": {"kind": "url", "urls": ["http://example.invalid/a.jpg"]}, "captions": {"seed_hint": "a dog on a couch"}}]
    assert extract.load_manifest(_manifest(tmp_path, rows)) == rows


def test_hint_joins_plain_captions_only():
    row = {"captions": {"caption_1": "a red barn", "vlm_0.8b_json": "{\"objects\": [\"barn\"]}", "animetimm_json": "{}", "empty": "  "}}
    assert extract.hint_of(row) == "a red barn"
    assert extract.hint_of({"captions": {}}) == ""


def test_depth_plan_maps_to_chain_depths():
    assert [extract.DEPTHS[d] for d in ("quick", "closer", "long")] == ["simple", "medium", "deep"]


def test_audit_of_a_whole_row_with_and_without_a_hint():
    text = ("see: a red barn stands in a field\nsee: a white fence runs along the front\nmood: a calm morning\n"
            "caption: a red barn behind a white fence in a quiet field\nprompt: red barn, white fence, morning light\n"
            "question: barn colour -> red [chain]\nquestion: time of day -> morning [chain]\nquestion: location -> unknown [unknown]")
    parsed = chain.parse(text)
    scores = chain.score(parsed, {"depth": "medium", "spec": {}, "thoughts": [], "hint": "a red barn behind a white fence"})
    assert scores["full_ok"], scores
    assert audit(scores, 0.2, 0.1) is None
    # the same row with no hint: grounding gates off, shape still read
    bare = chain.score(parsed, {"depth": "medium", "spec": {}, "thoughts": [], "hint": ""})
    assert bare["hint_said"] is None and bare["full_ok"]
    assert audit(bare, 0.0, 0.0) is None
