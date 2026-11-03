"""The extraction's helper lines without a model: the booru tagger's JSON as one tags line, the lines a manifest row gives, the ask
they follow, and the queue passing them through."""
import json

import pytest

from qwen_symbolic_thinking import extract, prompts, queue

TAGGER = json.dumps({"subjects": [{"name": "girl", "attributes": ["solo"]}, {"name": "pants", "attributes": ["blue"]},
                                  {"name": "jeans", "attributes": []}, {"name": "denim", "attributes": []},
                                  {"name": "shirt", "attributes": ["white"]}, {"name": "earrings", "attributes": []},
                                  {"name": "hair", "attributes": ["brown"]}, {"name": "sleeves", "attributes": ["short"]},
                                  {"name": "lips", "attributes": []}, {"name": "t-shirt", "attributes": []},
                                  {"name": "cowboy shot", "attributes": []}, {"name": "print shirt", "attributes": []},
                                  {"name": "background", "attributes": ["grey", "simple"]}],
                     "actions": ["standing"], "setting": "unknown"})
TAGS = ("girl (solo) · pants (blue) · jeans · denim · shirt (white) · earrings · hair (brown) · sleeves (short) · lips · t-shirt · "
        "cowboy shot · print shirt · background (grey, simple) · doing: standing")
CAPTION = "a woman wearing a pink shirt with a yellow flower on it in front of a white background"


def _row(tagger=TAGGER):
    return {"sample_id": "qti-000001", "source": "deepfashion", "depth_plan": "closer",
            "captions": {"deepfashion_caption": CAPTION, "vlm_0.8b_json": "{\"subjects\": []}", "animetimm_json": tagger}}


def test_tag_line_reads_the_tagger_json_as_one_line():
    assert extract.tag_line(TAGGER) == TAGS
    with_setting = json.dumps({"subjects": [{"name": "boy", "attributes": []}], "actions": [], "setting": "beach"})
    assert extract.tag_line(with_setting) == "boy · setting: beach"


@pytest.mark.parametrize("raw", [None, "", "{}", "null", "__PARSEFAIL__", "[1, 2]", "{\"subjects\": []}"])
def test_tag_line_is_empty_without_tags(raw):
    assert extract.tag_line(raw) == ""


def test_helper_lines_of_a_row():
    assert extract.helper_lines(_row(), ("caption", "tags")) == {"caption": CAPTION, "tags": TAGS}
    assert extract.helper_lines(_row("__PARSEFAIL__"), ("caption", "tags")) == {"caption": CAPTION}
    assert extract.helper_lines(_row(), ("tags",)) == {"tags": TAGS}
    with pytest.raises(ValueError):
        extract.helper_lines(_row(), ("colour",))


def test_the_ask_with_helpers_follows_the_trained_ask_caption_first():
    ask = prompts.full("medium")
    lines = extract.helper_lines(_row(), ("tags", "caption"))
    assert extract.with_helpers(ask, lines) == f"{ask}\nHelper caption: {CAPTION}\nHelper tags: {TAGS}"
    assert extract.with_helpers(ask, {}) == ask


def test_the_queue_passes_helpers_through_as_a_tuple(monkeypatch):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"rows": 0}

    monkeypatch.setattr(extract, "run", fake_run)
    monkeypatch.setattr(queue, "_mark", lambda *a, **k: None)
    spec = {"name": "XH_test", "kind": "extract",
            "extract": {"manifest": "m.jsonl", "sources": ["deepfashion"], "helpers": ["caption", "tags"], "split": None}}
    result = queue.run_spec(spec, "lane1", "repo", "/tmp/out")
    assert seen["helpers"] == ("caption", "tags") and seen["sources"] == ("deepfashion",) and seen["tag"] == "XH_test"
    assert result["status"] == "done"
