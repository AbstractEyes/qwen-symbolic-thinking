"""The text-in-pictures prep (v0.1.65): the general-row tool's own ask banks, and manifests read from the project repo."""
import json
import random

import pytest

from qwen_symbolic_thinking import extract, hub
from qwen_symbolic_thinking.tools import anchors


def old_draw(n, seed):
    """draw_requests as it was before banks (v0.1.64), to show the default draw did not move."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        bank = list(anchors.BANK)
        rng.shuffle(bank)
        out += bank
    return out[:n]


def test_default_draw_unchanged():
    for n, seed in ((1, 0), (22, 20261110), (100, 20261112), (1600, 20261110)):
        assert anchors.draw_requests(n, seed) == old_draw(n, seed)
        assert anchors.draw_requests(n, seed, None) == old_draw(n, seed)
        assert anchors.draw_requests(n, seed, "general") == old_draw(n, seed)


def test_text_bank():
    asks = anchors.draw_requests(27, 7, "text")
    assert set(asks) == set(anchors.TEXT_BANK)
    assert all(asks.count(a) == 3 for a in anchors.TEXT_BANK)   # three whole passes over the nine asks
    assert anchors.draw_requests(27, 7, "text") == asks          # seeded


def test_bank_of():
    assert anchors.bank_of(None) == anchors.BANK
    assert anchors.bank_of("text") == anchors.TEXT_BANK
    assert anchors.bank_of([" Read the sign. ", "What does it say?"]) == ("Read the sign.", "What does it say?")
    with pytest.raises(ValueError):
        anchors.bank_of("ocr")
    with pytest.raises(ValueError):
        anchors.bank_of([])
    with pytest.raises(ValueError):
        anchors.bank_of(["ok", "  "])


def test_manifest_from_the_project_repo(tmp_path, monkeypatch):
    rows = [{"sample_id": "coco:1", "source": "coco", "group": "text_held_apache"}]
    path = tmp_path / "m.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    calls = []

    def fake_download(repo_id, filename, repo_type=None, token=None, **kw):
        calls.append((repo_id, filename, repo_type))
        return str(path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert extract.load_manifest("repo:text/text_held_v1.jsonl") == rows
    assert calls == [(hub.REPO, "text/text_held_v1.jsonl", "model")]
    assert extract.load_manifest(str(path)) == rows   # a local path still reads as before
