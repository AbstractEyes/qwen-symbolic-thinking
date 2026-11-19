"""The extraction's ask suffix without a model: the ask a manifest row is given (extract.ask_for) with and without a suffix and
helper lines, and the queue passing the suffix through to extract.run."""
from qwen_symbolic_thinking import extract, prompts, queue

SUFFIX = " Ensure you include visible text and the expression related to it into the rows."
CAPTION = "a woman wearing a pink shirt with a yellow flower on it in front of a white background"


def test_the_trained_ask_alone_by_depth():
    for plan, depth in (("quick", "simple"), ("closer", "medium"), ("long", "deep"), (None, "medium")):
        assert extract.ask_for({"depth_plan": plan}) == prompts.full(depth)


def test_the_suffix_follows_the_trained_ask_as_written():
    row = {"depth_plan": "closer"}
    assert extract.ask_for(row, ask_suffix=SUFFIX) == prompts.full("medium") + SUFFIX
    assert extract.ask_for(row, ask_suffix="") == extract.ask_for(row, ask_suffix=None) == prompts.full("medium")
    assert extract.ask_for(row, ask_suffix="\nOne more line.") == prompts.full("medium") + "\nOne more line."


def test_helper_lines_come_after_the_suffix():
    row = {"sample_id": "qti-000001", "source": "deepfashion", "depth_plan": "closer",
           "captions": {"deepfashion_caption": CAPTION}}
    assert extract.ask_for(row, ("caption",), SUFFIX) == f"{prompts.full('medium')}{SUFFIX}\nHelper caption: {CAPTION}"
    assert extract.ask_for(row, ("caption",)) == extract.with_helpers(prompts.full("medium"), {"caption": CAPTION})


def test_the_queue_passes_the_suffix_through(monkeypatch):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"rows": 0}

    monkeypatch.setattr(extract, "run", fake_run)
    monkeypatch.setattr(queue, "_mark", lambda *a, **k: None)
    spec = {"name": "XA_test", "kind": "extract", "extract": {"manifest": "m.jsonl", "ask_suffix": SUFFIX}}
    result = queue.run_spec(spec, "lane1", "repo", "/tmp/out")
    assert seen["ask_suffix"] == SUFFIX and seen["tag"] == "XA_test"
    assert result["status"] == "done"
