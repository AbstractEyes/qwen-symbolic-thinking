"""The runs that follow a training unattended: a train spec's "after" entries read the adapter it trained with the trained model."""
import pytest

from qwen_symbolic_thinking import queue

PROBE = {"manifest": "hub:AbstractPhil/qwen-thinking-images/manifest_v1.4.jsonl", "split": "probe", "batch": 64, "max_new_tokens": 800}
SPEC = {"name": "G_med4_2ep_4b", "kind": "train",
        "train": {"model_id": "Qwen/Qwen3.5-4B", "rank": 32, "alpha": 64, "epochs": 2, "lr": 2e-4, "inputs": ["hub:sft/pictures_v2.json"]},
        "after": [{"name": "P_probe_4b", "kind": "extract", "extract": PROBE},
                  {"name": "P_museum_4b", "kind": "extract", "extract": {**PROBE, "split": "main", "model_id": "Qwen/Qwen3.5-9B"}}]}
PATH = "lora/Qwen3.5-4B/r32/20261101_010203_G_med4_2ep_4b"


def test_after_specs_read_the_trained_adapter_with_the_trained_model():
    out = queue.after_specs(SPEC, PATH)
    assert [s["name"] for s in out] == ["P_probe_4b", "P_museum_4b"]
    probe = out[0]["extract"]
    assert probe["adapter"] == queue.REPO and probe["adapter_subfolder"] == PATH + "/final"
    assert probe["model_id"] == "Qwen/Qwen3.5-4B" and probe["split"] == "probe" and probe["max_new_tokens"] == 800
    assert "system" not in probe
    assert out[1]["extract"]["model_id"] == "Qwen/Qwen3.5-9B"        # a key the entry sets itself is kept
    assert SPEC["after"][0]["extract"] == PROBE                         # the queue's own entries are not changed


def test_after_specs_carry_the_system_prompt_and_accept_other_repos():
    spec = {**SPEC, "train": {**SPEC["train"], "system": "v1"}}
    out = queue.after_specs(spec, "/" + PATH + "/", repo="owner/other")
    assert out[0]["extract"]["system"] == "v1"
    assert out[0]["extract"]["adapter"] == "owner/other" and out[0]["extract"]["adapter_subfolder"] == PATH + "/final"


def test_after_specs_without_after_or_with_a_non_extract_run():
    assert queue.after_specs({"name": "x", "kind": "train", "train": {}}, PATH) == []
    with pytest.raises(ValueError):
        queue.after_specs({**SPEC, "after": [{"name": "T", "kind": "train", "train": {}}]}, PATH)
