"""The runs that follow a training unattended: a train spec's "after" entries read what the training shipped (a LoRA on its base,
or a full fine-tune's folder); a spec that needs a newer package stays in the queue."""
import pytest

from qwen_symbolic_thinking import __version__, hub, queue

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


def test_after_specs_of_a_full_fine_tune_point_at_its_folder():
    spec = {**SPEC, "name": "F_med4_2ep_4b", "train": {**SPEC["train"], "rank": 0, "lr": 1e-5}}
    out = queue.after_specs(spec, "full/Qwen3.5-4B/20261101_010203_F_med4_2ep_4b")
    assert out[0]["extract"]["model_id"] == "hub:full/Qwen3.5-4B/20261101_010203_F_med4_2ep_4b/final"
    assert out[0]["extract"]["adapter"] is None and "adapter_subfolder" not in out[0]["extract"]
    assert hub.model_short(out[0]["extract"]["model_id"]) == "Qwen3.5-4B"


def test_after_specs_without_after_or_with_a_non_extract_run():
    assert queue.after_specs({"name": "x", "kind": "train", "train": {}}, PATH) == []
    with pytest.raises(ValueError):
        queue.after_specs({**SPEC, "after": [{"name": "T", "kind": "train", "train": {}}]}, PATH)


def test_a_spec_that_needs_a_newer_package_waits_for_one():
    assert queue.runnable_here({"name": "a", "kind": "train"})
    assert queue.runnable_here({"name": "a", "kind": "train", "needs": __version__})
    assert queue.runnable_here({"name": "a", "kind": "train", "needs": "v0.1.0"})
    assert not queue.runnable_here({"name": "a", "kind": "train", "needs": "9.0.0"})


def test_model_ids_resolve_and_shorten_without_the_hub():
    assert hub.resolve_model("Qwen/Qwen3.5-2B") == "Qwen/Qwen3.5-2B"
    assert hub.resolve_model("/content/train/x/adapter") == "/content/train/x/adapter"
    assert hub.model_short("Qwen/Qwen3.5-0.8B") == "Qwen3.5-0.8B"
    assert hub.model_short("hub:full/Qwen3.5-2B/20261101_000000_F/final") == "Qwen3.5-2B"
