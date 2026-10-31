"""Continuing a training run from a trained adapter: where the adapter is read from, and the option on train.run."""
import inspect

from qwen_symbolic_thinking import train


def test_adapter_source_hub_local_and_repo():
    assert train.adapter_source("hub:lora/Qwen3.5-9B/r32/20261030_153121_G_self_2ep/final") == (
        train.REPO, "lora/Qwen3.5-9B/r32/20261030_153121_G_self_2ep/final")
    assert train.adapter_source("hub:/lora/x/final/", repo="owner/other") == ("owner/other", "lora/x/final")
    assert train.adapter_source("/content/train/run/adapter") == ("/content/train/run/adapter", None)
    assert train.adapter_source("AbstractPhil/qwen3.5-9b-think-symbolic-image") == ("AbstractPhil/qwen3.5-9b-think-symbolic-image", None)


def test_run_takes_init_adapter_off_by_default():
    params = inspect.signature(train.run).parameters
    assert "init_adapter" in params and params["init_adapter"].default is None
