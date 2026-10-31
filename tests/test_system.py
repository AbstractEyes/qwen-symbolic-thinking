"""The system prompt: how a spec names it, and that the trainer's and the screen's conversations open with it when given."""
import inspect

from thought_seeds.thought_tools import STEPS

from qwen_symbolic_thinking import extract, prompts, screen, train


def test_system_lookup():
    assert prompts.system(None) is None and prompts.system("") is None
    assert prompts.system("v1") == prompts.SYSTEM_V1
    assert prompts.system("Look twice.") == "Look twice."
    for step in STEPS:
        assert step in prompts.SYSTEM_V1
    for word in ("caption:", "prompt:", "question:", "[record, chain or unknown]", "painting", "likely"):
        assert word in prompts.SYSTEM_V1


def test_screen_messages_open_with_the_system_turn():
    it = {"id": "x", "kind": "image", "depth": "medium", "spec": {}}
    plain = screen.messages("full", it, {}, None)
    assert [m["role"] for m in plain] == ["user"]
    with_system = screen.messages("full", it, {}, None, "SYS")
    assert [m["role"] for m in with_system] == ["system", "user"]
    assert with_system[0]["content"] == "SYS" and with_system[1] == plain[0]


def test_every_runner_takes_system_off_by_default():
    for fn in (train.run, screen.run, extract.run):
        params = inspect.signature(fn).parameters
        assert "system" in params and params["system"].default is None, fn.__name__
    assert inspect.signature(train.conversation).parameters["system"].default is None
