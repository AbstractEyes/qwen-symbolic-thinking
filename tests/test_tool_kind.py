"""The queue's "tool" kind runs a diagnostic of the tools package by name, with its args, the run's name as tag and the repo; the
picture probe's batch surgery rolls a picture to the next row (Gemma's stacked pictures, Qwen's flattened patches) and refuses rows
whose picture token counts differ."""
import sys
import types

import pytest

from qwen_symbolic_thinking import queue
from qwen_symbolic_thinking.tools import picture_probe


def test_tool_kind_runs_the_named_module_with_tag_and_repo(monkeypatch):
    fake = types.ModuleType("qwen_symbolic_thinking.tools.fake_probe")
    fake.run = lambda **kw: {"ok": True, "args": kw}
    monkeypatch.setitem(sys.modules, "qwen_symbolic_thinking.tools.fake_probe", fake)
    marks = []
    monkeypatch.setattr(queue, "_mark", lambda name, result, repo, out_dir: marks.append((name, result["status"])))
    spec = {"name": "D_fake", "kind": "tool", "needs": "0.1.23", "tool": {"module": "fake_probe", "args": {"model_id": "m", "n_rows": 2}}}
    result = queue.run_spec(spec, "lane", "owner/repo", "/tmp/runs")
    assert result["status"] == "done" and result["tool"]["ok"]
    assert result["tool"]["args"] == {"model_id": "m", "n_rows": 2, "tag": "D_fake", "repo": "owner/repo"}
    assert marks == [("D_fake", "started")]


def test_picture_probe_names_resolve():
    """The probe imports lazily inside run(); the names it takes from the package exist where it looks for them."""
    from qwen_symbolic_thinking.hub import resolve_model
    from qwen_symbolic_thinking.screen import encode, load_model, vlm_class
    from qwen_symbolic_thinking.template import stop_ids, turn_end
    from qwen_symbolic_thinking.train import TARGETS, Collator, conversation, fetch_images, load_items
    assert all(callable(f) for f in (resolve_model, vlm_class, load_model, encode, stop_ids, turn_end, conversation, fetch_images, load_items))
    assert TARGETS
    assert Collator.__call__ and picture_probe.run.__code__.co_varnames[:2] == ("model_id", "n_rows")


def test_unknown_kind_still_refused():
    with pytest.raises(ValueError):
        queue.run_spec({"name": "X", "kind": "dance"}, "lane", "owner/repo", "/tmp/runs")


def test_rolled_moves_stacked_pictures_to_the_next_row():
    torch = pytest.importorskip("torch")
    pix = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    pos = torch.arange(3 * 4 * 2).reshape(3, 4, 2)
    batch = {"input_ids": torch.zeros(3, 5, dtype=torch.long), "attention_mask": torch.ones(3, 5, dtype=torch.long),
             "labels": torch.zeros(3, 5, dtype=torch.long), "mm_token_type_ids": torch.zeros(3, 5, dtype=torch.long),
             "pixel_values": pix, "image_position_ids": pos}
    out = picture_probe.rolled(batch)
    assert torch.equal(out["pixel_values"][1], pix[0]) and torch.equal(out["pixel_values"][0], pix[2])
    assert torch.equal(out["image_position_ids"][1], pos[0])
    assert torch.equal(out["input_ids"], batch["input_ids"]) and torch.equal(out["mm_token_type_ids"], batch["mm_token_type_ids"])
    assert torch.equal(batch["pixel_values"], pix)                  # the given batch is not changed


def test_rolled_moves_flattened_patches_by_whole_pictures_and_refuses_unequal_ones():
    torch = pytest.importorskip("torch")
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pix = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    out = picture_probe.rolled({"input_ids": torch.zeros(2, 3, dtype=torch.long), "pixel_values": pix, "image_grid_thw": grid})
    assert out["pixel_values"].flatten().tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert picture_probe.rolled({"pixel_values": pix, "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 4]])}) is None
    assert picture_probe.rolled({"pixel_values": pix[:4].reshape(1, 4, 1)}) is None     # one picture: nothing to roll


def test_blanked_zeroes_the_pixels_only_and_verdicts():
    torch = pytest.importorskip("torch")
    b = {"pixel_values": torch.ones(2, 3), "input_ids": torch.ones(2, 2, dtype=torch.long)}
    out = picture_probe.blanked(b)
    assert float(out["pixel_values"].sum()) == 0 and torch.equal(out["input_ids"], b["input_ids"]) and float(b["pixel_values"].sum()) == 6
    assert picture_probe.verdict(1.0, 1.5, 1.6) == "sees"
    assert picture_probe.verdict(1.0, 1.01, 1.02) == "blind"
    assert picture_probe.verdict(1.0, None, 1.5) == "sees"
    assert picture_probe.verdict(1.0, 1.1, 1.1) == "partial"
