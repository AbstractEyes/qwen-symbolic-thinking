"""Seams for vision-language families beyond Qwen3.5 and Gemma 4: a processor that refuses a flat picture list in its own words
gets the per-text form; a configuration the multimodal auto class does not map loads through the next class; GLM-4V's fused MLP
projection is a LoRA target while the vision tower stays out."""
import re

from qwen_symbolic_thinking.screen import encode, load_vlm
from qwen_symbolic_thinking.train import TARGETS


class Pic:
    def __init__(self, name):
        self.name = name


class OwnWordsProcessor:
    """Refuses a flat list for a batch of texts with an error worded its own way; takes one list per text."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, images=None, **kw):
        self.calls.append(images)
        if images is not None and len(text) > 1 and not all(isinstance(x, list) for x in images):
            raise ValueError("Invalid input images. Please provide a list of lists of images.")
        return {"text": text, "images": images}


def test_encode_per_text_on_an_error_in_other_words():
    p = OwnWordsProcessor()
    a, b = Pic("a"), Pic("b")
    out = encode(p, ["a", "b", "s"], [a, b, None])
    assert out["images"] == [[a], [b], []] and len(p.calls) == 2


def test_encode_first_error_surfaces_when_both_forms_fail():
    class Refuses:
        def __call__(self, text, images=None, **kw):
            raise ValueError("first words" if not any(isinstance(x, list) for x in images) else "second words")
    try:
        encode(Refuses(), ["a", "b"], [Pic("a"), Pic("b")])
    except ValueError as e:
        assert "first words" in str(e)
    else:
        raise AssertionError("the first error must surface")


class Unmapped:
    calls = 0

    @classmethod
    def from_pretrained(cls, src, **kw):
        cls.calls += 1
        raise ValueError(f"Unrecognized configuration class <class 'X'> for this kind of AutoModel: {src}")


class Mapped:
    @classmethod
    def from_pretrained(cls, src, **kw):
        return ("loaded", src, kw.get("dtype"))


class Broken:
    @classmethod
    def from_pretrained(cls, src, **kw):
        raise ValueError("the weights are missing")


def test_load_vlm_falls_through_an_unmapped_class():
    assert load_vlm("org/model", classes=[Unmapped, Mapped], dtype="bf16") == ("loaded", "org/model", "bf16")


def test_load_vlm_surfaces_other_errors_and_the_last_unmapped():
    try:
        load_vlm("org/model", classes=[Broken, Mapped])
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("an unrelated load error must surface")
    try:
        load_vlm("org/model", classes=[Unmapped])
    except ValueError as e:
        assert "Unrecognized configuration class" in str(e)
    else:
        raise AssertionError("no class mapped the configuration")


class Mask:
    """An attention mask's interface as the collator uses it: per-row sums as a list."""

    def __init__(self, rows):
        self.rows = rows

    def sum(self, dim):
        assert dim == 1
        return Rows([sum(r) for r in self.rows])


class Rows:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_overrun_counts_the_examples_past_max_len():
    from qwen_symbolic_thinking.train import overrun
    mask = Mask([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    assert overrun(mask, 4) == (3, 1, 5)
    assert overrun(mask, 5) == (3, 0, 5)
    assert overrun(Mask([]), 5) == (0, 0, 0)


def test_targets_take_the_fused_mlp_and_leave_the_vision_tower():
    assert re.fullmatch(TARGETS, "model.language_model.layers.3.mlp.gate_up_proj")
    assert re.fullmatch(TARGETS, "model.language_model.layers.3.mlp.down_proj")
    assert re.fullmatch(TARGETS, "model.language_model.layers.3.self_attn.q_proj")
    assert not re.fullmatch(TARGETS, "model.visual.blocks.3.mlp.gate_up_proj")
    assert not re.fullmatch(TARGETS, "model.vision_tower.transformer.layers.0.attention.q_proj")
