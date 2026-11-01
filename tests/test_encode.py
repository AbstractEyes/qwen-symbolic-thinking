"""The batch encoder serves both picture conventions (Qwen3.5's flat list, Gemma 4's list per text), and the thinking-on decode keeps
Gemma's thought channel as <think> tags for the parse."""
from qwen_symbolic_thinking.screen import encode, plain, special_tokens


class Pic:
    def __init__(self, name):
        self.name = name


class FlatProcessor:
    """Takes the pictures as one flat list (Qwen3.5)."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, images=None, **kw):
        self.calls.append(images)
        return {"text": text, "images": images}


class NestedProcessor:
    """Reads a flat list as one text with several pictures and refuses the count (Gemma 4); takes one list per text."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, images=None, **kw):
        self.calls.append(images)
        if images is not None and not all(isinstance(x, list) for x in images):
            raise ValueError(f"Received inconsistently sized batches of images (1) and text ({len(text)}).")
        return {"text": text, "images": images}


def test_encode_flat_first():
    p = FlatProcessor()
    out = encode(p, ["a", "b"], [Pic("a"), Pic("b")], padding=True)
    assert [x.name for x in out["images"]] == ["a", "b"] and len(p.calls) == 1


def test_encode_nested_on_the_count_error_with_a_slot_per_text():
    p = NestedProcessor()
    a = Pic("a")
    out = encode(p, ["a", "s"], [a, None])
    assert out["images"] == [[a], []] and len(p.calls) == 2


def test_encode_no_pictures_passes_none():
    p = NestedProcessor()
    assert encode(p, ["s", "t"], [None, None])["images"] is None


def test_encode_other_errors_raise():
    class Bad:
        def __call__(self, text, images=None, **kw):
            raise ValueError("something else")
    try:
        encode(Bad(), ["a"], [Pic("a")])
    except ValueError as e:
        assert "something else" in str(e)
    else:
        raise AssertionError("an unrelated error must surface")


class Added:
    def __init__(self, content, special):
        self.content, self.special = content, special

    def __str__(self):
        return self.content


class GemmaTok:
    all_special_tokens = ["<pad>", "<eos>", "<turn|>"]
    added_tokens_decoder = {105: Added("<|turn>", True), 106: Added("<turn|>", True), 7: Added("<|channel>", True), 8: Added("<channel|>", True)}


def test_special_tokens_union():
    assert special_tokens(GemmaTok()) == ["<pad>", "<eos>", "<turn|>", "<|turn>", "<|channel>", "<channel|>"]


def test_plain_keeps_the_thought_as_think_tags():
    text = "<|channel>thought\nlook at the lamp first<channel|>see: a lamp.\ncaption: A lamp.<turn|>\n<pad><pad>"
    assert plain(text, GemmaTok()) == "<think>look at the lamp first</think>see: a lamp.\ncaption: A lamp.\n"


def test_plain_leaves_qwen_think_tags():
    class QwenTok:
        all_special_tokens = ["<|im_end|>", "<|endoftext|>"]
        added_tokens_decoder = {}
    assert plain("<think>\nhm\n</think>\n\nsee: x<|im_end|>\n<|endoftext|>", QwenTok()) == "<think>\nhm\n</think>\n\nsee: x\n"
