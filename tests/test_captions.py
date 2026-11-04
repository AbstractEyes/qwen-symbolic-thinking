"""captions and booru: the pure parts (the ask for one photo, the copy share, the tag filter), no GPU."""
from qwen_symbolic_thinking import booru
from qwen_symbolic_thinking.captions import FASHION_ASK, copy_share, fill, one_line


def test_fill_puts_the_description_on_one_line():
    ask = fill(FASHION_ASK, "The lady wears a tank top.\nThe tank top is cotton.")
    assert ask.endswith("Helper caption: The lady wears a tank top. The tank top is cotton.")
    assert "{original}" not in ask


def test_fill_without_a_description_drops_the_helper_line():
    ask = fill(FASHION_ASK, "  ")
    assert "Helper caption" not in ask and "{original}" not in ask
    assert ask.startswith("Describe this picture")


def test_copy_share():
    original = "The person wears a short-sleeve T-shirt with pure color patterns."
    assert copy_share("A short-sleeve T-shirt with pure color patterns.", original) == 1.0
    assert copy_share("A woman in a red dress walks along a beach at dusk.", original) == 0.0
    assert copy_share("Too short.", original) is None
    assert copy_share("A woman in a red dress walks along a beach.", "") is None


def test_one_line():
    assert one_line(" a\n b \t c ") == "a b c"


def test_keep_omit_floor_and_clothes():
    omit = ("short hair", "grey hair")
    assert not booru.keep("short_hair", 0.9, omit)
    assert not booru.keep("gray_hair", 0.9, omit)                # the list may be written in either spelling
    assert booru.keep("smile", 0.3, omit, 0.0)                   # no floor
    assert not booru.keep("smile", 0.3, omit, 0.4)
    assert booru.keep("smile", None, omit, 0.4)                  # no published precision: never dropped by the floor
    assert booru.keep("denim_jacket", 0.2, omit, 0.4)            # clothing is exempt from the floor
    assert not booru.keep("denim_jacket", 0.2, omit, 0.4, keep_clothes=False)


def test_kind_and_tag_name():
    assert booru.kind_of("animetimm/eva_giant_patch14_560.dbv4-full") == "animetimm"
    assert booru.kind_of("SmilingWolf/wd-eva02-large-tagger-v3") == "wd"
    assert booru.tag_name("looking_at_viewer") == "looking at viewer"
    assert booru.tag_name("^_^") == "^_^"
