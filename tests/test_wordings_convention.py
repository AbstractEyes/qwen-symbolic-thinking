"""The wordings a whole row carries at each depth, by the version of the rows a model learned: version 1's rule counts them (a
medium row = a caption and one of a prompt or tags); the version 1.1 rows carry a caption, a prompt and tags at medium. A row is
read by the convention of the rows its model learned (chain.wordings_whole; extract.run and screen.run take it as wordings)."""
import inspect

import pytest

from qwen_symbolic_thinking import chain, extract, screen

V11_MEDIUM_ROW = """see: Several beige robots crowd the close-up, their round heads and blue eyes repeated across the frame.
form: The photograph layers the figures at varied depths; signs held by the front robots remain clear while those behind blur.
mood: The warm beige palette suggests a gentle mood, its muted tones softening the repeated metallic forms.
caption: Several beige robots with blue eyes hold signs in a close-up photograph, their layered arrangement fading toward the background.
prompt: close-up photograph of several beige robots, round heads and blue eyes, signs held at varied depths, blurred background
tags: beige robots, blue eyes, close-up, photograph
question: thing name -> robots [record]
question: thing count -> several [record]
question: shot framing -> close-up [record]
question: form medium -> photograph [record]
question: form balance -> unknown [unknown]"""


def test_version_1_counts_the_wordings_and_version_1_1_names_them():
    medium_v11 = {"caption": "c", "prompt": "p", "tags": "t"}
    medium_v1 = {"caption": "c", "tags": "t"}
    assert chain.wordings_whole(medium_v1, "medium") and chain.wordings_whole(medium_v1, "medium", "v1")
    assert not chain.wordings_whole(medium_v11, "medium")                      # three wordings: not version 1's medium row
    assert chain.wordings_whole(medium_v11, "medium", "v1.1")
    assert not chain.wordings_whole(medium_v1, "medium", "v1.1")               # version 1.1 asks for all three at medium
    assert chain.wordings_whole({"caption": "c"}, "simple", "v1.1") and chain.wordings_whole({"caption": "c"}, "simple")
    deep = {"caption": "c", "prompt": "p", "tags": "t", "critique": "k"}
    assert chain.wordings_whole(deep, "deep", "v1.1") and chain.wordings_whole(deep, "deep")
    assert not chain.wordings_whole({"prompt": "p", "tags": "t", "critique": "k"}, "medium", "v1.1")   # a caption always
    assert chain.wordings_whole({"caption": "c", "tags": "t"}, "medium", {"medium": ("caption", "tags")})   # a map of names
    with pytest.raises(ValueError, match="convention"):
        chain.wordings_whole(medium_v11, "medium", "v9")


def test_a_v11_row_is_whole_under_its_own_convention_and_broken_under_version_1s():
    parsed = chain.parse(V11_MEDIUM_ROW)
    assert sorted(parsed["wordings"]) == ["caption", "prompt", "tags"] and len(parsed["questions"]) == 5
    row = {"depth": "medium", "spec": {}, "thoughts": []}
    old, new = chain.score(parsed, row), chain.score(parsed, row, "v1.1")
    assert not old["wordings_ok"] and not old["full_ok"]                       # what extraction marked "row shape" before
    assert new["wordings_ok"] and new["full_ok"]
    assert {k: v for k, v in old.items() if k not in ("wordings_ok", "full_ok")} == \
           {k: v for k, v in new.items() if k not in ("wordings_ok", "full_ok")}   # nothing else read differently


def test_extract_and_screen_take_the_convention_off_by_default_and_refuse_an_unknown_one():
    for fn in (extract.run, screen.run, screen.rescore):
        assert inspect.signature(fn).parameters["wordings"].default is None, fn.__name__
    with pytest.raises(ValueError, match="convention"):
        extract.run("x.jsonl", wordings="v2")
    with pytest.raises(ValueError, match="convention"):
        screen.run(["Qwen/Qwen3.5-4B"], wordings="v2")
