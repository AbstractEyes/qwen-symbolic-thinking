"""The general rows' pure parts (tools.anchors): the request bank, the picture pick, the user-turn opening cut from a chat template,
the request filter, the cap drop and the item format. No model, no network."""
import pytest

from qwen_symbolic_thinking.tools import anchors


def fake_template(messages):
    """A Qwen-style template: each turn is <|im_start|>role\\ncontent<|im_end|>\\n; the content may be a list of text parts."""
    out = ""
    for m in messages:
        c = m["content"]
        c = c if isinstance(c, str) else "".join(p["text"] for p in c)
        out += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
    return out


def test_the_bank_has_22_plain_requests_and_none_asks_for_the_row_form():
    assert len(anchors.BANK) == 22 and len(set(anchors.BANK)) == 22
    for ask in anchors.BANK + (anchors.DETAIL,):
        assert not any(w in ask.lower() for w in ("step", "thought", "question:", "tags", "row"))
    assert anchors.DETAIL.startswith("Describe this picture in detail")


def test_draw_requests_use_every_request_about_equally_and_follow_the_seed():
    drawn = anchors.draw_requests(44, 7)
    assert sorted(set(drawn)) == sorted(anchors.BANK) and all(drawn.count(a) == 2 for a in anchors.BANK)
    assert drawn == anchors.draw_requests(44, 7) and drawn != anchors.draw_requests(44, 8)
    assert len(anchors.draw_requests(5, 1)) == 5


def pics():
    out = [{"kind": "image", "id": f"c{i}", "seed_id": f"coco:{i}", "image_urls": ["u"], "source": "coco"} for i in range(6)]
    out += [{"kind": "image", "id": f"d{i}", "seed_id": f"docci:{i}", "image": {"kind": "url_archive"}, "source": "docci"} for i in range(2)]
    out += [{"kind": "image", "id": "c0b", "seed_id": "coco:0", "image_urls": ["u"], "source": "coco"},     # the same picture again
            {"kind": "image", "id": "bare", "seed_id": "coco:99", "source": "coco"},                          # no locator
            {"kind": "spec", "id": "s", "seed_id": None}]
    return out


def test_pick_pictures_goes_round_robin_over_sources_once_per_picture():
    got = anchors.pick_pictures(pics(), 6, seed=3)
    assert [anchors.source_of(p) for p in got] == ["coco", "docci", "coco", "docci", "coco", "coco"]
    assert len({p["seed_id"] for p in got}) == 6
    assert len(anchors.pick_pictures(pics(), 100, seed=3)) == 8                  # 6 coco + 2 docci: no duplicates, no bare, no spec
    assert [p["id"] for p in anchors.pick_pictures(pics(), 6, seed=3)] == [p["id"] for p in got]
    held = {"coco:1", "coco:2"}
    assert not held & {p["seed_id"] for p in anchors.pick_pictures(pics(), 100, seed=3, exclude=held)}


def test_user_turn_is_cut_around_the_user_text():
    opening, closing = anchors.user_turn(fake_template)
    assert opening == "<|im_start|>user\n" and closing == "<|im_end|>\n"

    def strings_only(messages):                      # a template that refuses a list of content parts
        if not isinstance(messages[0]["content"], str):
            raise TypeError("content must be a string")
        return fake_template(messages)
    assert anchors.user_turn(strings_only) == (opening, closing)
    with pytest.raises(ValueError):
        anchors.user_turn(lambda messages: "no text here")


def test_cut_request_ends_at_the_templates_closing():
    assert anchors.cut_request("  What is a prime?<|im_end|>\n<|endoftext|><|endoftext|>", "<|im_end|>\n", 9, 160) == "What is a prime?"
    assert anchors.cut_request("never ends", "<|im_end|>\n", 160, 160) is None
    assert anchors.cut_request("plain", "", 5, 160) == "plain" and anchors.cut_request("plain", "", 160, 160) is None


def test_keep_requests_drops_by_reason_and_remembers_repeats_case_folded():
    seen = set()
    texts = ["Write a haiku about autumn rain.", "WRITE A HAIKU ABOUT AUTUMN RAIN.", "short", "x" * 501, None,
             "What is <|im_start|> in a chat prompt?", "Explain how a bicycle stays upright."]
    kept, drops = anchors.keep_requests(texts, seen)
    assert kept == ["Write a haiku about autumn rain.", "Explain how a bicycle stays upright."]
    assert drops == {"repeat": 1, "length": 2, "unfinished": 1, "markup": 1}
    again, d2 = anchors.keep_requests(["write a haiku about autumn rain."], seen)
    assert again == [] and d2 == {"repeat": 1}
    assert anchors.keep_requests(["x" * 20, "y" * 500], set())[0] == ["x" * 20, "y" * 500]          # the bounds are in


def test_an_answer_that_reaches_its_cap_never_ended():
    assert anchors.reached_cap(512, 512) and anchors.reached_cap(600, 512) and not anchors.reached_cap(511, 512)


def test_item_format_and_locator_copy():
    src = {"id": "x", "image_urls": ["u1", "u2"], "image": None, "seed_id": "coco:5", "source": "coco"}
    it = anchors.make_item(3, "g1", "Qwen/Qwen3.5-4B", "Give this picture a title.", "Sunday Market", "train", "picture-train",
                           "coco", "coco:5", src)
    assert it == {"kind": "anchor", "id": "anc-g1-3", "prompt": "Give this picture a title.", "answer": "Sunday Market",
                  "writer": "Qwen/Qwen3.5-4B", "split": "train", "category": "picture-train", "source": "coco", "seed_id": "coco:5",
                  "image_urls": ["u1", "u2"]}
    arch = anchors.make_item(4, "g1", "m", "p", "a", "held", "picture-held", "docci", "docci:1", {"image": {"kind": "url_archive"}})
    assert arch["image"] == {"kind": "url_archive"} and "image_urls" not in arch
    text = anchors.make_item(5, "g1", "m", "p", "a", "held", "text-held")
    assert "image" not in text and text["source"] is None and text["seed_id"] is None
    assert set(anchors.CATEGORIES) == {"picture-train", "picture-held", "picture-held-detail", "text-train", "text-held"}


def test_run_needs_its_input_file():
    with pytest.raises(ValueError):
        anchors.run()
