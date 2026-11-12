"""The read of how far an adapter moved the model (tools.keep_read): the pure parts, and the scale helpers on stand-in layers.
No model, no network."""
import math

import pytest

from qwen_symbolic_thinking.tools import keep_read as kr


def test_percentile_interpolates_between_ranks():
    xs = [4.0, 1.0, 3.0, 2.0]
    assert kr.percentile(xs, 0) == 1.0 and kr.percentile(xs, 100) == 4.0
    assert kr.percentile(xs, 50) == pytest.approx(2.5) and kr.percentile(xs, 90) == pytest.approx(3.7)
    assert kr.percentile([5.0], 99) == 5.0 and kr.percentile([], 50) is None
    assert kr.percentile(list(range(101)), 99) == pytest.approx(99.0)


def test_takes_row_form_by_labelled_lines_or_thoughts():
    assert kr.takes_row_form("see: A dog on grass.\ncount: one dog\ncaption: A dog on grass.")    # two named thoughts
    assert kr.takes_row_form("see: A dog on grass.\ncaption: A dog.")                             # a thought and a wording
    assert kr.takes_row_form("see: a dog\nquestion: how many dogs -> one [record]")              # a thought and a question
    assert kr.takes_row_form("Step: look\nStep: count\nQuestion: what")                          # the form's generic labels
    assert not kr.takes_row_form("A dog on grass.\nIt is sunny.")
    assert not kr.takes_row_form("caption: only one label line")
    assert not kr.takes_row_form("Caption: Golden hour at the pier\nTags: #sunset #pier")       # a social media caption
    assert not kr.takes_row_form("Title: The Fox\nStory: Once upon a time a fox lived by a river.")   # one step word in a story
    assert not kr.takes_row_form("Mood: calm and quiet")
    assert not kr.takes_row_form("")


def test_token_f1_counts_words_as_a_bag():
    assert kr.token_f1("The cat sat", "the CAT sat") == 1.0
    assert kr.token_f1("", "") == 1.0 and kr.token_f1("a", "") == 0.0 and kr.token_f1("cat", "dog") == 0.0
    assert kr.token_f1("a b c d", "a b") == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert kr.token_f1("a a a", "a") == pytest.approx(2 * (1 / 3) * 1.0 / (1 / 3 + 1.0))
    assert kr.word_tokens("Où est le café?") == ["où", "est", "le", "café"]


def test_content_words_and_recall():
    assert 120 <= len(kr.STOPWORDS) <= 200
    assert kr.content_words("The red cars and a dog, by the 2 big hats!") == {"red", "car", "dog", "big", "hat"}
    assert kr.content_words("It is on the of") == set()
    assert kr.recall("A red car and two hats", "red cars, a hat") == pytest.approx(1.0)
    assert kr.recall("red car blue hat", "a red car") == pytest.approx(0.5)
    assert kr.recall("the and of", "anything") is None


def rec(category, kl, nb, nt, first):
    return {"category": category, "kl": kl, "nll_base": nb, "nll_tuned": nt, "kl_first": first}


def test_drift_summary_by_category_and_overall():
    records = [rec("text-held", 0.1, 1.0, 1.2, [0.2] * 8), rec("text-held", 0.3, 1.0, 1.4, [0.4] * 3),
               rec("picture-held", 0.5, 0.5, 0.5, [])]
    out = kr.drift_summary(records)
    assert set(out) == {"all", "text-held", "picture-held"}
    t = out["text-held"]
    assert t["rows"] == 2 and t["kl_mean"] == pytest.approx(0.2) and t["kl_p50"] == pytest.approx(0.2)
    assert t["kl_p90"] == pytest.approx(0.28) and len(t["kl_by_position"]) == 8
    assert t["kl_by_position"][0] == pytest.approx(0.3) and t["kl_by_position"][3] == pytest.approx(0.2)
    assert t["perplexity_ratio"] == pytest.approx(math.exp(1.3 - 1.0))
    assert out["picture-held"]["kl_by_position"] == [None] * 8
    assert out["all"]["rows"] == 3 and out["all"]["kl_mean"] == pytest.approx(0.3)


def test_takeover_and_detail_summaries():
    take = [{"category": "text-held", "takes_row_form": True, "ended": True, "length_ratio": 2.0, "f1": 0.5},
            {"category": "text-held", "takes_row_form": False, "ended": False, "length_ratio": 1.0, "f1": 1.0},
            {"category": "picture-held", "takes_row_form": False, "ended": True, "length_ratio": None, "f1": 0.0}]
    out = kr.takeover_summary(take)
    assert out["text-held"] == {"rows": 2, "takes_row_form": 0.5, "ended": 0.5, "length_ratio": 1.5, "length_median": 1.5,
                                "length_total": None, "f1": 0.75}
    assert out["all"]["length_ratio"] == 1.5 and out["picture-held"]["length_ratio"] is None
    d = kr.detail_summary([{"recall": 0.5, "row_chars": 100, "row_tokens": 30, "ended": True},
                           {"recall": None, "row_chars": 300, "row_tokens": 50, "ended": False}])
    assert d == {"rows": 2, "recall": 0.5, "row_chars": 200.0, "row_tokens": 40.0, "ended": 0.5}


def test_the_length_read_by_the_median_and_the_total_is_not_pulled_by_a_few_word_base_answer():
    # three answers as long as the base's and one where the base wrote 4 characters and the model 200
    take = [{"category": "text-held", "takes_row_form": False, "ended": True, "length_ratio": a / b, "chars": a, "base_chars": b,
             "f1": 1.0} for a, b in ((300, 300), (310, 300), (290, 300), (200, 4))]
    out = kr.takeover_summary(take)["text-held"]
    assert out["length_ratio"] == pytest.approx((1 + 310 / 300 + 290 / 300 + 50) / 4)     # the mean of ratios: 13.25
    assert out["length_median"] == pytest.approx((1 + 310 / 300) / 2)                      # the middle two of four
    assert out["length_total"] == pytest.approx(1100 / 904)
    assert kr.median([None, 3, 1, 2]) == 2 and kr.median([]) is None


class Layer:
    def __init__(self, scaling):
        self.scaling = scaling


class Net:
    def __init__(self, layers):
        self.layers = layers

    def modules(self):
        return [self, *self.layers]


def test_set_scale_multiplies_the_trained_value_and_resets():
    a, b = Layer({"default": 2.0}), Layer({"default": 0.5})
    net = Net([a, b])
    net.scaling = None                                                  # a module that is no LoRA layer
    layers = kr.lora_layers(net)
    assert len(layers) == 2
    kr.set_scale(layers, 0.5)
    assert (a.scaling["default"], b.scaling["default"]) == (1.0, 0.25)
    kr.set_scale(layers, 0.0)
    assert a.scaling["default"] == 0.0
    kr.set_scale(layers, 1.0)                                           # back to the trained values, not a product of the earlier scales
    assert (a.scaling["default"], b.scaling["default"]) == (2.0, 0.5)
    with pytest.raises(ValueError):
        kr.lora_layers(Net([]))


def test_kl_per_position_agrees_with_keep_kl_full():
    torch = pytest.importorskip("torch")
    from qwen_symbolic_thinking import keep
    g = torch.Generator().manual_seed(0)
    zb, zt = torch.randn(5, 11, generator=g), torch.randn(5, 11, generator=g)
    per = kr.kl_per_position(zb, zt)
    assert per.shape == (5,) and float(per.mean()) == pytest.approx(float(keep.kl_full(zb, zt)), rel=1e-5)
    assert float(kr.kl_per_position(zb, zb).abs().max()) == pytest.approx(0.0, abs=1e-6)
