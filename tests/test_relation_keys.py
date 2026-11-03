"""The relation keys: a question's key written one word a key (thing_position) and a relation's things lettered relation_a,
relation_b, ... (its word relation_a_b); the specification's records written with the same keys, so the prompt and the answer
name a record alike; the dataset's own record keys stay the default, unchanged."""
import pytest

from qwen_symbolic_thinking import chain, prompts, queue, screen, train

STAR = {"id": "art-900001", "kind": "spec", "depth": "deep", "domain": "photograph", "genre": "sport",
        "spec": {"person": {"count": "1", "gender": "man"}, "thing": {"name": "tennis racket"},
                 "relation": [{"a": "man", "rel": "holding", "b": "tennis racket"},
                              {"a": "man", "rel": "in front of", "b": "fence"}],
                 "mood": {"feeling": "calm", "from": "look expression"}},
        "thoughts": [{"step": "see", "text": "One man holding a tennis racket."},
                     {"step": "relate", "text": "A fence stands behind him."}],
        "wordings": {"caption": "A man with a tennis racket in front of a fence."},
        "questions": [{"key": "person gender", "value": "man", "needs": "record"},
                      {"key": "rel1 b", "value": "tennis racket", "needs": "record"},
                      {"key": "rel2 b", "value": "fence", "needs": "record"},
                      {"key": "rel2 rel", "value": "in front of", "needs": "record"},
                      {"key": "rel1 a", "value": "man", "needs": "record"},
                      {"key": "scene time", "value": "unknown", "needs": "unknown"}]}
CHAIN = [{"a": "cup", "rel": "on", "b": "table"}, {"a": "table", "rel": "beside", "b": "window"}]
MUSEUM = {"id": "qti-900002", "kind": "image", "depth": "medium", "spec": {"form": {"medium": "oil on canvas"}},
          "thoughts": [{"step": "see", "text": "A vase on a stand."}], "wordings": {"caption": "A vase on a stand."},
          "questions": [{"key": "rel1 b", "value": "stand", "needs": "record"}, {"key": "rel2 b", "value": "table", "needs": "record"},
                        {"key": "rel1 rel", "value": "on", "needs": "chain"}, {"key": "form  layout", "value": "centred", "needs": "chain"}]}


def test_things_are_lettered_in_order_and_keep_their_letter():
    assert chain.relation_letters(STAR["spec"]["relation"]) == [("a", "b"), ("a", "c")]       # the man recurs: a star
    assert chain.relation_letters(CHAIN) == [("a", "b"), ("b", "c")]                           # the table recurs: a chain
    assert chain.relation_letters(CHAIN + [{"a": "dog", "rel": "beside", "b": "cup"}]) == [("a", "b"), ("b", "c"), ("d", "a")]
    assert chain.relation_letters([{"a": "cat", "rel": "facing", "b": "cat"}]) == [("a", "a")]
    assert chain.relation_letters([]) == []


def test_relation_key_names():
    star = chain.relation_letters(STAR["spec"]["relation"])
    assert [chain.relation_key(k, star) for k in ("rel1 a", "rel1 b", "rel1 rel", "rel2 a", "rel2 b", "rel2 rel")] == [
        "relation_a", "relation_b", "relation_a_b", "relation_a", "relation_c", "relation_a_c"]
    links = chain.relation_letters(CHAIN)
    assert [chain.relation_key(k, links) for k in ("rel2 a", "rel2 b", "rel2 rel")] == ["relation_b", "relation_c", "relation_b_c"]
    # a row without relation records: relation i joins the i-th and (i+1)-th things
    assert [chain.relation_key(k) for k in ("rel1 b", "rel2 b", "rel3 b", "rel1 rel", "rel2 a")] == [
        "relation_b", "relation_c", "relation_d", "relation_a_b", "relation_b"]
    assert chain.relation_key("thing position") == "thing_position" and chain.relation_key("form  layout ") == "form_layout"
    assert chain.relation_key("relation_b") == "relation_b" and chain.relation_key("season") == "season"


def test_the_record_keys_stay_the_default():
    full = chain.render(STAR, full=True)
    assert full == chain.render(STAR, full=True, keys="record")
    assert "question: rel1 b -> tennis racket [record]" in full and "question: person gender -> man [record]" in full
    assert prompts.records(STAR["spec"]) == ("person count: 1\nperson gender: man\nthing name: tennis racket\n"
                                             "relation: man holding tennis racket\nrelation: man in front of fence\nmood feeling: calm")
    assert prompts.spec_full("deep", STAR["spec"]) == prompts.spec_full("deep", STAR["spec"], "record")


def test_relation_keys_in_the_answer():
    lines = [l for l in chain.render(STAR, full=True, keys="relation").splitlines() if l.startswith("question:")]
    assert lines == ["question: person_gender -> man [record]", "question: relation_b -> tennis racket [record]",
                     "question: relation_c -> fence [record]", "question: relation_a_c -> in front of [record]",
                     "question: relation_a -> man [record]", "question: scene_time -> unknown [unknown]"]
    parsed = chain.parse(chain.render(STAR, full=True, keys="relation"))
    assert [q["key"] for q in parsed["questions"]][:3] == ["person_gender", "relation_b", "relation_c"]
    museum = [l for l in chain.render(MUSEUM, full=True, keys="relation").splitlines() if l.startswith("question:")]
    assert museum == ["question: relation_b -> stand [record]", "question: relation_c -> table [record]",
                      "question: relation_a_b -> on [chain]", "question: form_layout -> centred [chain]"]
    assert chain.render(STAR, keys="relation") == chain.render(STAR)          # the short form carries no keys


def test_the_records_line_up_with_the_questions():
    records = prompts.records(STAR["spec"], "relation")
    assert records.splitlines() == ["- person_count: 1", "- person_gender: man", "- thing_name: tennis racket",
                                    "- relation_a: man", "- relation_a_b: holding", "- relation_b: tennis racket",
                                    "- relation_a_c: in front of", "- relation_c: fence", "- mood_feeling: calm"]
    for q in chain.parse(chain.render(STAR, full=True, keys="relation"))["questions"]:
        if q["needs"] == "record":                     # every record a question settles is listed under the same key and value
            assert f"- {q['key']}: {q['value']}" in records.splitlines()
    assert prompts.records({"relation": CHAIN}, "relation").splitlines() == [
        "- relation_a: cup", "- relation_a_b: on", "- relation_b: table", "- relation_b_c: beside", "- relation_c: window"]
    assert prompts.records({"relation": [{"a": "cat", "rel": "facing", "b": "cat"}]}, "relation").splitlines() == [
        "- relation_a: cat", "- relation_a_a: facing"]


def test_unknown_keys_are_refused():
    with pytest.raises(ValueError):
        chain.render(STAR, full=True, keys="relative")
    with pytest.raises(ValueError):
        prompts.records(STAR["spec"], "relative")
    with pytest.raises(ValueError):
        train.run(keys="relative")                   # refused before anything loads
    with pytest.raises(ValueError):
        screen.run(models=["x"], keys="relative")


def test_the_conversation_carries_the_keys_on_both_sides():
    messages, answer = train.conversation(STAR, "spec_full", keys="relation")
    user = messages[0]["content"][0]["text"]
    assert "- relation_c: fence" in user and "question: relation_c -> fence [record]" in answer
    assert "relation: man in front of fence" not in user and "rel2 b" not in answer
    assert train.conversation(STAR, "spec_full") == train.conversation(STAR, "spec_full", keys="record")
    shown = screen.messages("spec_full", STAR, {}, None, keys="relation")
    assert shown[0]["content"] == messages[0]["content"]
    picture = dict(MUSEUM)
    assert train.conversation(picture, "full", image="IMG", keys="relation")[0] == train.conversation(picture, "full", image="IMG")[0]


def test_a_train_spec_screens_with_its_keys(monkeypatch):
    seen = {}
    monkeypatch.setattr(train, "run", lambda **kw: {"adapter_dir": "/tmp/a", "path_in_repo": "lora/x", "full": False})
    monkeypatch.setattr(screen, "run", lambda **kw: seen.update(kw) or {"ok": True})
    monkeypatch.setattr(queue, "_mark", lambda *a, **k: None)
    spec = {"name": "K_rel_test", "kind": "train", "train": {"model_id": "Qwen/Qwen3.5-2B", "keys": "relation", "max_steps": 500}}
    result = queue.run_spec(spec, "lane1", "owner/repo", "/tmp")
    assert result["status"] == "done" and seen["keys"] == "relation" and seen["models"] == ["Qwen/Qwen3.5-2B"]
    seen.clear()
    queue.run_spec({**spec, "train": {"model_id": "Qwen/Qwen3.5-2B"}}, "lane1", "owner/repo", "/tmp")
    assert "keys" not in seen                          # an arm on the record keys is screened as before
