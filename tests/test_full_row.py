from qwen_symbolic_thinking import chain, prompts
from qwen_symbolic_thinking.data import item
from qwen_symbolic_thinking.synth import audit
from qwen_symbolic_thinking.train import conversation, examples_of

DEEP = {"id": "art-000002", "depth": "deep", "domain": "illustration", "genre": "portrait",
        "spec": {"person": {"count": "1", "gender": "woman", "age": "20s"}, "scene": {"setting": "garden"}},
        "thoughts": [{"step": s, "text": t} for s, t in (
            ("see", "One woman in a garden."), ("place", "A walled garden, roses on the left."),
            ("light", "Soft overcast light, no hard shadows."), ("colour", "Greens and one red coat."),
            ("mood", "Quiet, likely a pause in a walk."), ("intend", "A portrait made where she lives."))],
        "wordings": {"caption": "A woman in a red coat in a walled garden.",
                     "prompt": "portrait, woman in her twenties, red coat, walled garden, roses, overcast light",
                     "tags": "1girl, solo, red coat, garden, roses, overcast",
                     "critique": "The roses pull the eye from her face."},
        "questions": [{"key": "person age", "value": "20s", "needs": "record"},
                      {"key": "time of day", "value": "unknown", "needs": "unknown"},
                      {"key": "roses position", "value": "left", "needs": "chain"},
                      {"key": "person expression", "value": "calm", "needs": "chain"},
                      {"key": "season", "value": "summer", "needs": "chain"}],
        "hint": "A woman in a red coat stands in a garden with roses.",
        "source": {"kind": "seeded", "seed_id": "coco:12", "license": "CC BY 4.0"}}


def test_full_render_round_trip_and_score():
    text = chain.render(DEEP, full=True)
    lines = text.splitlines()
    assert lines[6].startswith("caption: ") and lines[7].startswith("prompt: ") and lines[8].startswith("tags: ")
    assert lines[9].startswith("critique: ") and lines[10] == "question: person age -> 20s [record]"
    parsed = chain.parse(text)
    assert parsed["thoughts"] == DEEP["thoughts"] and parsed["wordings"] == DEEP["wordings"]
    assert parsed["questions"] == DEEP["questions"] and parsed["stray"] == []
    s = chain.score(parsed, DEEP)
    assert s["format_ok"] and s["in_range"] and s["wordings_ok"] and s["questions_in_range"] and s["full_ok"]
    assert s["questions_n"] == 5 and s["questions_formed"] == 5 and s["wordings_n"] == 4
    assert s["hint_said"] > 0.5 and s["hint_backed"] > 0.2


def test_short_answer_is_not_a_full_row():
    s = chain.score(chain.parse(chain.render(DEEP)), DEEP)
    assert s["format_ok"] and not s["wordings_ok"] and s["questions_n"] == 0 and not s["full_ok"]


def test_malformed_question_is_kept_as_text():
    parsed = chain.parse("see: A dog.\ncaption: A dog.\nquestion: what breed is it?")
    assert parsed["questions"] == [{"text": "what breed is it?"}]
    s = chain.score(parsed, dict(DEEP, depth="simple"))
    assert s["questions_n"] == 1 and s["questions_formed"] == 0 and s["questions_in_range"] and not s["full_ok"]


def test_grounding_reads_the_thoughts_and_the_caption_only():
    parsed = chain.parse("see: A woman in a red coat.\ncaption: A woman in a garden.\ntags: xylophone, quasar, nebula")
    s = chain.score(parsed, DEEP)
    assert s["hint_backed"] == 1.0          # every content word of the chain and the caption is in the hint; the tags are not read
    assert 0.5 < s["hint_said"] < 1.0


def test_full_prompt_forms():
    full = prompts.full("deep")
    assert "a long look" in full and "a caption, a prompt, tags and a critique" in full and "question: what -> answer" in full
    assert prompts.full("simple").count("caption") == 1
    spec_full = prompts.spec_full("medium", DEEP["spec"])
    assert "person age: 20s" in spec_full and "a caption and a prompt or tags" in spec_full


def test_item_carries_questions_and_forms():
    it = item(DEEP, hint=DEEP["hint"], forms=("full", "short"))
    assert it["kind"] == "image" and it["questions"] == DEEP["questions"] and it["forms"] == ["full", "short"]
    assert item(dict(DEEP, source={"kind": "synthetic"}))["forms"] == ["spec_short"]
    assert [f for _, f in examples_of([it])] == ["full", "short"]
    messages, answer = conversation(it, "full", image="IMG")
    assert messages[0]["content"][0] == {"type": "image", "image": "IMG"} and answer.endswith("[chain]")
    messages, answer = conversation(dict(it, kind="spec"), "spec_full")
    assert messages[0]["content"][0]["type"] == "text" and "question:" in answer


def test_synth_audit():
    good = chain.score(chain.parse("see: A woman in a red coat.\nplace: A garden with roses.\nmood: Quiet.\n"
                                   "caption: A woman in a garden."), dict(DEEP, depth="medium"))
    assert audit(good, 0.2, 0.2) is None
    bad = chain.score(chain.parse("see: A woman.\ncaption: A woman."), dict(DEEP, depth="medium"))
    assert audit(bad, 0.2, 0.2) == "depth"
