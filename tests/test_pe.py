"""The prompt-enhancer read: the answer split, the step map, the rulers on free text, the conversations (no model, no network)."""
from qwen_symbolic_thinking import pe
from qwen_symbolic_thinking.chain import STEPS

GOOD = ("<think>I see a giraffe lying on dry ground. It is in the centre of the frame. Warm sunlight falls across the dirt. "
        "Keep everything fixed. The prompt should describe it.</think>\n"
        '{"rewritten_prompt": "A giraffe lies on dry dirt at the centre of the frame, lit by warm sunlight.", "ratio_follow": "<image1>"}')


def test_split_answer_reads_thinking_and_json():
    a = pe.split_answer(GOOD)
    assert a["thinking"].startswith("I see a giraffe") and "</think>" not in a["thinking"]
    assert a["json_ok"] and a["prompt"].startswith("A giraffe lies") and a["ratio"] == "<image1>" and a["ratio_key"] == "ratio_follow"
    assert not a["cut"]


def test_split_answer_marks_cut_and_broken_json():
    cut = pe.split_answer("<think>Still thinking about the picture")
    assert cut["cut"] and cut["prompt"] is None and not cut["json_ok"] and cut["thinking"].startswith("Still")
    broken = pe.split_answer("<think>Done.</think>\n{\"rewritten_prompt\": \"unterminated")
    assert not broken["cut"] and not broken["json_ok"] and broken["prompt"] is None
    ratio = pe.split_answer("<think>x</think>{\"rewritten_prompt\": \"A corgi in the rain.\", \"wh_ratio\": \"16:9\"}")
    assert ratio["json_ok"] and ratio["ratio"] == "16:9" and ratio["ratio_key"] == "wh_ratio"


def test_step_map_follows_the_chain_order():
    assert tuple(pe.STEP_WORDS) == tuple(STEPS)
    m = pe.step_map(pe.split_answer(GOOD)["thinking"])
    assert m["sentences"] == 5
    assert m["per_sentence"][0] == "see" and m["per_sentence"][1] == "place" and m["per_sentence"][2] == "light"
    assert m["per_sentence"][3] == "change" and m["per_sentence"][4] == "say"
    assert m["order"] == ["see", "place", "light", "change", "say"] and abs(m["coverage"] - 5 / 16) < 1e-9
    empty = pe.step_map("")
    assert empty["sentences"] == 0 and empty["order"] == [] and empty["coverage"] == 0
    whole = pe.step_map("The stone is done. A lighter tone. Money lies there.")      # one, lit and one inside other words count for nothing
    assert whole["per_sentence"] == [None, "light", None]


def test_grounding_and_score_on_free_text():
    hint = "A giraffe sitting on the ground in the dirt. A giraffe is lying down on the ground."
    g = pe.grounding("The giraffe lies on the dirt ground.", hint)
    assert 0 < g["said"] <= 1 and 0 < g["backed"] <= 1 and g["words"] >= 3
    s = pe.score(pe.split_answer(GOOD), hint)
    assert s["json_ok"] and not s["cut"] and s["think_sentences"] == 5 and s["prompt_sentences"] == 1
    assert s["think_said"] is not None and s["prompt_backed"] is not None and s["steps"][0] == "see"
    none = pe.score(pe.split_answer("<think>a</think>{}"), "")
    assert none["think_said"] is None and none["prompt_said"] is None and not none["json_ok"]


def test_conversation_and_seed_prompt():
    c = pe.conversation("SYSTEM", pe.NO_CHANGE_REQUEST, image="IMG")
    assert c[0] == {"role": "system", "content": "SYSTEM"} and c[1]["role"] == "user"
    assert c[1]["content"][0] == {"type": "image", "image": "IMG"} and c[1]["content"][1]["text"] == pe.NO_CHANGE_REQUEST
    t = pe.conversation("SYSTEM", "a corgi in the rain")
    assert len(t[1]["content"]) == 1 and t[1]["content"][0]["type"] == "text"
    assert pe.seed_prompt({"id": "x", "hint": "A giraffe sitting on the ground. A giraffe is lying down."}) == "A giraffe sitting on the ground."
    assert pe.seed_prompt({"id": "x", "hint": ""}) == "x"
    assert pe.MODELS["i2i"].endswith("PE-I2I") and pe.MODELS["t2i"].endswith("PE-T2I") and pe.SAMPLING["top_k"] == 20
