import json
import os

import pytest

from qwen_symbolic_thinking import chain
from qwen_symbolic_thinking.tools import mix

LOOKERS = [{"key": "l9", "family": "lucia", "size": "9b", "runs": ["R_l9"], "rows": False},
           {"key": "l4", "family": "lucia", "size": "4b", "runs": ["R_l4"]},
           {"key": "l4f", "family": "lucia", "size": "4b", "runs": ["R_l4f"]},
           {"key": "u12", "family": "ulisse", "size": "12b", "runs": ["R_u12"]}]
ORDER = [lk["key"] for lk in LOOKERS]


def row(sid, caption, prompt, see, full_ok=True, said=0.2, backed=0.3, source="coco"):
    text = f"see: {see}\ncaption: {caption}\nprompt: {prompt}\nquestion: thing name -> x [record]\n"
    p = chain.parse(text)
    return {"sample_id": sid, "source": source, "group": "photo", "depth": "medium", "text": text, "thoughts": p["thoughts"],
            "wordings": p["wordings"], "questions": p["questions"], "rejected": None, "hinted": True,
            "scores": {"full_ok": full_ok, "in_range": True, "wordings_ok": True, "questions_in_range": True, "n_thoughts": 1,
                       "questions_n": 1, "gloss_hits": 0, "remark_hits": 0, "stray_lines": 0, "chars": len(see),
                       "hint_said": said, "hint_backed": backed}}


def test_word_rule_whole_words_and_case():
    out, n = mix.word_rule("A girl and two boys; a Cowboy, his girlfriend. Girl at the gate. GIRLS")
    assert out == "A young woman and two young men; a Cowboy, his girlfriend. Young woman at the gate. YOUNG WOMEN"
    assert n == 4
    assert mix.word_rule("") == ("", 0)


def test_weights_by_family_size_and_each():
    w = mix.weights(LOOKERS, "family")
    assert w == {"l9": 0.25, "l4": 0.125, "l4f": 0.125, "u12": 0.5}
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert mix.weights(LOOKERS, "size") == {"l9": 1.0, "l4": 0.5, "l4f": 0.5, "u12": 1.0}
    assert mix.weights(LOOKERS, "each") == {k: 1.0 for k in ORDER}
    with pytest.raises(ValueError):
        mix.weights(LOOKERS, "votes")


def test_agreement_pick_takes_the_consensus_and_breaks_ties_by_order():
    w = mix.weights(LOOKERS, "each")
    words = {"l9": {"red", "apple", "table"}, "l4": {"red", "apple", "wooden", "table"}, "l4f": {"red", "apple", "table"},
             "u12": {"blue", "car", "street"}}
    a = mix.agreement(words, w)
    assert a["u12"] == 0.0
    assert mix.pick(a, ORDER) == "l9"            # l9 and l4f tie at the top; the earlier looker wins
    assert mix.pick({"u12": None}, ORDER) == "u12"  # a lone row is picked
    assert mix.pick({}, ORDER) is None


def test_mix_picture_fields_pools_and_vote():
    rows = {"l9": row("p", "A red apple on a table.", "photograph, red apple, table", "A red apple on a wooden table."),
            "l4": row("p", "A red apple on a wooden table.", "photograph, red apple, wooden table", "One red apple on a table."),
            "l4f": row("p", "A girl holds a red apple.", "photograph, girl, apple", "A girl with an apple.", full_ok=False),
            "u12": row("p", "A blue car in the street.", "oil painting, blue car, street", "A blue car on a street.")}
    m = mix.mix_picture(rows, mix.weights(LOOKERS, "family"), ORDER)
    assert m["n_rows"] == 4 and m["n_form"] == 3           # the row without the form is no candidate
    assert m["pick"] in ("l9", "l4") and m["pick_score"] > m["agreement"]["u12"]
    assert m["caption_pick_looker"] in ("l9", "l4") and m["prompt_pick_looker"] in ("l9", "l4")
    # all four first terms vote; the Lucia family's half says photograph, the Ulisse family's half says painting: a tie at 0.5,
    # broken by the looker order
    assert m["kind_vote"] == "photograph" and m["kind_share"] == 0.5
    assert m["captions"] == ["A red apple on a table.", "A red apple on a wooden table.", "A blue car in the street."]
    assert m["words_replaced"] == 0                         # the girl row held no form, so its words are in no pool


def test_mix_picture_rules_generator_words_in_pools():
    rows = {"l9": row("p", "A girl reads.", "photograph, girl reading", "A girl reading."),
            "u12": row("p", "Two boys play.", "photograph, two boys", "Two boys playing.")}
    m = mix.mix_picture(rows, mix.weights(LOOKERS, "family"), ORDER)
    assert "A young woman reads." in m["captions"] and "Two young men play." in m["captions"]
    assert all("girl" not in p and "boys" not in p for p in m["prompts"])
    assert m["words_replaced"] >= 4


def test_mix_picture_single_row():
    m = mix.mix_picture({"u12": row("p", "A cat.", "photograph, cat", "A cat on a mat.")}, mix.weights(LOOKERS, "family"), ORDER)
    assert m["pick"] == "u12" and m["pick_score"] is None and m["captions"] == ["A cat."]


def test_read_accumulates_picks_random_oracle_and_probe():
    w = mix.weights(LOOKERS, "family")
    rd = mix.Read(ORDER)
    rows = {"l9": row("p", "A red apple on a table.", "photograph, red apple", "A red apple on a wooden table.", said=0.4, backed=0.6),
            "u12": row("p", "A blue car.", "oil painting, blue car", "A blue car.", said=0.0, backed=0.0)}
    m = mix.mix_picture(rows, w, ORDER)
    rd.add("coco", "a red apple sits on a wooden table", rows, m, None)
    probe_rows = {"l9": row("q", "A drawing of a man.", "drawing, a man", "A man.", source="nga"),
                  "u12": row("q", "A print of a man.", "print, a man", "A man.", source="nga")}
    pm = mix.mix_picture(probe_rows, w, ORDER)
    rd.add("nga", "", probe_rows, pm, "drawing")
    res = rd.result()
    v = res["volume"]["coco"]
    assert v["l9"]["backed"] == 0.6 and v["u12"]["backed"] == 0.0
    assert v["random looker"]["backed"] == pytest.approx(0.3) and v["oracle (best-backed row)"]["backed"] == 0.6
    assert v["agreement pick"]["n"] == 1
    assert res["volume"]["all"]["l9"]["n"] == 1             # the uncaptioned probe picture is in no volume read
    assert res["probe"]["l9"]["drawing"] == 1 and res["probe"]["u12"]["drawing"] == 0 and res["probe_n"] == {"drawing": 1}
    assert "### Said / backed" in mix.read_markdown(res, ORDER)


def test_mean_se():
    assert mix.mean_se([]) == (0, None, None)
    n, m, se = mix.mean_se([0.0, 1.0, None])
    assert n == 2 and m == 0.5 and se == pytest.approx(0.5)


def test_rows_record_and_table_schema():
    pa = pytest.importorskip("pyarrow")
    rec = mix.rows_record(row("p", "A cat.", "photograph, cat", "A cat."), "20261101_000000_XU_u12b_A")
    assert list(rec) == mix.ROWS_COLUMNS and rec["training_item"] is False and rec["full_ok"] == 1
    t = mix.rows_table([rec])
    assert t.schema.names == mix.ROWS_COLUMNS
    assert t.schema.field("full_ok").type == pa.int64() and t.schema.field("hint_backed").type == pa.float64()
    assert t.schema.field("hinted").type == pa.bool_() and t.schema.field("rejected").type == pa.string()


def test_run_offline_end_to_end(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    root = tmp_path / "data"
    (root / "runs" / "done").mkdir(parents=True)
    pics = [("a", "coco", "a red apple on a wooden table", None), ("b", "nga", "", "Drawings")]
    texts = {"l9": [("A red apple on a table.", "photograph, red apple, table", "A red apple on a wooden table."),
                    ("A man.", "drawing, a man", "A man drawn in ink.")],
             "l4": [("A red apple on a wooden table.", "photograph, red apple", "One red apple."), ("A man.", "drawing, man", "A man.")],
             "l4f": [("An apple.", "photograph, apple", "An apple."), ("A man.", "print, a man", "A man.")],
             "u12": [("A blue car.", "oil painting, blue car", "A car."), ("A man.", "drawing, a man", "A man.")]}
    for k, rr in texts.items():
        folder = f"20261101_000000_R_{k}"
        d = root / "extractions" / folder
        d.mkdir(parents=True)
        with open(d / "rows.jsonl", "w", encoding="utf-8") as fh:
            for (sid, src, _, _), (cap, pr, see) in zip(pics, rr):
                fh.write(json.dumps(row(sid, cap, pr, see, source=src)) + "\n")
        (root / "runs" / "done" / f"R_{k}.json").write_text(json.dumps(
            {"status": "done", "extract": {"path_in_repo": f"extractions/{folder}"}}), encoding="utf-8")
    man = tmp_path / "manifest.jsonl"
    with open(man, "w", encoding="utf-8") as fh:
        for sid, src, cap, genre in pics:
            fh.write(json.dumps({"sample_id": sid, "source": src, "split": "probe" if genre else "main",
                                 "captions": {"text": cap} if cap else {}, "labels": {"genre": genre} if genre else {}}) + "\n")
    out = tmp_path / "out"
    s = mix.run(LOOKERS, str(man), data=str(root), out_dir=str(out), offline=True, ship=False, tag="test")
    assert s["pictures"] == 2 and s["agreement_rows"] == 2
    assert "rows/l9/extracted_rows.parquet" not in s["files"] and "rows/u12/extracted_rows.parquet" in s["files"]
    mixed = pq.read_table(out / "mixed.parquet").to_pylist()
    assert [m["sample_id"] for m in mixed] == ["a", "b"] and mixed[0]["kind_vote"] == "photograph"
    reads = json.load(open(out / "reads.json", encoding="utf-8"))
    assert reads["probe"]["kind vote"]["drawing"] == 1 and reads["volume"]["all"]["l9"]["n"] == 1
    assert pq.read_table(out / "rows_u12.parquet").schema.names == mix.ROWS_COLUMNS
    assert os.path.exists(out / "reads.md")
