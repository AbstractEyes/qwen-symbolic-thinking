"""The record-only reads (tools.text_names, keep_text_names, lettering_ask, quote_precision, bars) on small records in a local
folder, every expected number worked out by hand; the command's read sets (reads.py) against the sets read by hand for the 2B,
9B and 0.8B v1.2 lines; the command end to end on local records, offline."""
import json
import os
import random

import pytest

from qwen_symbolic_thinking import lines, reads, registry
from qwen_symbolic_thinking.tools import bars, keep_text_names, lettering_ask, quote_precision, text_names
from qwen_symbolic_thinking.tools import records as rec

PROJECT = "someone/their-project"
SENTENCE = lines.LETTERING


def _put(root, path, data):
    p = os.path.join(root, *path.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        if path.endswith(".jsonl"):
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
        else:
            json.dump(data, fh)
    return p


def _extraction(root, name, rows, adapter=None, subfolder=None, manifest="repo:text/text_held_v1.jsonl", ask_suffix=None,
                model="Qwen/Qwen3.5-2B", status="done"):
    path = f"extractions/20261120_000000_{name}"
    ex = {"model": model, "adapter": adapter, "adapter_subfolder": subfolder, "adapter_scale": 1.0, "manifest": manifest,
          "ask_suffix": ask_suffix, "rows": len(rows), "path_in_repo": path}
    _put(root, f"runs/done/{name}.json", {"status": status, "extract": ex})
    _put(root, f"{path}/summary.json", {**ex, "split": "text_held", "package": "0.1.67"})
    if rows:
        _put(root, f"{path}/rows.jsonl", rows)


MANIFEST = [
    {"sample_id": "txt-coco-1", "group": "text_held_apache", "text_words": ["main", "stop"], "text_shown": ["STOP", "Main"],
     "name_screen": "clear"},
    {"sample_id": "txt-coco-2", "group": "text_held_apache", "text_words": ["cafe"], "text_shown": ["CAFE"], "name_screen": "clear"},
    {"sample_id": "txt-coco-3", "group": "text_held_nc", "text_words": ["hut", "pizza"], "text_shown": ["Pizza Hut"],
     "name_screen": "flagged"},
    {"sample_id": "txt-coco-4", "group": "text_held_nc", "text_words": ["bus"], "text_shown": ["BUS"], "name_screen": "clear"},
]
ROWS = [  # what an extraction of the four pictures wrote
    {"sample_id": "txt-coco-1", "text": "see: A red STOP sign on Main street.\ncaption: A stop sign.\nprompt: street photo"},
    {"sample_id": "txt-coco-2", "text": "see: A shop front.\ncaption: A shop."},
    {"sample_id": "txt-coco-3", "text": "see: A Pizza box.\ncaption: pizza on a table\nquestion: brand -> pizza hut [record]"},
    {"sample_id": "txt-coco-4", "text": "see: A bus at a stop.\ncaption: A bus."},
]


@pytest.fixture
def held(tmp_path):
    return _put(str(tmp_path), "held/text_held_v1_names.jsonl", MANIFEST)


def _derangement(ids, seed):
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return {sid: order[(k + 1) % len(order)] for k, sid in enumerate(order)}


def test_the_text_read_counts_rows_naming_the_shown_words(tmp_path, held):
    root = str(tmp_path / "repo")
    _extraction(root, "XT_a", ROWS, adapter=PROJECT, subfolder="lora/Qwen3.5-2B/r32/20261120_064248_V12_A1_gkl1_1ep_2b_by/final")
    _extraction(root, "XT_base", ROWS[:2])
    _extraction(root, "XT_other", ROWS[3:], adapter="AbstractPhil/qwen-symbolic-thinking-image-training",
                subfolder="lora/Qwen3.5-2B/r32/20261101_041601_G_med4_2ep_2b/final")
    records = rec.FolderRecords(root, PROJECT)
    results, lines_ = text_names.run(records, held, ["XT_a", "XT_base", "XT_other"], str(tmp_path / "out" / "text_read"))
    a = results["XT_a"]
    assert a["adapter"] == "project/20261120_064248_V12_A1_gkl1_1ep_2b_by x1" and a["rows"] == 4 and a["model"] == "Qwen/Qwen3.5-2B"
    assert results["XT_base"]["adapter"] == "none"
    assert results["XT_other"]["adapter"] == "qwen-symbolic-thinking-image-training/20261101_041601_G_med4_2ep_2b x1"
    assert list(a["by_line"]) == ["apache", "all", "name_clear", "nc", "name_flagged"]   # in order of first appearance
    al = a["by_line"]["all"]
    # rows 1, 3, 4 name a shown word; the chain and caption of the same three; no prompt; one question (pizza hut)
    assert (al["n"], al["row"], al["chain"], al["caption"], al["prompt"], al["questions"], al["recall"]) == (4, 0.75, 0.75, 0.75, 0.0, 0.25, 0.75)
    assert a["by_line"]["apache"]["row"] == 0.5 and a["by_line"]["nc"]["row"] == 1.0
    assert a["by_line"]["name_clear"]["n"] == 3 and a["by_line"]["name_clear"]["row"] == 0.667 and a["by_line"]["name_flagged"]["row"] == 1.0
    # the chance control: each row against its successor's words in the seeded order
    other = _derangement([m["sample_id"] for m in MANIFEST], text_names.SEED)
    words = {m["sample_id"]: set(m["text_words"]) for m in MANIFEST}
    chance = sum(bool(words[other[r["sample_id"]]] & text_names.tokens(r["text"])) for r in ROWS)
    assert all(other[s] != s for s in other) and al["chance"] == round(chance / 4, 3)
    # the words most often named: a tie is listed in the order first named (rows in order; within a row alphabetically)
    assert a["top_matched"] == [("main", 1), ("stop", 1), ("hut", 1), ("pizza", 1), ("bus", 1)]
    assert lines_[2].startswith("| XT_a | Qwen3.5-2B | project/20261120_064248_V12_A1_gkl1_1ep_2b_by x1 | 4 | 0.750 |")
    assert any(x.startswith("THE NAME SPLIT") for x in lines_) and lines_[-3] == "- XT_a: main 1, stop 1, hut 1, pizza 1, bus 1"
    out = tmp_path / "out"
    assert json.loads((out / "text_read.json").read_text(encoding="utf-8"))["XT_a"]["by_line"]["all"]["row"] == 0.75
    assert (out / "text_read.md").read_bytes() == (out / "text_read.txt").read_bytes()
    assert b"\r\n" not in (out / "text_read.md").read_bytes()


def test_the_text_read_takes_local_rows_and_their_hint_split(tmp_path, held):
    rows = tmp_path / "bench.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in [
        {**ROWS[0], "hint_words": ["Main"]},              # the hint gave "main": the row names it, and "stop" on its own
        {**ROWS[1], "hint_words": []}]), encoding="utf-8")  # unaided: "cafe" not named
    results, lines_ = text_names.run(rec.FolderRecords(str(tmp_path)), held, [str(rows)])
    r = results[str(rows)]
    assert r["adapter"] == "local file" and set(r["by_line"]) == {"apache", "all", "name_clear", "aided", "unaided"}
    h = r["by_line"]["all"]["hint_split"]
    assert h == {"rows": 2, "hinted_words_named": 1.0, "rows_with_hinted_words": 1, "unhinted_words_named": 0.5,
                 "names_an_unhinted_word": 0.5, "rows_with_unhinted_words": 2}
    assert r["by_line"]["unaided"]["hint_split"]["hinted_words_named"] is None
    assert any(x.startswith("THE HINT SPLIT") for x in lines_)
    assert text_names.hint_forms(["Security Camera", "ab"]) == {"security", "camera", "securitycamera"}


def test_the_direct_text_read_compares_the_answer_with_the_base(tmp_path, held):
    root = str(tmp_path / "repo")
    rows = [{"seed_id": "coco:1", "scale": 1.0, "text": "It says STOP.", "base_text": "STOP and Main.", "f1": 0.5},
            {"seed_id": "coco:2", "scale": 1.0, "text": "Cafe.", "base_text": "A shop.", "f1": 0.25},
            {"seed_id": "coco:3", "scale": 0.5, "text": "Pizza Hut", "base_text": "Pizza Hut", "f1": 1.0},
            {"seed_id": "coco:9", "scale": 1.0, "text": "not held", "base_text": "", "f1": 0.0}]
    _put(root, "runs/done/KT_a.json", {"status": "done", "tool": {"path_in_repo": "keep/20261120_000000_KT_a",
                                                                   "adapter_subfolder": "lora/x/final"}})
    _put(root, "keep/20261120_000000_KT_a/rows.jsonl", rows)
    results, lines_ = keep_text_names.run(rec.FolderRecords(root), held, ["KT_a"])
    s = results["KT_a"]["by_scale"]
    assert list(s) == ["1.0", "1.0 name_clear", "0.5", "0.5 name_flagged"]
    assert s["1.0"] == {"n": 2, "tuned": 1.0, "base": 0.5, "tuned_recall": 0.75, "base_recall": 0.5,
                        "chance": s["1.0"]["chance"], "f1": 0.375}
    assert s["0.5 name_flagged"]["n"] == 1 and s["0.5"]["tuned_recall"] == 1.0
    other = _derangement(["coco:1", "coco:2", "coco:3", "coco:4"], keep_text_names.SEED)
    words = {"coco:" + m["sample_id"].split("-")[-1]: set(m["text_words"]) for m in MANIFEST}
    want = sum(bool(words[other[r["seed_id"]]] & keep_text_names.tokens(r["text"])) for r in rows[:2])
    assert s["1.0"]["chance"] == round(want / 2, 3)
    assert lines_[2] == f"| KT_a | 1.0 | 2 | 1.000 | 0.500 | 0.750 | 0.500 | {s['1.0']['chance']:.3f} | 0.375 |"


def _row(sid, thoughts=(), caption="", prompt="", full_ok=True, rejected=None):
    return {"sample_id": sid, "thoughts": [{"step": s, "text": t} for s, t in thoughts],
            "wordings": {"caption": caption, "prompt": prompt}, "scores": {"full_ok": full_ok}, "rejected": rejected}


def test_the_lettering_read_counts_say_thoughts_and_quotes(tmp_path):
    root = str(tmp_path / "repo")
    rows = [_row("ctl-coco-1", [("see", "A sign."), ("say", 'The sign reads "OPEN".')]),       # a say thought and a quote
            _row("ctl-coco-2", [("see", "A wall.")], caption='A wall with “NO PARKING”.'),     # a quote in the caption only
            _row("ctl-coco-3", [("see", "A tree.")], full_ok=False, rejected="row shape"),      # nothing; not whole, not kept
            _row("ctl-coco-4", [("say", "Small print, unreadable.")])]                         # a say thought, no quote
    _extraction(root, "XCA2_a", rows, manifest="repo:text/notext_control_v1.jsonl", ask_suffix=SENTENCE)
    _extraction(root, "XT_empty", [], manifest="repo:text/text_held_v1.jsonl")
    report, lines_ = lettering_ask.run(rec.FolderRecords(root), ["XCA2_a", "XT_empty"], str(tmp_path / "out" / "lettering"))
    r = report["XCA2_a"]
    assert (r["rows"], r["kept"], r["full_ok"], r["say"], r["quoted"], r["either"]) == (4, 3, 3, 2, 2, 3)
    assert [e["sample_id"] for e in r["examples"]] == ["ctl-coco-1", "ctl-coco-2", "ctl-coco-4"]
    assert report["XT_empty"] == {"rows": 0, "manifest": "repo:text/text_held_v1.jsonl", "split": "text_held", "empty": True}
    assert lines_[2] == ("| XCA2_a | lettering sentence | 4 | 3 (0.750) | 3 (0.750) | 2 (0.500) | 2 (0.500) | 3 (0.750) |")
    assert lines_[3].startswith("| XT_empty | trained | 0 (selected no picture: split text_held)")
    assert "XCA2_a (lettering sentence), rows that write lettering on a picture COCO-Text marks as holding none:" in lines_
    assert '- ctl-coco-1: say: The sign reads "OPEN".; quote "OPEN"' in lines_
    saved = json.loads((tmp_path / "out" / "lettering.json").read_text(encoding="utf-8"))
    assert saved["XCA2_a"]["examples"][1]["quotes"] == ["“NO PARKING”"]


def test_quote_precision_classifies_the_quoted_words(tmp_path, held):
    root = str(tmp_path / "repo")
    rows = [_row("txt-coco-1", [("say", 'It reads "STOP" and "MAIN".')]),        # every quoted word the picture's: all
            _row("txt-coco-2", [("say", 'It reads "CAFE ROYAL".')]),            # cafe yes, royal no: some
            _row("txt-coco-3", [("see", "A box.")], caption='A box saying "DOMINOS".'),   # none of the picture's words
            _row("txt-coco-4", [("say", "Some lettering, too small.")])]        # lettering without a quote: unquoted
    _extraction(root, "XTA_a", rows, ask_suffix=SENTENCE)
    report, lines_ = quote_precision.run(rec.FolderRecords(root), held, ["XTA_a"])
    a = report["XTA_a"]
    assert {k: a["all"][k] for k in ("pictures", "writing", "all", "some", "none", "unquoted")} == \
        {"pictures": 4, "writing": 4, "all": 1, "some": 1, "none": 1, "unquoted": 1}
    assert (a["clear"]["pictures"], a["flagged"]["pictures"], a["flagged"]["none"]) == (3, 1, 1)
    other = _derangement([m["sample_id"] for m in MANIFEST], quote_precision.SEED)
    words = {m["sample_id"]: set(m["text_words"]) for m in MANIFEST}
    quoted = {"txt-coco-1": {"stop", "main"}, "txt-coco-2": {"cafe", "royal"}, "txt-coco-3": {"dominos"}, "txt-coco-4": set()}
    want = sum(bool(q & words[other[s]]) for s, q in quoted.items())
    assert a["all"]["chance_all_or_some"] == want
    assert lines_[2] == f"| XTA_a | all | 4 | 4 (1.000) | 1 | 1 | 1 | 1 | {want} |"
    assert lines_[-4].startswith("- txt-coco-1 (shown: STOP | Main): say: It reads")
    assert quote_precision.cls(set(), {"a"}) == "unquoted" and quote_precision.words("Hi THERE, 42 x") == {"there"}


# ------------------------------------------------------------------------------------------------------------ the bars
def _arm(root, run, tag, said, backed, probe_texts, drift, form, ended, base_form, base_ended, full_rows, manifest):
    _put(root, f"runs/done/{run}.json", {"status": "done", "screen": {"Qwen/Qwen3.5-2B | short": {},
                                                                     "Qwen/Qwen3.5-2B | full": {"hint_said": said, "hint_backed": backed}}})
    _put(root, f"screens/20261120_000000_{run}/results.jsonl", full_rows + [{"form": "short", "scores": {}, "parsed": {}}])
    _put(root, f"runs/done/K_{tag}.json", {"status": "done", "tool": {
        "by_scale": {"1.0": {"drift": {"picture-held": {"kl_mean": drift[0]}, "text-held": {"kl_mean": drift[1]},
                                       "all": {"kl_mean": drift[2]}},
                             "takeover": {"all": {"takes_row_form": form, "ended": ended}}}},
        "base": {"takeover": {"all": {"takes_row_form": base_form, "ended": base_ended}}}}})
    path = f"extractions/20261120_000000_P_probe_{tag}"
    _put(root, f"runs/done/P_probe_{tag}.json", {"status": "done", "extract": {"path_in_repo": path, "manifest": manifest}})
    _put(root, f"{path}/rows.jsonl", [{"sample_id": f"nga-{i}", "text": t} for i, t in enumerate(probe_texts)])


def _full(depth, wordings, fmt=True, in_range=True, needs=True):
    return {"form": "full", "depth": depth, "scores": {"format_ok": fmt, "questions_in_range": in_range},
            "parsed": {"wordings": {w: "x" for w in wordings}, "questions": [{"needs": "record"} if needs else {"key": "k"}]}}


@pytest.fixture
def bars_repo(tmp_path):
    root = str(tmp_path / "repo")
    probe_manifest = _put(root, "pictures/manifest_v1.4.jsonl", [
        {"sample_id": "nga-0", "split": "probe", "labels": {"genre": "Painting"}},
        {"sample_id": "nga-1", "split": "probe", "labels": {"genre": "Print"}},
        {"sample_id": "nga-2", "split": "probe", "labels": {"genre": "Photograph"}},
        {"sample_id": "nga-3", "split": "main", "labels": {"genre": "Painting"}}])     # not the probe split: not counted
    texts = ["prompt: oil painting, a harbor", "prompt: etching, a street", "prompt: drawing, a man", "prompt: oil painting"]
    whole = [_full("medium", ("caption", "prompt", "tags")), _full("deep", ("caption", "critique", "prompt", "tags")),
             _full("simple", ("caption",)), _full("medium", ("caption",))]               # the last: medium wordings missing
    _arm(root, "V12_A1_gkl1_1ep_2b_by", "v12_A1_2b_by", 0.217, 0.220, texts, (0.066, 0.069, 0.085), 0.0, 0.883, 0.0,
         0.887, whole, "repo:pictures/manifest_v1.4.jsonl")
    _arm(root, "V11_A0_rep_2ep_4b", "v11_A0", 0.209, 0.225, texts[:1], (1.597, 0.593, 1.154), 0.95, 0.80, 0.001, 0.887,
         whole[:2], "repo:pictures/manifest_v1.4.jsonl")
    return root, probe_manifest


def test_the_bars_read_a_version_beside_its_yardstick(tmp_path, bars_repo):
    root, _ = bars_repo
    data, lines_ = bars.run(rec.FolderRecords(root), "V12_A1_gkl1_1ep_2b_by", "v12_A1_2b_by", "V11_A0_rep_2ep_4b", "v11_A0",
                            str(tmp_path / "out" / "bars_v12_A1_2b_by"))
    v = data["version"]
    # T1: four full-form answers in the format, three of them whole (the fourth lacks the medium wordings); the probe: the
    # painting and the print named (a drawing's prompt on a photograph is not; the main split's picture does not count)
    assert v["t1"] == [1.0, 0.75] and v["probe"] == 2 and data["yardstick"]["probe"] == 1
    assert lines_[0] == "V12_A1_gkl1_1ep_2b_by against V11_A0_rep_2ep_4b (drift all 0.085 against 1.154)"
    assert lines_[3] == "| T1 full-form format | 1.000 | >= 0.95 | pass |  |"
    assert lines_[4] == "| T1 full rows | 0.750 | >= 0.95 | MISS |  |"
    assert lines_[7] == "| T3 probe | 2 | within 10 of A0 1 | pass |  |"
    assert lines_[10] == "| K2 row form | 0.000 | excess <= 0.02 and <= 0.5 x A0's 0.949 | pass |  |"
    assert lines_[-1] == "NOT READY: T1 full rows" and data["pass"] is False and data["misses"] == ["T1 full rows"]
    assert (tmp_path / "out" / "bars_v12_A1_2b_by.txt").read_text(encoding="utf-8") == "\n".join(lines_) + "\n"
    assert not (tmp_path / "out" / "bars_v12_A1_2b_by.md").exists()
    saved = json.loads((tmp_path / "out" / "bars_v12_A1_2b_by.json").read_text(encoding="utf-8"))
    assert saved["yardstick"]["run"] == "V11_A0_rep_2ep_4b" and saved["bars"][1]["read"] == "MISS"


def test_the_bars_marks_and_their_edges():
    a0 = {"said": 0.209, "backed": 0.225, "probe": 143, "kl_picture": 1.6, "kl_text": 0.6, "form": 0.95, "base_form": 0.0}
    v = {"t1": [1.0, 0.96], "said": 0.30, "backed": 0.196, "probe": 133, "kl_picture": 0.8, "kl_text": 0.31, "form": 0.02,
         "base_form": 0.0, "ended": 0.866, "base_ended": 0.887}
    rows, misses = bars.check(v, a0)
    got = {r[0]: (r[3], r[4]) for r in rows}
    assert got["T2 said"] == ("pass", "outside the band upward") and got["T2 backed"] == ("pass", "")      # -0.029: in the band
    assert got["T3 probe"] == ("pass", "") and got["K1 drift picture"] == ("pass", "")                    # -10; exactly half
    assert got["K1 drift text"] == ("MISS", "") and got["K2 row form"] == ("pass", "") and got["K3 ended"] == ("MISS", "")
    assert misses == ["K1 drift text", "K3 ended"]


def test_the_bars_take_the_probe_manifest_from_the_probe_or_a_local_copy(tmp_path, bars_repo):
    root, local = bars_repo
    records = rec.FolderRecords(root)
    assert bars.probe_manifest(records, "v12_A1_2b_by") == "repo:pictures/manifest_v1.4.jsonl"
    data, _ = bars.run(records, "V12_A1_gkl1_1ep_2b_by", "v12_A1_2b_by", "V11_A0_rep_2ep_4b", "v11_A0", probe_manifest_path=local)
    assert data["probe_manifest"] == local and data["version"]["probe"] == 2


# ------------------------------------------------------------------------------------------------------------ the read sets
def _sets(key, line):
    return {stem: [n for n, _ in runs] for stem, _, runs in reads.text_sets(registry.get(key), line)}


# The sets read by hand for the 2B v1.2 lines (Apache-2.0 2026-11-20, -nc 2026-11-21), the 9B v1.2 Apache-2.0 line
# (2026-11-21) and written for the 0.8B lines; the command must take the same runs in the same order.
HAND = {
    ("2b", "apache"): {
        "text_read_v12_2b_by": "XT_base_2b_noadapter XT_v1_2b_s1 XT_v12_2b_by_s1",
        "text_read_v12_2b_by_deep": "XT_base_2b_deep XT_v1_2b_s1_deep XT_v12_2b_by_s1_deep",
        "text_read_lettering_ask_2b_by": "XTA_base_2b XTA_v12_2b_by_s1",
        "keep_text_read_v12_2b_by": "KT_v12_2b_by",
        "lettering_ask_read_text_2b_by": "XT_base_2b_noadapter XT_v1_2b_s1 XT_v12_2b_by_s1 XT_v12_2b_by_s1_deep XTA_base_2b "
                                         "XTA_v12_2b_by_s1",
        "lettering_ask_read_ctl_2b_by": "XC2_base_2b XCA2_base_2b XC2_v12_2b_by_s1 XCA2_v12_2b_by_s1",
        "lettering_quote_precision_2b_by": "XTA_v12_2b_by_s1 XT_v12_2b_by_s1 XT_v12_2b_by_s1_deep XTA_base_2b"},
    ("2b", "nc"): {
        "text_read_v12_2b_nc": "XT_base_2b_noadapter XT_v1_2b_s1 XT_v12_2b_by_s1 XT_v12_2b_nc_s1",
        "text_read_v12_2b_nc_deep": "XT_base_2b_deep XT_v1_2b_s1_deep XT_v12_2b_by_s1_deep XT_v12_2b_nc_s1_deep",
        "text_read_lettering_ask_2b_nc": "XTA_base_2b XTA_v12_2b_by_s1 XTA_v12_2b_nc_s1",
        "keep_text_read_v12_2b_nc": "KT_v12_2b_by KT_v12_2b_nc",
        "lettering_ask_read_text_2b_nc": "XT_base_2b_noadapter XT_v1_2b_s1 XT_v12_2b_nc_s1 XT_v12_2b_nc_s1_deep XTA_base_2b "
                                         "XTA_v12_2b_by_s1 XTA_v12_2b_nc_s1",
        "lettering_ask_read_ctl_2b_nc": "XC2_base_2b XCA2_base_2b XC2_v12_2b_by_s1 XCA2_v12_2b_by_s1 XC2_v12_2b_nc_s1 "
                                        "XCA2_v12_2b_nc_s1",
        "lettering_quote_precision_2b_nc": "XTA_v12_2b_nc_s1 XT_v12_2b_nc_s1 XT_v12_2b_nc_s1_deep XTA_base_2b"},
    ("9b", "apache"): {
        "text_read_v12_9b_by": "XT_base_9b_noadapter XT_v11_9b_s1 XT_v11nc_9b_s1 XT_v12_9b_by_s1",
        "text_read_v12_9b_by_deep": "XT_base_9b_deep XT_v11_9b_by_s1_deep XT_v11nc_9b_s1_deep XT_v12_9b_by_s1_deep",
        "text_read_lettering_ask_9b_by": "XTA_base_9b XTA_v11_9b_by_s1 XTA_v11nc_9b_s1 XTA_v12_9b_by_s1",
        "keep_text_read_v12_9b_by": "KT_v11_9b_by KT_v12_9b_by",
        "lettering_ask_read_text_9b_by": "XT_base_9b_noadapter XT_v11_9b_s1 XT_v12_9b_by_s1 XT_v12_9b_by_s1_deep XTA_base_9b "
                                         "XTA_v11_9b_by_s1 XTA_v12_9b_by_s1",
        "lettering_ask_read_ctl_9b_by": "XC2_base_9b XCA2_base_9b XC2_v11_9b_by_s1 XCA2_v11_9b_by_s1 XC2_v12_9b_by_s1 "
                                        "XCA2_v12_9b_by_s1",
        "lettering_quote_precision_9b_by": "XTA_v12_9b_by_s1 XT_v12_9b_by_s1 XT_v12_9b_by_s1_deep XTA_base_9b"},
    ("08b", "apache"): {
        "text_read_v12_08b_by": "XT_base_08b_noadapter XT_v1_08b_s1 XT_v12_08b_by_s1",
        "text_read_v12_08b_by_deep": "XT_base_08b_deep XT_v1_08b_s1_deep XT_v12_08b_by_s1_deep",
        "text_read_lettering_ask_08b_by": "XTA_base_08b XTA_v12_08b_by_s1",
        "keep_text_read_v12_08b_by": "KT_v12_08b_by",
        "lettering_ask_read_text_08b_by": "XT_base_08b_noadapter XT_v1_08b_s1 XT_v12_08b_by_s1 XT_v12_08b_by_s1_deep "
                                          "XTA_base_08b XTA_v12_08b_by_s1",
        "lettering_ask_read_ctl_08b_by": "XC2_base_08b XCA2_base_08b XC2_v12_08b_by_s1 XCA2_v12_08b_by_s1",
        "lettering_quote_precision_08b_by": "XTA_v12_08b_by_s1 XT_v12_08b_by_s1 XT_v12_08b_by_s1_deep XTA_base_08b"},
    ("08b", "nc"): {
        "text_read_v12_08b_nc": "XT_base_08b_noadapter XT_v1_08b_s1 XT_v12_08b_by_s1 XT_v12_08b_nc_s1",
        "text_read_v12_08b_nc_deep": "XT_base_08b_deep XT_v1_08b_s1_deep XT_v12_08b_by_s1_deep XT_v12_08b_nc_s1_deep",
        "text_read_lettering_ask_08b_nc": "XTA_base_08b XTA_v12_08b_by_s1 XTA_v12_08b_nc_s1",
        "keep_text_read_v12_08b_nc": "KT_v12_08b_by KT_v12_08b_nc",
        "lettering_ask_read_text_08b_nc": "XT_base_08b_noadapter XT_v1_08b_s1 XT_v12_08b_nc_s1 XT_v12_08b_nc_s1_deep "
                                          "XTA_base_08b XTA_v12_08b_by_s1 XTA_v12_08b_nc_s1",
        "lettering_ask_read_ctl_08b_nc": "XC2_base_08b XCA2_base_08b XC2_v12_08b_by_s1 XCA2_v12_08b_by_s1 XC2_v12_08b_nc_s1 "
                                         "XCA2_v12_08b_nc_s1",
        "lettering_quote_precision_08b_nc": "XTA_v12_08b_nc_s1 XT_v12_08b_nc_s1 XT_v12_08b_nc_s1_deep XTA_base_08b"},
}
HAND_BARS = {("2b", "apache"): ("V12_A1_gkl1_1ep_2b_by", "v12_A1_2b_by", "V11_A0_rep_2ep_4b", "v11_A0"),
             ("2b", "nc"): ("V12NC_A1_gkl1_1ep_2b", "v12_A1_2b_nc", "V11NC_A0_rep_2ep_4b_r2", "v11nc_A0_r2"),
             ("9b", "apache"): ("V12_A1_gkl1_1ep_9b_by", "v12_A1_9b_by", "V11_A0_rep_2ep_4b", "v11_A0"),
             ("08b", "apache"): ("V12_A1_gkl1_1ep_08b_by", "v12_A1_08b_by", "V11_A0_rep_2ep_4b", "v11_A0"),
             ("08b", "nc"): ("V12NC_A1_gkl1_1ep_08b", "v12_A1_08b_nc", "V11NC_A0_rep_2ep_4b_r2", "v11nc_A0_r2")}


@pytest.mark.parametrize("key,line", list(HAND))
def test_the_read_sets_are_the_ones_read_by_hand(key, line):
    assert _sets(key, line) == {stem: runs.split() for stem, runs in HAND[(key, line)].items()}
    assert reads.bars_set(registry.get(key), line) == HAND_BARS[(key, line)]


def test_every_model_and_line_has_a_read_set_from_the_generators_names():
    for key in registry.keys():
        e = registry.get(key)
        for line in e["lines"]:
            sets = reads.text_sets(e, line)
            assert [s[0] for s in sets][0] == f"text_read_v12_{key}_{lines.TAG[line]}" and len(sets) == 7
            own = set(lines.names(e, line)["reads"])
            assert own & {n for _, _, runs in sets for n, _ in runs}, (key, line)
            assert reads.bars_set(e, line)[0] == lines.names(e, line)["train"]
    nine = reads.subjects(registry.get("9b"), "nc")
    assert [s["who"] for s in nine] == ["base", "older", "older", "sibling", "line"]
    assert [s["on_line"] for s in nine if s["who"] == "older"] == [False, True]        # v1.1 Apache-2.0, v1.1 -nc
    assert reads.subjects(registry.get("2b"), "nc")[1]["on_line"] is True               # version 1 was one line


# ------------------------------------------------------------------------------------------------------------ the command
def _lane_records(root, entry, line, skip=()):
    """Records for every run a line's text set reads, on the four held pictures (or the control's), each run done."""
    _put(root, "text/text_held_v1_names.jsonl", MANIFEST)
    for _, tool, runs in reads.text_sets(entry, line):
        for name, _ in runs:
            if name in skip or os.path.exists(os.path.join(root, "runs", "done", f"{name}.json")):
                continue
            if tool == "keep_text_names":
                _put(root, f"runs/done/{name}.json", {"status": "done", "tool": {"path_in_repo": f"keep/s_{name}"}})
                _put(root, f"keep/s_{name}/rows.jsonl", [{"seed_id": "coco:1", "scale": 1.0, "text": "STOP", "base_text": "", "f1": 0.5}])
            elif name.startswith("XC"):
                _extraction(root, name, [_row("ctl-coco-1", [("see", "A tree.")])], manifest="repo:text/notext_control_v1.jsonl")
            else:
                _extraction(root, name, [{**r, **_row(r["sample_id"], [("say", 'It reads "STOP".')])} for r in ROWS],
                            ask_suffix=SENTENCE if name.startswith("XTA") else None)


def test_the_command_reads_a_lines_text_set_from_local_records(tmp_path, capsys):
    root, out = str(tmp_path / "repo"), str(tmp_path / "out")
    entry = registry.get("08b")
    _lane_records(root, entry, "apache", skip={"XT_v1_08b_s1_deep"})           # a comparison read the project repo lacks
    code = reads.main(["08b", "apache", "text", "--records", root, "--project-repo", PROJECT, "--out", out])
    said = capsys.readouterr().out
    assert code == 0 and "left out: XT_v1_08b_s1_deep" in said
    assert sorted(os.listdir(out)) == sorted(f"{stem}{ext}" for stem in HAND[("08b", "apache")] for ext in (".json", ".md", ".txt"))
    deep = json.loads(open(os.path.join(out, "text_read_v12_08b_by_deep.json"), encoding="utf-8").read())
    assert list(deep) == ["XT_base_08b_deep", "XT_v12_08b_by_s1_deep"]
    quotes = json.loads(open(os.path.join(out, "lettering_quote_precision_08b_by.json"), encoding="utf-8").read())
    q = quotes["XTA_v12_08b_by_s1"]["all"]        # every row quotes "STOP": the first picture's own word, the others' none
    assert (q["writing"], q["all"], q["some"], q["none"]) == (4, 1, 0, 3)
    # the line's own read missing: nothing is read
    os.remove(os.path.join(root, "runs", "done", "KT_v12_08b_by.json"))
    assert reads.main(["08b", "apache", "text", "--records", root, "--out", str(tmp_path / "again")]) == 2
    assert "KT_v12_08b_by" in capsys.readouterr().err and not os.path.exists(os.path.join(str(tmp_path / "again"), "text_read_v12_08b_by.json"))


def test_the_command_needs_the_named_manifest_or_says_what_to_pass(tmp_path, capsys):
    root = str(tmp_path / "repo")
    _lane_records(root, registry.get("2b"), "apache")
    os.remove(os.path.join(root, "text", "text_held_v1_names.jsonl"))
    _put(root, "text/text_held_v1.jsonl", [{k: v for k, v in m.items() if k != "name_screen"} for m in MANIFEST])
    assert reads.main(["2b", "apache", "text", "--records", root, "--out", str(tmp_path / "o1")]) == 2
    assert "--manifest repo:text/text_held_v1.jsonl" in capsys.readouterr().err
    assert reads.main(["2b", "apache", "text", "--records", root, "--out", str(tmp_path / "o2"), "--manifest",
                       "repo:text/text_held_v1.jsonl"]) == 0
    md = open(os.path.join(str(tmp_path / "o2"), "text_read_v12_2b_by.md"), encoding="utf-8").read()
    assert "THE NAME SPLIT" not in md                                          # no name screen: no split


def test_the_command_reads_the_bars_and_prints_a_plan_without_reading(tmp_path, capsys, bars_repo):
    root, local = bars_repo
    out = str(tmp_path / "out")
    assert reads.main(["2b", "apache", "bars", "--records", root, "--out", out, "--probe-manifest", local]) == 0
    assert "NOT READY: T1 full rows" in capsys.readouterr().out
    assert sorted(os.listdir(out)) == ["bars_v12_A1_2b_by.json", "bars_v12_A1_2b_by.txt"]
    assert reads.main(["9b", "nc", "text", "--dry-run"]) == 0
    plan = capsys.readouterr().out
    assert "keep_text_read_v12_9b_nc (tools.keep_text_names): KT_v11_9b_by KT_v12_9b_by KT_v12_9b_nc" in plan
    assert reads.main(["north", "nc", "text", "--dry-run"]) == 2


def test_the_new_modules_name_no_repo_that_is_not_public():
    """The reads take the project repo as a parameter; the repo the lane fixtures name is in none of them, nor the in-house
    words the hand-run tools used."""
    fix = os.path.join(os.path.dirname(__file__), "fixtures", "v12_lanes", "queue_lucia_v12_9b.json")
    with open(fix, encoding="utf-8") as fh:
        private = next(s["extract"]["out_repo"] for s in json.load(fh) if s["kind"] == "extract")
    pkg = os.path.dirname(registry.__file__)
    for rel in ("reads.py", *(os.path.join("tools", f"{m}.py") for m in (
            "records", "bars", "text_names", "keep_text_names", "lettering_ask", "quote_precision"))):
        with open(os.path.join(pkg, rel), encoding="utf-8") as fh:
            text = fh.read()
        assert private not in text and "-private" not in text and "twin" not in text and "his sentence" not in text, rel
