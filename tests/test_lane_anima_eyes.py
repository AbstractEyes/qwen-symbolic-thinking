"""The lane of Anima's eyes (registry.LATER["ae1b"]; lines.py's prepare run, requests file, family train keys and full
fine-tune): built from the registry like every lane, with the base made by its first run, the general rows' text requests from
the cleaned requests file, a LoRA and a full fine-tune each after its own smoke, and every run waiting for the base. Its retrain
(ae1bt: the LoRA with its output head's turn-end row trained, registry retrains): one training and its nine reads."""
import json

import pytest

from qwen_symbolic_thinking import __version__, anima_eyes, lines, registry

PROJECT = "someone/their-project"
FILES = set(lines.PROJECT_FILES) | {"text/requests_qwen_v1_clean.json"}


def built(files=FILES):
    return lines.build_lane(registry.get("ae1b"), "apache", PROJECT, files)


def test_the_entry_is_a_later_addition_and_the_v1_list_is_unchanged():
    assert "ae1b" in registry.LATER and "ae1b" not in registry.ENTRIES and "ae1b" not in registry.keys()
    assert registry.all_keys()[-1] == "ae1b"
    e = registry.get("ae1b")
    assert e["base"] == anima_eyes.MODEL_ID and registry.get(anima_eyes.MODEL_ID)["key"] == "ae1b"
    assert set(registry.FIELDS) <= set(e) and e["lines"] == ("apache",) and e["licenses"] == {"apache": "Apache-2.0"}
    f = registry.full_entry(e)
    assert f["key"] == "ae1bf" and f["lora"]["rank"] == 0 and f["lr"] == 2e-5 and f["is_full"] and f["full"] is None
    assert registry.smoke_stop(f, "apache")["steady_s_per_step"] == round(1.5 * 1.8, 2)
    assert registry.smoke_stop(e, "apache")["steady_s_per_step"] == round(1.5 * 2.1, 2)
    assert registry.full_entry(registry.get("08b")) is None
    text = registry.describe(e)
    assert "full fine-tune (ae1bf)" in text and "base_ae1b" in text and "requests_qwen_v1_clean.json" in text


def test_the_lane_builds_in_order_with_nothing_missing():
    b = built()
    names = [s["name"] for s in b["specs"]]
    assert names == ["base_ae1b", "v11a_ae1b", "v11a_ae1b_by", "v11txt_ae1b", "S_v12_smoke_gc1_ae1b", "S_v12_smoke_gc1_ae1bf",
                     "V12_A1_gkl1_1ep_ae1bf_by", "V12_A1_gkl1_1ep_ae1b_by", "XT_base_ae1b_noadapter", "XT_base_ae1b_deep",
                     "XTA_base_ae1b", "XCA2_base_ae1b", "XC2_base_ae1b"]
    assert b["problems"] == [] and b["missing"] == [] and b["lane"] == "anima_eyes_v12"
    assert all(s["needs"] == __version__ for s in b["specs"])
    base = b["specs"][0]
    assert base["tool"] == {"module": "anima_eyes_base", "args": {"head": anima_eyes.HEAD}}
    assert base["makes"][0] == f"{anima_eyes.FOLDER}/model.safetensors" and "waits_for_runs" not in base
    for s in b["specs"][1:]:
        assert s["waits_for_runs"][0] == "base_ae1b" and s["waits_for"][0] == base["makes"][0], s["name"]
    bare = built(set(lines.PROJECT_FILES))
    assert [m.split(" ")[0] for m in bare["missing"]] == ["text/requests_qwen_v1_clean.json"]


def test_the_general_rows_ask_the_cleaned_requests_and_the_model_answers():
    g = built()["specs"][1]["tool"]["args"]
    assert g["model_id"] == anima_eyes.MODEL_ID and g["requests"] == "hub:text/requests_qwen_v1_clean.json"
    assert (g["n_picture"], g["n_text"], g["n_held_picture"], g["n_held_text"]) == (4000, 500, 200, 40)
    assert built()["specs"][1]["waits_for"][-1] == "text/requests_qwen_v1_clean.json"


def test_the_lora_and_the_full_finetune():
    specs = {s["name"]: s for s in built()["specs"]}
    lo, fu = specs["V12_A1_gkl1_1ep_ae1b_by"]["train"], specs["V12_A1_gkl1_1ep_ae1bf_by"]["train"]
    assert (lo["rank"], lo["alpha"], lo["lr"], lo["resume"], lo["ship_every"]) == (32, 64, 2e-4, True, 1000)
    assert (fu["rank"], fu["lr"], fu["resume"], fu["ship_every"]) == (0, 2e-5, False, 0)
    for t in (lo, fu):
        assert t["model_id"] == anima_eyes.MODEL_ID and t["slot"] == 1.0 and t["slot_batch"] == 4 and t["general_kl"] == 1.0
        assert t["anchors"] == "hub:anchors/v11a_ae1b_by/anchors.json" and t["writers"][-1] == anima_eyes.MODEL_ID
    assert specs["V12_A1_gkl1_1ep_ae1bf_by"]["waits_for_runs"] == ["base_ae1b", "S_v12_smoke_gc1_ae1bf"]
    assert specs["S_v12_smoke_gc1_ae1bf"]["train"]["rank"] == 0 and specs["S_v12_smoke_gc1_ae1bf"]["train"]["max_steps"] == 30
    reads = [a["name"] for a in specs["V12_A1_gkl1_1ep_ae1bf_by"]["after"]]
    assert reads[0] == "P_probe_v12_A1_ae1bf_by" and len(reads) == 9
    # the keep reads: the full fine-tune against the model it started from (no off switch), the LoRA against its own off switch
    for run, want in (("V12_A1_gkl1_1ep_ae1bf_by", {"scales": [1.0], "base_model_id": anima_eyes.MODEL_ID}),
                      ("V12_A1_gkl1_1ep_ae1b_by", {"scales": [1.0, 0.5]})):
        keeps = [a["tool"]["args"] for a in specs[run]["after"] if a["kind"] == "tool"]
        assert len(keeps) == 2 and all({k: a.get(k) for k in ("scales", "base_model_id")} == {"base_model_id": None, **want}
                                       for a in keeps), run


def test_a_base_already_built_is_not_built_again():
    files = FILES | set(registry.get("ae1b")["prepare"]["makes"])
    b = built(files)
    assert b["specs"][0]["name"] == "v11a_ae1b" and not any("waits_for_runs" in s and "base_ae1b" in s["waits_for_runs"]
                                                             for s in b["specs"])
    assert any("in the project repo (not made again)" in c for c in b["choices"]) and b["missing"] == []


def test_the_notebook_installs_this_version_and_describes_the_slot():
    e = registry.get("ae1b")
    b = built()
    nb = lines.notebook(e, b, PROJECT)
    md, c1, c2 = ("".join(c["source"]) for c in nb["cells"])
    assert f"@v{__version__}\"" in c1 and 'LANE = "anima_eyes_v12"' in c2 and PROJECT in c2
    assert "InternVL3_5-1B-HF" in md and "Qwen3-0.6B-Base" in md and "slot term" in md and "full fine-tune" in md
    assert "huggingface.co/hub:" not in md


def test_the_command_refuses_the_nc_line(tmp_path):
    listing = tmp_path / "files.json"
    listing.write_text(json.dumps(sorted(FILES)), encoding="utf-8")
    assert lines.main(["ae1b", "nc", "--project-repo", PROJECT, "--files", str(listing), "--out", str(tmp_path)]) == 2
    assert lines.main(["ae1b", "apache", "--project-repo", PROJECT, "--files", str(listing), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "queue_anima_eyes_v12.json").exists() and (tmp_path / "anima_eyes_v12.ipynb").exists()


# ------------------------------------------------------------------- the retrain: the LoRA with its head's turn-end row
LANE_DONE = FILES | set(registry.get("ae1b")["prepare"]["makes"]) | {
    f"anchors/{n}/{f}" for n in ("v11a_ae1b", "v11a_ae1b_by", "v11txt_ae1b") for f in ("anchors.json", "held.json", "summary.json")}
BASE_READS = {"closer": "XT_base_ae1b_noadapter_t1", "deep": "XT_base_ae1b_deep", "lettering": "XTA_base_ae1b",
              "control_lettering": "XCA2_base_ae1b", "control": "XC2_base_ae1b"}


def test_the_retrain_is_the_lora_with_its_turn_end_row_trained():
    e, lo = registry.get("ae1bt"), registry.get("ae1b")
    assert registry.get("AE1BT")["key"] == "ae1bt" and e["retrain_of"] == "ae1b" and e["lane"] == lo["lane"]
    assert "ae1bt" not in registry.all_keys() and registry.retrain_keys() == ["ae1bt"]
    assert registry.get(anima_eyes.MODEL_ID)["key"] == "ae1b"             # the base id still finds the model, not the retrain
    assert e["lora"] == {"rank": 32, "alpha": 64, "token_rows": {"lm_head": ["turn_end"]}} and e["full"] is None
    assert e["retrains"] is None and e["no_smoke"] and e["comparisons"] == [{"of": "base", "reads": [], "earlier": BASE_READS}]
    for k in ("base", "lr", "batch", "grad_accum", "max_len", "image_settings", "general_answers", "general_requests",
              "train_extra", "prepare", "lines", "licenses"):
        assert e[k] == lo[k], k
    r, rl = registry.recipe(e), registry.recipe(lo)
    assert list(r) == list(rl)[:3] + ["token_rows"] + list(rl)[3:] and {k: v for k, v in r.items() if k != "token_rows"} == rl
    text = registry.describe(e)
    assert text.startswith("ae1bt: ") and "a retrain of ae1b's LoRA" in text and "no smoke" in text and "token rows" in text
    assert "retrain ae1bt" in registry.describe(lo)
    with pytest.raises(KeyError):
        registry.retrain_entry(lo, "ae1bx")


def test_the_retrain_lane_is_its_training_and_its_nine_reads_only():
    """With the first lane done (the base, the general answers and the base comparisons in the project repo), the retrain's
    lane is one spec: the LoRA's training spec key for key, plus its token rows, under the retrain's run names, waiting for no
    smoke; no base build, no general answers, no smoke, no comparisons."""
    b = lines.build_lane(registry.get("ae1bt"), "apache", PROJECT, LANE_DONE)
    assert [s["name"] for s in b["specs"]] == ["V12_A1_gkl1_1ep_ae1bt_by"] and b["problems"] == [] and b["missing"] == []
    new = b["specs"][0]
    old = next(s for s in lines.build_lane(registry.get("ae1b"), "apache", PROJECT, LANE_DONE)["specs"]
               if s["name"] == "V12_A1_gkl1_1ep_ae1b_by")
    assert "waits_for_runs" not in new and old["waits_for_runs"] == ["S_v12_smoke_gc1_ae1b"]
    assert {k: v for k, v in new["train"].items() if k != "token_rows"} == old["train"]
    assert new["train"]["token_rows"] == {"lm_head": ["turn_end"]}
    assert new["screen"] == old["screen"] and new["waits_for"] == old["waits_for"] and new["needs"] == __version__
    assert [a["name"] for a in new["after"]] == lines.names(registry.get("ae1bt"), "apache")["reads"]
    assert [a["name"].replace("ae1bt", "ae1b") for a in new["after"]] == [a["name"] for a in old["after"]]
    assert [{k: v for k, v in a.items() if k != "name"} for a in new["after"]] == \
           [{k: v for k, v in a.items() if k != "name"} for a in old["after"]]
    keeps = [a["tool"]["args"] for a in new["after"] if a["kind"] == "tool"]
    assert len(keeps) == 2 and all(a["scales"] == [1.0, 0.5] for a in keeps)           # the row scales with the pairs
    assert any(c.startswith("  no smoke: ") for c in b["choices"])
    assert any("none made in this lane" in c and "XT_base_ae1b_noadapter_t1" in c for c in b["choices"])


def test_the_retrain_lane_builds_what_the_project_repo_lacks_first():
    names = [s["name"] for s in lines.build_lane(registry.get("ae1bt"), "apache", PROJECT, FILES)["specs"]]
    assert names == ["base_ae1b", "v11a_ae1b", "v11a_ae1b_by", "v11txt_ae1b", "V12_A1_gkl1_1ep_ae1bt_by"]


def test_the_retrain_notebook_installs_this_version_and_names_its_run():
    e = registry.get("ae1bt")
    b = lines.build_lane(e, "apache", PROJECT, LANE_DONE)
    md, c1, c2 = ("".join(c["source"]) for c in lines.notebook(e, b, PROJECT)["cells"])
    assert f"@v{__version__}\"" in c1 and f'"{lines.PEFT_TOKEN_ROWS}"' in c1 and 'LANE = "anima_eyes_v12"' in c2
    assert "The queue runs, in order: the training V12_A1_gkl1_1ep_ae1bt_by and its nine reads." in md
    assert "A smoke that runs slower" not in md and "turn-end row" in md and "about 5.6 h of training at 2.0 s a step" in md
    c1_lora = "".join(lines.notebook(registry.get("ae1b"), built(), PROJECT)["cells"][1]["source"])
    assert f'"{lines.PEFT}"' in c1_lora and lines.PEFT_TOKEN_ROWS not in c1_lora         # a lane without rows: as before


def test_a_lane_without_smokes_carries_no_full_finetune():
    e = registry.get("ae1b")
    e["no_smoke"] = "measured elsewhere"
    with pytest.raises(lines.LaneError, match="no full fine-tune"):
        lines.build_lane(e, "apache", PROJECT, LANE_DONE)


def test_the_command_writes_the_retrain_s_queue(tmp_path, capsys):
    listing = tmp_path / "files.json"
    listing.write_text(json.dumps(sorted(LANE_DONE)), encoding="utf-8")
    assert lines.main(["ae1bt", "apache", "--project-repo", PROJECT, "--files", str(listing), "--out", str(tmp_path)]) == 0
    q = json.loads((tmp_path / "queue_anima_eyes_v12.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in q] == ["V12_A1_gkl1_1ep_ae1bt_by"] and "every input is there" in capsys.readouterr().out
