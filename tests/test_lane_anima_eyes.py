"""The lane of Anima's eyes (registry.LATER["ae1b"]; lines.py's prepare run, requests file, family train keys and full
fine-tune): built from the registry like every lane, with the base made by its first run, the general rows' text requests from
the cleaned requests file, a LoRA and a full fine-tune each after its own smoke, and every run waiting for the base."""
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
