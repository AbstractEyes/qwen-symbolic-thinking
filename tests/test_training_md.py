"""The TRAINING.md generator: a page for every model of the registry, from the registry and the one template; every file with
no public source is a marked placeholder and listed; nothing on a page names a repo that is not public."""
import json
import os

import pytest

from qwen_symbolic_thinking import lines, registry, resume, training_md


@pytest.mark.parametrize("key", registry.keys())
def test_every_model_gets_a_page_from_its_registry_entry(key):
    e = registry.get(key)
    text, placeholders = training_md.render(e, "9.9.9")
    assert text.startswith(f"# Training {e['name']} v1.2\n") and f"](https://huggingface.co/{e['base']})" in text
    assert "@v9.9.9" in text and f"python -m qwen_symbolic_thinking.lines {key} " in text
    line = "<apache|nc>" if len(e["lines"]) > 1 else "apache"
    assert f"python -m qwen_symbolic_thinking.reads {key} {line} bars --project-repo" in text
    for line in e["lines"]:
        s = registry.V12_SETS[line]
        assert f"`{s['path']}`" in text and f"{s['items']:,} items" in text and e["licenses"][line] in text
        p = e["pace"][line]
        assert f"{p['s']} s; {p['source']}" in text and f"about {registry.hours(e, line)}" in text
        assert f"above {registry.smoke_stop(e, line)['steady_s_per_step']} s a step" in text
    files = training_md.project_files(e)
    for path in files:
        assert f"`{path}`" in text
    for line in ("apache", "nc"):          # a line the model does not get: its training set is not on the page
        if line not in e["lines"]:
            assert f"`{registry.V12_SETS[line]['path']}`" not in text
    # every file without a public source is marked on the page and listed; the Apache lanes' inputs are public (2026-11-27)
    assert text.count("**not published**") == len(placeholders)
    listed = [w for w, _ in placeholders]
    assert {p for p in files if training_md.SOURCES[p][0] is None} <= set(listed)
    assert listed[:1] == (["sft/v12_rows_nc.json"] if "nc" in e["lines"] else ["the yardstick's records"])
    assert "sft/v12_rows_a_by.json" not in listed and "screens/inputs/screen_v1.json" not in listed
    assert "[PLACEHOLDER" not in text
    assert f"after\n   {resume.LATE_SAVE_H:g} hours of the runtime" in text
    assert "|  |" not in text.split("## The settings")[1].split("##")[0]                     # no empty setting


def test_the_models_we_do_not_train_say_so_with_what_stops_them():
    for key in ("molmo2o", "phi4rv", "mi14"):
        e = registry.get(key)
        text, _ = training_md.render(e)
        assert f"We do not train {e['name']} at v1.2: this page is for anyone who wants to." in text
        assert "not trained by us" in text
        for b in e["blockers"]:
            assert b in text
    assert "Nothing in the package stops its lane" in training_md.render(registry.get("mi14"))[0]
    assert "--force" in training_md.render(registry.get("phi4rv"))[0]
    trained, _ = training_md.render(registry.get("north"))
    assert trained.startswith("# Training North-Micro-Vision v1.2\n\nWe train North-Micro-Vision at v1.2 on the Apache-2.0 line.")


def test_our_numbers_and_the_yardstick_appear_where_we_have_them():
    two, _ = training_md.render(registry.get("2b"))
    assert "| T2 said | within 0.03 of the yardstick (above the band is never a miss) | 0.209 / 0.241 | 0.217 / 0.231 |" in two
    assert "| the verdict | every bar passes | - / - | PASS / PASS |" in two
    assert "On a G4 the three took 30.4 minutes (measured:" in two
    small, _ = training_md.render(registry.get("08b"))
    assert "| T3 probe (of 500 held-out pictures) | within 10 of the yardstick (above the band is never a miss) | 143 / 134 | 136 / 155 |" in small
    glm, _ = training_md.render(registry.get("glm46v"))
    assert "the trained adapter follows its base model's license, MIT" in glm and "learning rate | 2e-4" in glm
    assert "learning rate | 5e-5" in training_md.render(registry.get("g12b"))[0]


def test_a_model_with_a_requests_file_lists_it_with_its_own_counts():
    # the two faults of 2026-11-27: the requests file was left out of the files table, and the general answers' counts were
    # the defaults (2,000 + 200) where the lane asks 500 + 40 from the file
    for key in ("north", "glm46v"):
        text, _ = training_md.render(registry.get(key))
        assert f"`{training_md.REQUESTS}`" in text.split("## What your project repo must hold")[1].split("## The steps")[0]
        assert ("4,000 requests\n   about pictures and 500 plain requests from `text/requests_qwen_v1_clean.json` (200 + 40 more "
                "held out)") in text
    plain, _ = training_md.render(registry.get("2b"))
    assert f"`{training_md.REQUESTS}`" not in plain and "2,000 plain requests (200 + 200 more held out)" in plain


def test_the_gemma_pages_name_the_requests_their_lanes_asked_and_how_ours_differed():
    # the three holds of 2026-11-27: the Gemma pages described a lane that writes its own requests, and the E4B's bars
    # command stopped on a preservation read without text asks
    for key in ("ge4b", "g12b"):
        text, placeholders = training_md.render(registry.get(key))
        files = text.split("## What your project repo must hold")[1].split("## The steps")[0]
        assert f"`{training_md.REQUESTS_GEMMA}`" in files and f"`{training_md.REQUESTS}`" not in files
        assert "less two requests the public rows release's word rule holds back" in files
        assert ("4,000 requests\n   about pictures and 2,000 plain requests from `text/requests_qwen_v1.json` (200 + 200 more "
                "held out)") in text
        assert [w for w, _ in placeholders] == ["sft/v12_rows_nc.json", "the yardstick's records",
                                               "text/text_held_v1_names.jsonl"], key
    e4b, _ = training_md.render(registry.get("ge4b"))
    assert "Our Apache-2.0 version was trained before this file existed" in e4b
    assert "on the preservation read K_v12_A1_ge4b_by_t (its held text requests asked apart" in e4b
    assert "`--keep K_v12_A1_ge4b_by_t` reads it" in e4b and "-nc: V12NC_A1_gkl1_1ep_ge4b, read 2026-11-23." in e4b
    twelve, _ = training_md.render(registry.get("g12b"))
    assert "trained before this file existed" not in twelve and "--keep" not in twelve


def test_the_anima_eyes_page_is_its_full_fine_tune_with_the_lora_beside_it():
    # held 2026-11-27: the page called it a LoRA of "hub:base/..." (a link to nowhere) with the LoRA's pace; its release is the
    # full fine-tune, of a starting model the lane's first run builds
    text, placeholders = training_md.render(registry.get("ae1b"))
    assert "hub:base" not in text and "huggingface.co/hub:" not in text
    assert "Anima Eyes 1B v1.2 is a full fine-tune of Anima's eyes: [OpenGVLab/InternVL3_5-1B-HF]" in text
    assert "as its output head, on the rows of the public dataset" in text
    assert "| what trains | the full fine-tune: every weight but those its family keeps frozen (below); the LoRA beside it: " \
           "rank 32, alpha 64 |" in text and "| learning rate | 2e-5 (the full fine-tune); 2e-4 (the LoRA) |" in text
    assert "| Apache-2.0 (the full fine-tune) | 1.87 s; measured:" in text and "| Apache-2.0 (the LoRA) | 2.01 s; measured:" in text
    assert "`base_ae1b` builds the starting model and writes it to `base/InternVL3_5-1B-anima-last4/`" in text
    assert "The full fine-tune runs first, right after the two smokes" in text and "The LoRA's whole state saves" in text
    assert "the full fine-tune's smoke, a steady pace above 2.34 s a step after step 10" in text
    assert "python -m qwen_symbolic_thinking.reads ae1bf apache bars --project-repo <you>/<project> --out reads" in text
    assert "it misses T2 backed (0.194, the band's floor 0.195) and T3 probe (118, the floor 133)" in text
    assert ("it misses T1 full rows (0.017, the mark 0.95), T2 backed (0.186, the band's floor 0.195) and T3 probe (125, the "
            "floor 133)") in text
    assert "| the verdict | every bar passes | - | NOT READY |" in text
    assert [w for w, _ in placeholders] == ["the yardstick's records", "text/text_held_v1_names.jsonl"]
    two, _ = training_md.render(registry.get("2b"))                               # the LoRA pages keep their form
    assert "is a LoRA fine-tune of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) on the rows" in two
    assert "| LoRA | rank 32, alpha 64 |" in two and "The full fine-tune runs first" not in two and "Its whole state saves" in two
    assert registry.get("ae1bf")["is_full"] and registry.get("ae1bf")["key"] == "ae1bf"


def test_no_page_names_a_repo_that_is_not_public():
    fix = os.path.join(os.path.dirname(__file__), "fixtures", "v12_lanes", "queue_lucia_v12_9b.json")
    with open(fix, encoding="utf-8") as fh:
        private = next(s["extract"]["out_repo"] for s in json.load(fh) if s["kind"] == "extract")
    for key in registry.keys():
        text, placeholders = training_md.render(registry.get(key))
        both = text + json.dumps(placeholders)
        assert private not in both and "-private" not in both and "twin" not in both, key
    with open(training_md.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert private not in source and "-private" not in source and "twin" not in source


def test_the_command_writes_the_page_and_lists_its_placeholders(tmp_path, capsys):
    out = tmp_path / "ulisse" / "TRAINING.md"
    assert training_md.main(["ge4b", "--out", str(out)]) == 0
    said = capsys.readouterr().out
    assert said.startswith(f"wrote {out} (Ulisse E4B;") and "NOT PUBLISHED sft/v12_rows_nc.json:" in said
    assert "NOT PUBLISHED sft/v12_rows_a_by.json" not in said
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Training Ulisse E4B v1.2") and b"\r\n" not in out.read_bytes()
