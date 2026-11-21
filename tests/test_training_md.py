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
    for path in lines.PROJECT_FILES:
        assert f"`{path}`" in text
    # every placeholder printed on the page is listed, and every file without a public source is one
    assert text.count("[PLACEHOLDER:") == len(placeholders)
    listed = [w for w, _ in placeholders]
    assert listed[:2] == ["sft/v12_rows_a_by.json", "sft/v12_rows_nc.json"]
    assert {p for p, (public, _) in training_md.SOURCES.items() if public is None} <= set(listed)
    assert "screens/inputs/screen_v1.json" not in listed
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
    assert "| T3 probe (of 500 held-out pictures) | within 10 of the yardstick (above the band is never a miss) | 143 / 134 | not read yet / not read yet |" in small
    glm, _ = training_md.render(registry.get("glm46v"))
    assert "the trained adapter follows its base model's license, MIT" in glm and "learning rate | 2e-4" in glm
    assert "learning rate | 5e-5" in training_md.render(registry.get("g12b"))[0]


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
    assert said.startswith(f"wrote {out} (Ulisse E4B;") and "PLACEHOLDER sft/v12_rows_a_by.json:" in said
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Training Ulisse E4B v1.2") and b"\r\n" not in out.read_bytes()
