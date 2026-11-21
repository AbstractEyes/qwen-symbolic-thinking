"""The model registry: every model of the version-1 list with every field, every measured or estimated number with its source,
every open value named in the entry's todo, and the catalogue's lines (both lines for the Lucia and Ulisse models; the
Apache-2.0 line for the rest, three of them as training instructions only)."""
import pytest

from qwen_symbolic_thinking import registry

V1_LIST = {"9b": "Qwen/Qwen3.5-9B", "4b": "Qwen/Qwen3.5-4B", "2b": "Qwen/Qwen3.5-2B", "08b": "Qwen/Qwen3.5-0.8B",
           "ge4b": "google/gemma-4-E4B-it", "g12b": "google/gemma-4-12B-it", "north": "CohereLabs/North-Micro-Vision-Instruct",
           "glm46v": "zai-org/GLM-4.6V-Flash", "molmo2o": "allenai/Molmo2-O-7B", "phi4rv": "microsoft/Phi-4-reasoning-vision-15B",
           "mi14": "mistralai/Ministral-3-14B-Instruct-2512-BF16"}


def test_every_model_of_the_v1_list_has_every_field():
    assert {k: e["base"] for k, e in registry.ENTRIES.items()} == V1_LIST
    for key, e in registry.ENTRIES.items():
        assert set(e) == set(registry.FIELDS), (key, set(e) ^ set(registry.FIELDS))
        assert e["key"] == key and e["lane"] and e["general_answers"]["anchors"].startswith("anchors/")
        assert set(e["lines"]) <= set(registry.LINES) and set(e["pace"]) == set(e["lines"]) == set(e["peak"])
        assert set(e["licenses"]) == set(e["lines"])
        for line in e["lines"]:
            assert e["pace"][line]["s"] > 0 and e["pace"][line]["source"], (key, line)
            assert e["pace"][line]["source"].startswith("measured" if e["pace"][line]["measured"] else "estimate"), (key, line)
            assert e["peak"][line]["source"], (key, line)
        assert e["versions"] and all(v["adapter"] in (registry.RECORD, registry.PROJECT) and v["subfolder"].startswith("lora/")
                                     for v in e["versions"]), key
        assert all(v["source"] for v in e["versions"]), key
        assert e["space"]["gb"] > 0 and e["space"]["tps"] > 0 and e["space"]["lookers"], key
        assert (e["base_revision"] is not None) == e["trust_remote_code"], key      # a repository's code runs pinned
        assert (e["transformers"] == "==4.57.6") == e["trust_remote_code"], key
        assert e["install"]["fast_path"] == (e["family"] == "Qwen3.5"), key         # the kernels are Qwen3.5's linear attention


def test_every_open_value_is_named_with_what_decides_it():
    for key, e in registry.ENTRIES.items():
        todo = " | ".join(e["todo"])
        for line in e["lines"]:
            if e["peak"][line]["gb"] is None:
                assert "peak" in todo, (key, line)
            if e["repos"][line] is None:
                assert f"repos {line}" in todo, (key, line)
        if e["comparisons"] is None:
            assert "comparisons" in todo, key
        for v in e["versions"]:
            if not v["ready"] or (v["release"] is None):
                assert f"versions {v['version']} {v['line']}" in todo, (key, v["version"], v["line"])
    assert len(registry.todo()) == sum(len(e["todo"]) for e in registry.ENTRIES.values())


def test_the_catalogue_lines():
    both, apache = ("apache", "nc"), ("apache",)
    for key in ("9b", "4b", "2b", "08b", "ge4b", "g12b"):
        assert registry.ENTRIES[key]["lines"] == both and registry.ENTRIES[key]["scheduled"], key
    for key in ("north", "glm46v"):
        assert registry.ENTRIES[key]["lines"] == apache and registry.ENTRIES[key]["scheduled"], key
    for key in ("molmo2o", "phi4rv", "mi14"):
        assert registry.ENTRIES[key]["lines"] == apache and not registry.ENTRIES[key]["scheduled"], key
    assert registry.ENTRIES["molmo2o"]["blockers"] and registry.ENTRIES["phi4rv"]["blockers"]
    assert not any(registry.ENTRIES[k]["blockers"] for k in ("9b", "4b", "2b", "08b", "ge4b", "g12b", "north", "glm46v", "mi14"))


def test_the_recorded_settings():
    assert registry.ENTRIES["g12b"]["lr"] == 5e-5                      # the encoder-free 12B's two-seed lesson
    assert all(registry.ENTRIES[k]["lr"] == 2e-4 for k in registry.ENTRIES if k != "g12b")
    assert {k for k, e in registry.ENTRIES.items() if e["max_len"] == 2560} == {"glm46v", "molmo2o", "mi14"}
    assert registry.ENTRIES["phi4rv"]["image_settings"] == {"max_num_patches": 1024}
    assert registry.ENTRIES["molmo2o"]["read_batch"] == 32 and registry.ENTRIES["molmo2o"]["base_revision"].startswith("78441065")
    assert registry.ENTRIES["phi4rv"]["base_revision"].startswith("c3e4fac7")
    assert all(e["lora"] == {"rank": 32, "alpha": 64} and (e["batch"], e["grad_accum"]) == (8, 2) for e in registry.ENTRIES.values())
    # the measured v1.2 paces and the estimates of the others
    paces = {k: {ln: e["pace"][ln]["s"] for ln in e["lines"]} for k, e in registry.ENTRIES.items()}
    assert paces["9b"] == {"apache": 9.03, "nc": 8.1} and paces["4b"] == {"apache": 6.23, "nc": 5.6}
    assert paces["2b"] == {"apache": 3.18, "nc": 2.87} and paces["08b"] == {"apache": 2.1, "nc": 1.9}
    assert paces["ge4b"] == {"apache": 4.1, "nc": 3.7} and paces["g12b"] == {"apache": 7.0, "nc": 6.3}
    assert (paces["north"], paces["glm46v"], paces["molmo2o"], paces["phi4rv"], paces["mi14"]) == (
        {"apache": 3.6}, {"apache": 14.4}, {"apache": 11.5}, {"apache": 21.8}, {"apache": 31.6})


@pytest.mark.parametrize("key", list(registry.ENTRIES))
def test_the_smoke_limits_follow_the_stop_rule(key):
    e = registry.get(key)
    for line in e["lines"]:
        stop = registry.smoke_stop(e, line)
        factor = 1.25 if e["pace"][line]["measured"] else 1.5
        assert stop == {"steady_s_per_step": round(factor * e["pace"][line]["s"], 2), "peak_gb": 85, "after_step": 10}
        assert registry.hours(e, line) == round(registry.V12_SETS[line]["steps"] * e["pace"][line]["s"] / 3600, 1)
    assert registry.describe(e).startswith(f"{key}: ")


def test_lookups_by_key_base_repo_and_looker():
    assert registry.get("2b")["key"] == "2b" and registry.get("Qwen/Qwen3.5-0.8B")["key"] == "08b"
    assert registry.get("AbstractPhil/gemma-4-12b-think-symbolic-image")["key"] == "g12b"
    assert registry.get("ulisse-e4b")["key"] == "ge4b" and registry.get("LUCIA-9B-V1.2")["key"] == "9b"
    with pytest.raises(KeyError):
        registry.get("qwen3-vl")
    e = registry.get("2b")
    e["lines"] = ()
    assert registry.ENTRIES["2b"]["lines"] == ("apache", "nc")            # a copy: the registry itself is not changed


def test_the_recipe_and_the_versions():
    r = registry.recipe(registry.get("phi4rv"))
    assert list(r)[:4] == ["model_id", "trust_remote_code", "revision", "image_settings"] and r["epochs"] == 1
    assert registry.writers(registry.get("2b"))[-1] == "Qwen/Qwen3.5-2B" and len(registry.writers(registry.get("2b"))) == 5
    v1 = registry.version(registry.get("2b"), "v1")
    assert v1["adapter"] == registry.RECORD and v1["keep_tag"] == "lucia1_2b" and v1["release"]["tag"] == "lucia-v1"
    assert registry.version(registry.get("9b"), "v1.1", "nc")["adapter"] == registry.PROJECT
    assert not registry.version(registry.get("9b"), "v1.2", "nc")["ready"]
    with pytest.raises(KeyError):
        registry.version(registry.get("08b"), "v1.1")


def test_the_command_prints_the_entries(capsys):
    registry.main(["2b", "molmo2o"])
    out = capsys.readouterr().out
    assert out.startswith("2b: Lucia 2B (Qwen3.5, qwen3_5) on Qwen/Qwen3.5-2B") and "BLOCKER the general answers" in out
