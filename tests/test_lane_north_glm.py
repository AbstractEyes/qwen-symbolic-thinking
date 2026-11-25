"""The v1.2 lanes of North-Micro-Vision and GLM-4.6V-Flash (Apache-2.0 line only): built from the registry like every lane, with
their general rows' text requests from the cleaned requests file (registry "general_requests"), so the base model answers
requests written for it instead of writing its own."""
import pytest

from qwen_symbolic_thinking import __version__, lines, registry

PROJECT = "someone/their-project"
CLEAN = "text/requests_qwen_v1_clean.json"
FILES = set(lines.PROJECT_FILES) | {CLEAN}


def built(key, files=FILES):
    return lines.build_lane(registry.get(key), "apache", PROJECT, files)


@pytest.mark.parametrize("key", ["north", "glm46v"])
def test_the_lane_builds_with_nothing_missing(key):
    b = built(key)
    assert b["problems"] == [] and b["missing"] == [] and b["lane"] == f"{key}_v12"
    assert all(s["needs"] == __version__ for s in b["specs"])
    names = [s["name"] for s in b["specs"]]
    assert names[:3] == [f"v11a_{key}", f"v11a_{key}_by", f"v11txt_{key}"]
    assert f"S_v12_smoke_gc1_{key}" in names and f"V12_A1_gkl1_1ep_{key}_by" in names
    bare = built(key, set(lines.PROJECT_FILES))
    assert [m.split(" ")[0] for m in bare["missing"]] == [CLEAN]


@pytest.mark.parametrize("key", ["north", "glm46v"])
def test_the_general_rows_ask_the_cleaned_requests(key):
    spec = built(key)["specs"][0]
    g = spec["tool"]["args"]
    assert g["model_id"] == registry.get(key)["base"] and g["requests"] == f"hub:{CLEAN}"
    assert (g["n_picture"], g["n_text"], g["n_held_picture"], g["n_held_text"]) == (4000, 500, 200, 40)
    assert spec["waits_for"][-1] == CLEAN
    assert f"general text requests from hub:{CLEAN} (500 training, 40 held)" in registry.describe(registry.get(key))


def test_the_other_version_1_lanes_still_write_their_own_requests():
    for key in ("9b", "4b", "2b", "08b", "ge4b", "g12b", "molmo2o", "phi4rv", "mi14"):
        assert "general_requests" not in registry.get(key), key
