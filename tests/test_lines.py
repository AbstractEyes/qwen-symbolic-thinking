"""The lane generator (lines.py) against the lanes it replaces: Lucia v1.2 at 2B and at 9B, each size's Apache-2.0 and -nc lines in
one lane, as uploaded. The fixtures (tests/fixtures/v12_lanes) are the four lane drafts as drafted by hand; the launch merged each
size's two (the Apache-2.0 line's queue, then the -nc line's) and uploaded the merged pair unchanged. Generated with `both` for
the same project repo, given the project repo's files as they stood at the launch, the queue equals the uploaded lane spec for
spec but for these differences, each one checked here:

| difference | why |
|---|---|
| every top-level spec: "needs" 0.1.67 -> the generating package's version | the keys below are honored from 0.1.68; a lane on 0.1.67 would ignore them and train past a failed smoke |
| each smoke: "stop_if" = the registry's limits (1.25 x the pace basis, 85 GB, after step 10) | the stop rule the v1.2 lanes applied by hand, carried by the spec |
| each training: "waits_for_runs" = [its smoke] | a smoke past its limits holds its training back; without it the lane would train next |
| each training's train keys: "progress_every_min" 10 | the progress file |

The project repo is read from the fixtures themselves (where their reads write), so no repo name is written here."""
import copy
import json
import os

import pytest

from qwen_symbolic_thinking import __version__, lines, registry

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "v12_lanes")
UPLOADED_NEEDS = "0.1.67"


def _load(name: str) -> list:
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def uploaded(size: str) -> list:
    """A size's lane as uploaded: its Apache-2.0 draft, then its -nc draft (the launch's merge)."""
    return _load(f"queue_lucia_v12_{size}.json") + _load(f"queue_lucia_v12_{size}_nc.json")


PROJECT = next(s["extract"]["out_repo"] for s in _load("queue_lucia_v12_9b.json") if s["kind"] == "extract")


def files_at_launch() -> set:
    """The project repo's files the generator asks about, as they stood at the launch: every project-repo input the four lanes
    read and every adapter of the project repo they compare against, but not what the lanes made themselves (the 2B's general
    answers, made first in its lane)."""
    specs = [s for name in sorted(os.listdir(FIX)) for s in _load(name)]
    units = specs + [a for s in specs for a in s.get("after") or []]
    paths, made = set(), set()
    for u in units:
        paths.update(p.strip("/") for p in u.get("waits_for") or [])
        paths.update(lines._paths({k: v for k, v in u.items() if k not in ("after", "waits_for")}))
        body = u.get("extract") or (u.get("tool") or {}).get("args") or {}
        if body.get("adapter") == PROJECT and body.get("adapter_subfolder"):
            paths.add(body["adapter_subfolder"] + "/adapter_model.safetensors")
        if u["kind"] == "tool" and u["tool"]["module"] in ("anchors", "filter_anchors"):
            made.update(f"anchors/{u['name']}/{f}" for f in ("anchors.json", "held.json", "summary.json"))
    return {p for p in paths if p.startswith(lines.PROJECT_FOLDERS)} - made


def without_listed(spec: dict, entry: dict) -> dict:
    """A generated spec with the listed differences checked and taken out."""
    s = copy.deepcopy(spec)
    assert s["needs"] == __version__, s["name"]
    s["needs"] = UPLOADED_NEEDS
    if s["name"].startswith("S_v12_smoke_gc1_"):
        line = "nc" if s["name"].endswith("_nc") else "apache"
        assert s.pop("stop_if") == registry.smoke_stop(entry, line), s["name"]
    elif s["kind"] == "train":
        line = "nc" if s["name"].startswith("V12NC_") else "apache"
        assert s.pop("waits_for_runs") == [lines.names(entry, line)["smoke"]], s["name"]
        assert s["train"].pop("progress_every_min") == 10, s["name"]
    assert "stop_if" not in s and "waits_for_runs" not in s, s["name"]
    return s


def _diff(a, b, where=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += _diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{where}{k}.")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and any(isinstance(x, dict) for x in a):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{where}{i}.")]
    return [] if a == b else [f"{where[:-1]}: generated {json.dumps(a)} | uploaded {json.dumps(b)}"]


@pytest.mark.parametrize("size", ["2b", "9b"])
def test_both_lines_equal_the_uploaded_lane_spec_for_spec(size):
    entry = registry.get(size)
    built = lines.build_lane(entry, "both", PROJECT, files_at_launch())
    want = uploaded(size)
    assert built["lane"] == f"lucia_v12_{size}"
    assert [s["name"] for s in built["specs"]] == [s["name"] for s in want]
    for got, exp in zip(built["specs"], want):
        norm = without_listed(got, entry)
        assert norm == exp, f"{exp['name']}: " + "; ".join(_diff(norm, exp))
    assert built["problems"] == [] and built["missing"] == []      # every input there or made earlier; every name free


@pytest.mark.parametrize("size", ["2b", "9b"])
def test_each_line_alone_equals_its_draft(size):
    entry, files = registry.get(size), files_at_launch()
    apache = lines.build_lane(entry, "apache", PROJECT, files)
    assert apache["lane"] == f"lucia_v12_{size}"
    assert [without_listed(s, entry) for s in apache["specs"]] == _load(f"queue_lucia_v12_{size}.json")
    # the -nc lane alone: the Apache-2.0 lane has made the general answers by then, and the comparisons ride with that lane
    ga = entry["general_answers"]
    made = {f"{ga['anchors']}/anchors.json", f"{ga['anchors']}/held.json", f"{ga['text']}/held.json"}
    nc = lines.build_lane(entry, "nc", PROJECT, files | made)
    assert nc["lane"] == f"lucia_v12_{size}_nc"
    assert [without_listed(s, entry) for s in nc["specs"]] == _load(f"queue_lucia_v12_{size}_nc.json")


def test_the_general_answers_follow_the_project_repo():
    files = files_at_launch()
    two = registry.get("2b")
    ga = two["general_answers"]
    have = files | {f"{ga['anchors']}/anchors.json", f"{ga['anchors']}/held.json", f"{ga['anchors_by']}/anchors.json",
                    f"{ga['text']}/held.json"}
    names = [s["name"] for s in lines.build_lane(two, "both", PROJECT, have)["specs"]]
    assert not {"v11a_2b", "v11a_2b_by", "v11txt_2b"} & set(names) and names[0] == "S_v12_smoke_gc1_2b"
    nine = registry.get("9b")
    bare = {f for f in files if not f.startswith("anchors/v11")}
    built = lines.build_lane(nine, "both", PROJECT, bare)
    assert [s["name"] for s in built["specs"][:3]] == ["v11a_9b", "v11a_9b_by", "v11txt_9b"]
    assert built["specs"][0]["tool"]["args"]["model_id"] == "Qwen/Qwen3.5-9B" and built["missing"] == []
    nc_only = lines.build_lane(nine, "nc", PROJECT, bare)                  # the -nc line needs no Apache-2.0 filter
    assert [s["name"] for s in nc_only["specs"][:2]] == ["v11a_9b", "v11txt_9b"]


def test_a_missing_input_and_a_taken_name_are_reported():
    files = files_at_launch()
    built = lines.build_lane(registry.get("9b"), "both", PROJECT, (files - {"sft/v12_rows_nc.json"}) | {"runs/done/XTA_base_9b.json"})
    assert any(m.startswith("sft/v12_rows_nc.json") for m in built["missing"])
    assert any("XTA_base_9b" in p for p in built["problems"])


def test_the_lines_the_catalogue_gives_and_the_blockers():
    with pytest.raises(lines.LaneError, match="Apache"):
        lines.build_lane(registry.get("north"), "both", PROJECT, set())
    with pytest.raises(lines.LaneError, match="cannot run as generated"):
        lines.build_lane(registry.get("phi4rv"), "apache", PROJECT, set())
    forced = lines.build_lane(registry.get("phi4rv"), "apache", PROJECT, set(), force=True)
    assert any(c.startswith("FORCED past a blocker") for c in forced["choices"])
    assert all(s.get("transformers") == "==4.57.6" for s in forced["specs"])           # every spec on its transformers lane
    t = next(s for s in forced["specs"] if s["kind"] == "train" and not s["name"].startswith("S_"))["train"]
    assert t["trust_remote_code"] and t["revision"] == registry.get("phi4rv")["base_revision"]
    assert t["image_settings"] == {"max_num_patches": 1024} and t["max_len"] == 2048
    reads = [s["extract"] for s in forced["specs"] if s["kind"] == "extract"]
    assert reads and all(r["trust_remote_code"] and r["image_settings"] == {"max_num_patches": 1024} for r in reads)
    with pytest.raises(lines.LaneError):
        lines.build_lane(registry.get("2b"), "both", "no-owner", set())


def test_the_08b_lane_builds_with_its_estimates_and_the_default_comparisons():
    entry = registry.get("08b")
    built = lines.build_lane(entry, "both", "someone/their-project", set(lines.PROJECT_FILES))
    names = [s["name"] for s in built["specs"]]
    assert names[:5] == ["v11a_08b", "v11a_08b_by", "v11txt_08b", "S_v12_smoke_gc1_08b", "V12_A1_gkl1_1ep_08b_by"]
    assert names[-2:] == ["S_v12_smoke_gc1_08b_nc", "V12NC_A1_gkl1_1ep_08b"]
    assert names[5:-2] == ["XT_base_08b_noadapter", "XT_base_08b_deep", "XTA_base_08b", "XCA2_base_08b", "XC2_base_08b",
                           "K0_lucia1_08b", "XT_v1_08b_s1_deep", "XT_v1_08b_s1"]
    smoke = built["specs"][3]
    assert smoke["stop_if"] == {"steady_s_per_step": round(1.25 * 2.1, 2), "peak_gb": 85, "after_step": 10}
    assert built["specs"][-2]["stop_if"]["steady_s_per_step"] == round(1.25 * 1.9, 2)
    k0 = built["specs"][10]["tool"]["args"]
    assert k0["adapter"] == registry.RECORD and k0["adapter_subfolder"].endswith("_G_med4_2ep_08b/final")
    assert all(s.get("needs") == __version__ for s in built["specs"]) and built["problems"] == [] and built["missing"] == []
    assert any(c.startswith("comparisons: none recorded") for c in built["choices"])
    # a project repo that holds none of the inputs: each one is named as missing, nothing is a problem
    bare = lines.build_lane(entry, "both", "someone/their-project", set())
    assert {m.split(" ")[0] for m in bare["missing"]} == set(lines.PROJECT_FILES) and bare["problems"] == []


def test_older_versions_in_another_project_repo_are_left_out():
    built = lines.build_lane(registry.get("9b"), "both", "someone/their-project", set(lines.PROJECT_FILES) |
                             {f for f in files_at_launch() if f.startswith("anchors/")})
    names = [s["name"] for s in built["specs"]]
    assert "XT_base_9b_deep" in names and not any("v11" in n and n.startswith(("XT_", "XTA_", "XC")) for n in names)
    assert sum("not in the project repo" in c for c in built["choices"]) == 2


def test_the_notebook_matches_the_v12_notebooks_but_the_pin():
    root = os.path.dirname(os.path.dirname(__file__))
    for size in ("2b", "9b"):
        entry = registry.get(size)
        built = lines.build_lane(entry, "both", PROJECT, files_at_launch())
        nb = lines.notebook(entry, built, PROJECT)
        with open(os.path.join(root, "notebooks", f"lucia_v12_{size}.ipynb"), encoding="utf-8") as fh:
            live = json.load(fh)
        assert [c["cell_type"] for c in nb["cells"]] == [c["cell_type"] for c in live["cells"]]
        assert {k: v for k, v in nb.items() if k != "cells"} == {k: v for k, v in live.items() if k != "cells"}
        assert "".join(nb["cells"][1]["source"]) == "".join(live["cells"][1]["source"]).replace("@v0.1.67", f"@v{__version__}")

        def statements(cell):
            return [ln.split("#")[0].rstrip() for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith("#")]
        assert statements(nb["cells"][2]) == statements(live["cells"][2])
        md = "".join(nb["cells"][0]["source"])
        assert entry["base"] in md and f"notebooks/lucia_v12_{size}.ipynb" in md and PROJECT in md


def test_the_command_writes_the_queue_and_the_notebook(tmp_path, capsys):
    listing = tmp_path / "files.json"
    listing.write_text(json.dumps(sorted(files_at_launch())), encoding="utf-8")
    code = lines.main(["2b", "both", "--project-repo", PROJECT, "--out", str(tmp_path / "out"), "--files", str(listing)])
    out = capsys.readouterr().out
    assert code == 0 and "nothing uploaded" in out and "every input is there" in out
    q = json.loads((tmp_path / "out" / "queue_lucia_v12_2b.json").read_text(encoding="utf-8"))
    assert [without_listed(s, registry.get("2b")) for s in q] == uploaded("2b")
    nb = json.loads((tmp_path / "out" / "lucia_v12_2b.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 3
    assert lines.main(["north", "nc", "--project-repo", PROJECT, "--files", str(listing), "--out", str(tmp_path)]) == 2


def test_the_package_names_no_repo_that_is_not_public():
    """The registry, the generator and the progress file take the project repo as a parameter; the repo the fixtures name is
    in none of them."""
    pkg = os.path.dirname(registry.__file__)
    for rel in ("registry.py", "lines.py", os.path.join("tools", "progress.py"), "queue.py"):
        with open(os.path.join(pkg, rel), encoding="utf-8") as fh:
            text = fh.read()
        assert PROJECT not in text and "-private" not in text, rel
