"""The lane generator: a model's v1.2 lane, its queue and its notebook, built from its registry entry (registry.py).

    python -m qwen_symbolic_thinking.lines <model key> <apache|nc|both> --project-repo <owner/name> [--out <folder>]
        [--files <a JSON list of the project repo's paths>] [--lane <name>] [--progress-every-min 10]
        [--comparisons | --no-comparisons] [--force]

writes <out>/queue_<lane>.json and <out>/<lane>.ipynb, prints every choice it made, and checks the queue against the project
repo's files (listed from the Hub with HF_TOKEN, or passed with --files). Nothing is uploaded: putting the queue in the project
repo as runs/queue_<lane>.json starts the lane on any card that runs the notebook.

The queue, in order:
1. the base model's own answers, when the project repo lacks them (the keeping objective holds a model near its base ON THE
   BASE'S OWN ANSWERS, so each model needs its own): its answers to ordinary requests, with and without a picture
   (tools.anchors -> anchors/v11a_<key>/: anchors.json for training, held.json held out), those filtered to the Apache-2.0
   line's picture licenses (tools.filter_anchors -> anchors/v11a_<key>_by/), and its answers to the held requests about
   pictures with lettering (anchors/v11txt_<key>/);
2. per line, the smoke and the training. The smoke trains 30 steps on the set's first 2,000 items, ships nothing and carries
   the registry's stop limits ("stop_if"): past them its marker says failed and the training, which waits for it
   ("waits_for_runs"), does not start. The training is the v1.2 recipe with the model's own settings, writes its progress file
   ("progress_every_min"), is screened, and is followed by nine reads of what it shipped (the after chain): the probe, the
   museum read, the preservation read, the closer text read, the text preservation read, the long text read, the lettering
   read and the two controls (photos without lettering, with and without the lettering sentence);
3. after the first line (or not at all: --no-comparisons; an -nc lane alone leaves them to the Apache-2.0 lane unless
   --comparisons): the comparison reads its cards need, of the base model and of the older versions the registry names.

The project repo is a parameter. Everything the lane reads is a path in it (PROJECT_FILES; the run names its general answers,
adapters and reads under it) or in a public repo (the picture lists of the probe and museum reads, the version-1 adapters in
the public record repo); nothing here names a repo that is not public, so anyone can run a lane on a repo of their own once
those files are in it. Every top-level spec says "needs": the package version that generated it, the first that honors
stop_if, waits_for_runs and progress_every_min."""
from __future__ import annotations

import copy
import json
import os
import sys

from . import __version__, registry
from .registry import PROJECT, V12_SETS

# The files a v1.2 lane reads from the project repo (paths in it), and what each holds. The two training sets come from the
# public dataset's release 5 split by license; the rest are the inputs of the general answers and of the text reads.
PROJECT_FILES = {
    V12_SETS["apache"]["path"]: "the Apache-2.0 line's training set: release 5's rows whose pictures may be used that way",
    V12_SETS["nc"]["path"]: "the -nc line's training set: every row of release 5",
    "sft/v11_rows_a.json": "the rows whose pictures the general answers ask about (the base answers ordinary requests on them)",
    "sft/v11_rows_a_held.json": "pictures held out of training, for the held general answers",
    "screens/inputs/screen_v1.json": "the screen's 60 pictures and 40 specifications (held out; also the training's screen)",
    "licenses/coco_photo_licenses.json": "each COCO photo's own license (the Apache-2.0 line's filter of the general answers)",
    "text/anchor_pool_v1.json": "pictures with lettering, for the general answers' text bank",
    "text/anchor_held_v1.json": "held pictures with lettering, for the text preservation read",
    "text/text_held_v1.jsonl": "the 300 held pictures with lettering (the closer text read and the lettering read)",
    "text/text_held_v1_deep.jsonl": "the same pictures asked for a long look",
    "text/notext_control_v1.jsonl": "200 photos without lettering (the two controls)",
}
PROJECT_FOLDERS = ("sft/", "anchors/", "text/", "licenses/", "screens/", "lora/", "full/", "keep/", "runs/", "resume/", "synth/",
                   "extractions/", "manifests/")
PICTURES = "AbstractPhil/qwen-thinking-images"      # the public picture lists the probe and museum reads draw from
PROBE = {"manifest": f"hub:{PICTURES}/manifest_v1.4.jsonl", "split": "probe"}
MUSEUM = {"manifest": f"hub:{PICTURES}/manifest_v1.2.jsonl", "split": "main", "sources": ["aic", "nga", "cleveland", "met"],
          "per_source": 50}
LETTERING = " Ensure you include visible text and the expression related to it into the rows."
TEXT_READS = {                                      # kind -> (the read's own keys, its name's prefix and suffix)
    "closer": ({"manifest": "repo:text/text_held_v1.jsonl", "split": "text_held"}, "XT_", ""),
    "deep": ({"manifest": "repo:text/text_held_v1_deep.jsonl", "split": "text_held"}, "XT_", "_deep"),
    "lettering": ({"manifest": "repo:text/text_held_v1.jsonl", "split": "text_held", "ask_suffix": LETTERING}, "XTA_", ""),
    "control_lettering": ({"manifest": "repo:text/notext_control_v1.jsonl", "split": "notext_control", "ask_suffix": LETTERING},
                          "XCA2_", ""),
    "control": ({"manifest": "repo:text/notext_control_v1.jsonl", "split": "notext_control"}, "XC2_", ""),
}
SCREEN = {"forms": ["full", "short", "spec_full"], "batch": 4, "max_new_tokens": 800}
MAX_NEW_TOKENS, CHUNK = 800, 512
OWN_WORDINGS = "v1.1"          # the wordings the v1.2 rows carry at each depth (the whole-row check reads them)
# The general answers: the same requests, pictures, seeds and caps for every model (the 9B's and 2B's specs, but the model).
GENERAL = {"rows": "hub:sft/v11_rows_a.json", "held_pictures": ["hub:screens/inputs/screen_v1.json", "hub:sft/v11_rows_a_held.json"],
           "n_picture": 4000, "n_text": 2000, "n_held_picture": 200, "n_held_text": 200, "max_new_tokens": 512,
           "max_new_tokens_text": 768, "seed": 20261110}
GENERAL_TEXT = {"rows": "hub:text/anchor_pool_v1.json", "held_pictures": "hub:text/anchor_held_v1.json", "n_picture": 1726,
                "n_text": 0, "n_held_picture": 300, "n_held_text": 0, "max_new_tokens": 512, "max_new_tokens_text": 768,
                "seed": 20261113, "bank": "text", "detail": None}
APACHE_FILTER = {"own_license_sources": ["coco", "localized_narratives"], "picture_licenses": "hub:licenses/coco_photo_licenses.json",
                 "licenses": ["CC BY 4.0", "Apache-2.0", "Attribution License", "No known copyright restrictions",
                              "United States Government Work"]}
DEFAULT_COMPARISONS = {"base": ["closer", "deep", "lettering", "control_lettering", "control"],
                       "older": ["keep", "deep", "closer"]}
TAG = {"apache": "by", "nc": "nc"}      # a line's tag in run names
LINE_NAMES = {"apache": "Apache-2.0", "nc": "non-commercial (-nc)"}
WHICH = {"apache": ("apache",), "nc": ("nc",), "both": ("apache", "nc")}


class LaneError(ValueError):
    """A lane that cannot be generated as asked (a line the model does not get, a model with open blockers)."""


def _common(entry: dict, project: str) -> dict:
    return {"batch": entry["read_batch"], "max_new_tokens": MAX_NEW_TOKENS, "chunk": CHUNK, "out_repo": project, "repo_type": "model"}


def _remote(entry: dict) -> dict:
    """The keys a read of a repository-code family carries itself (a read after a training takes them from it)."""
    out = {}
    if entry["trust_remote_code"]:
        out.update(trust_remote_code=True, revision=entry["base_revision"])
    if entry["image_settings"]:
        out["image_settings"] = copy.deepcopy(entry["image_settings"])
    return out


def _top(entry: dict, name: str, kind: str, needs: str) -> dict:
    spec = {"name": name, "kind": kind, "needs": needs}
    if entry["transformers"]:
        spec["transformers"] = entry["transformers"]
    return spec


def names(entry: dict, line: str) -> dict:
    """The run names of a line: the smoke, the training and its nine reads."""
    k, t = entry["key"], TAG[line]
    return {"smoke": f"S_v12_smoke_gc1_{k}" + ("_nc" if line == "nc" else ""),
            "train": f"V12_A1_gkl1_1ep_{k}_by" if line == "apache" else f"V12NC_A1_gkl1_1ep_{k}",
            "reads": [f"P_probe_v12_A1_{k}_{t}", f"P_museum_v12_A1_{k}_{t}", f"K_v12_A1_{k}_{t}", f"XT_v12_{k}_{t}_s1",
                      f"KT_v12_{k}_{t}", f"XT_v12_{k}_{t}_s1_deep", f"XTA_v12_{k}_{t}_s1", f"XCA2_v12_{k}_{t}_s1",
                      f"XC2_v12_{k}_{t}_s1"]}


def general_answer_specs(entry: dict, lines: tuple, files: set, needs: str) -> tuple:
    """The specs that make the base model's own answers the lines need and the project repo lacks, and the choices."""
    ga, specs, choices = entry["general_answers"], [], []
    model, batch = entry["base"], entry["read_batch"]
    a, a_by, txt = ga["anchors"], ga["anchors_by"], ga["text"]
    if {f"{a}/anchors.json", f"{a}/held.json"} <= files:
        choices.append(f"general answers {a}/: in the project repo (not made again)")
    else:
        s = _top(entry, a.split("/")[-1], "tool", needs)
        s["tool"] = {"module": "anchors", "args": {"model_id": model, "rows": GENERAL["rows"],
                                                    "held_pictures": list(GENERAL["held_pictures"]),
                                                    **{k: GENERAL[k] for k in ("n_picture", "n_text", "n_held_picture", "n_held_text")},
                                                    "batch": batch, **{k: GENERAL[k] for k in ("max_new_tokens",
                                                                                               "max_new_tokens_text", "seed")}}}
        specs.append(s)
        choices.append(f"general answers {a}/: missing -> {s['name']} (tools.anchors: {GENERAL['n_picture']} picture and "
                       f"{GENERAL['n_text']} text requests, {GENERAL['n_held_picture']} + {GENERAL['n_held_text']} held, "
                       f"batch {batch}, seed {GENERAL['seed']})")
    if "apache" in lines:
        if f"{a_by}/anchors.json" in files:
            choices.append(f"general answers {a_by}/ (the Apache-2.0 filter): in the project repo")
        else:
            s = _top(entry, a_by.split("/")[-1], "tool", needs)
            s["tool"] = {"module": "filter_anchors", "args": {"anchors": f"hub:{a}/anchors.json", **copy.deepcopy(APACHE_FILTER)}}
            s["waits_for"] = [f"{a}/anchors.json"]
            specs.append(s)
            choices.append(f"general answers {a_by}/: missing -> {s['name']} (tools.filter_anchors: the Apache-2.0 line's "
                           "picture licenses)")
    if f"{txt}/held.json" in files:
        choices.append(f"held text answers {txt}/: in the project repo")
    else:
        s = _top(entry, txt.split("/")[-1], "tool", needs)
        s["tool"] = {"module": "anchors", "args": {**{k: GENERAL_TEXT[k] for k in ("rows", "held_pictures", "n_picture", "n_text",
                                                                                   "n_held_picture", "n_held_text")},
                                                    "batch": batch, **{k: GENERAL_TEXT[k] for k in ("max_new_tokens",
                                                                                                    "max_new_tokens_text", "seed",
                                                                                                    "bank", "detail")},
                                                    "model_id": model}}
        specs.append(s)
        choices.append(f"held text answers {txt}/: missing -> {s['name']} (tools.anchors, the text bank: "
                       f"{GENERAL_TEXT['n_picture']} pictures with lettering, {GENERAL_TEXT['n_held_picture']} held)")
    return specs, choices


def _text_read(entry: dict, kind: str, project: str, extra: dict) -> dict:
    keys, _, _ = TEXT_READS[kind]
    body = {k: v for k, v in keys.items() if k != "ask_suffix"}
    body.update(_common(entry, project))
    body.update(extra)
    if "ask_suffix" in keys:
        body["ask_suffix"] = keys["ask_suffix"]
    return body


def line_specs(entry: dict, line: str, project: str, needs: str, progress_every_min: float | None) -> tuple:
    """A line's smoke and training (its nine reads after it), and the choices."""
    ga, n = entry["general_answers"], names(entry, line)
    s_set = V12_SETS[line]["path"]
    anchors = f"{ga['anchors_by'] if line == 'apache' else ga['anchors']}/anchors.json"
    held, held_txt = f"{ga['anchors']}/held.json", f"{ga['text']}/held.json"
    train = registry.recipe(entry)
    train.update(writers=registry.writers(entry), inputs=[f"hub:{s_set}"], anchors=f"hub:{anchors}",
                 anchor_batch=registry.V12_RECIPE["anchor_batch"], general_kl=registry.V12_RECIPE["general_kl"],
                 grad_checkpointing=registry.V12_RECIPE["grad_checkpointing"])
    full = dict(train, resume=registry.V12_RECIPE["resume"], ship_every=registry.V12_RECIPE["ship_every"],
                lazy_pictures=registry.V12_RECIPE["lazy_pictures"], picture_workers=registry.V12_RECIPE["picture_workers"])
    if progress_every_min:
        full["progress_every_min"] = progress_every_min
    smoke_train = dict(train, resume=registry.SMOKE["resume"], lazy_pictures=registry.V12_RECIPE["lazy_pictures"],
                       picture_workers=registry.V12_RECIPE["picture_workers"],
                       **{k: registry.SMOKE[k] for k in ("n_items", "max_steps", "do_ship")})
    stop = registry.smoke_stop(entry, line)
    smoke = _top(entry, n["smoke"], "train", needs)
    smoke.update(train=smoke_train, screen=False, waits_for=[s_set, anchors], stop_if=stop)
    reads = []
    for name, (kind, body) in zip(n["reads"], [
            ("extract", {**PROBE, **_common(entry, project)}),
            ("extract", {**MUSEUM, **_common(entry, project)}),
            ("tool", {"module": "keep_read", "args": {"held": f"hub:{held}", "scales": [1.0, 0.5], "batch": entry["keep_batch"]}}),
            ("extract", _text_read(entry, "closer", project, {"adapter_scale": 1.0, "wordings": OWN_WORDINGS})),
            ("tool", {"module": "keep_read", "args": {"held": f"hub:{held_txt}", "scales": [1.0, 0.5], "batch": entry["keep_batch"]}}),
            ("extract", _text_read(entry, "deep", project, {"adapter_scale": 1.0, "wordings": OWN_WORDINGS})),
            ("extract", _text_read(entry, "lettering", project, {"adapter_scale": 1.0, "wordings": OWN_WORDINGS})),
            ("extract", _text_read(entry, "control_lettering", project, {"adapter_scale": 1.0, "wordings": OWN_WORDINGS})),
            ("extract", _text_read(entry, "control", project, {"adapter_scale": 1.0, "wordings": OWN_WORDINGS}))]):
        reads.append({"name": name, "kind": kind, kind: body})
    training = _top(entry, n["train"], "train", needs)
    training.update(train=full, screen=copy.deepcopy(SCREEN), waits_for=[s_set, anchors, held, held_txt],
                    waits_for_runs=[n["smoke"]], after=reads)
    pace = entry["pace"][line]
    choices = [f"{line} line: the set {s_set} ({V12_SETS[line]['items']:,} items, {V12_SETS[line]['examples']:,} examples, "
               f"{V12_SETS[line]['steps']:,} steps); the general rows {anchors}",
               f"  smoke {n['smoke']}: {registry.SMOKE['max_steps']} steps on the set's first {registry.SMOKE['n_items']:,} items, "
               f"nothing shipped; stops at a steady pace above {stop['steady_s_per_step']} s a step after step "
               f"{stop['after_step']} ({registry.STOP_FACTOR if pace['measured'] else registry.STOP_FACTOR_ESTIMATE} x "
               f"{pace['s']} s, {pace['source']}) or a peak above {stop['peak_gb']} GB",
               f"  training {n['train']}: LoRA r{full['rank']} alpha {full['alpha']}, lr {full['lr']}, {full['batch']} x "
               f"{full['grad_accum']}, max_len {full['max_len']}, one pass, general_kl {full['general_kl']}"
               + (f", pictures {entry['image_settings']}" if entry["image_settings"] else "")
               + (f", repository code at {entry['base_revision']}" if entry["trust_remote_code"] else "")
               + f"; about {registry.hours(entry, line)} h at {pace['s']} s a step; waits for its smoke to pass"
               + (f"; progress file every {progress_every_min} min" if progress_every_min else "; no progress file"),
               f"  its reads after it: {', '.join(n['reads'])}"]
    return [smoke, training], choices


def comparison_specs(entry: dict, project: str, files: set | None, needs: str) -> tuple:
    """The comparison reads the model's cards need (the registry's, or the default set), and the choices."""
    k, model, ga = entry["key"], entry["base"], entry["general_answers"]
    plan = entry["comparisons"]
    choices = []
    if plan is None:
        plan = [{"of": "base", "reads": list(DEFAULT_COMPARISONS["base"])}]
        plan += [{"of": v["version"], "line": v["line"], "reads": list(DEFAULT_COMPARISONS["older"])}
                 for v in entry["versions"] if v["version"] != "v1.2"]
        choices.append("comparisons: none recorded for this model; the default set: the base model's "
                       f"{', '.join(DEFAULT_COMPARISONS['base'])} reads; each older version's "
                       f"{', '.join(DEFAULT_COMPARISONS['older'])} reads")
    specs = []
    for c in plan:
        if c["of"] == "base":
            for kind in c["reads"]:
                if kind not in TEXT_READS:
                    raise LaneError(f"{entry['key']}: a base comparison of kind {kind!r} (the base's own text reads only)")
                _, prefix, suffix = TEXT_READS[kind]
                name = f"XT_base_{k}_noadapter" if kind == "closer" else f"{prefix}base_{k}{suffix}"
                s = _top(entry, name, "extract", needs)
                s["extract"] = _text_read(entry, kind, project, {"adapter": None, "model_id": model, **_remote(entry)})
                specs.append(s)
            choices.append(f"comparisons of the base {model}: {', '.join(c['reads'])}")
            continue
        v = registry.version(entry, c["of"], c.get("line", "apache"))
        repo = project if v["adapter"] == PROJECT else v["adapter"]
        where = f"{repo}/{v['subfolder']}"
        if not v["ready"] or not v["subfolder"]:
            choices.append(f"comparisons of {v['version']} {v['line']}: skipped (its adapter is not ready: {v['source']})")
            continue
        if v["adapter"] == PROJECT and files is not None and f"{v['subfolder']}/adapter_model.safetensors" not in files:
            choices.append(f"comparisons of {v['version']} {v['line']}: skipped (its adapter {v['subfolder']} is not in the "
                           "project repo; it was trained in another)")
            continue
        for kind in c["reads"]:
            if kind == "keep":
                s = _top(entry, f"K0_{v['keep_tag']}", "tool", needs)
                s["tool"] = {"module": "keep_read", "args": {"model_id": model, "adapter": repo, "adapter_subfolder": v["subfolder"],
                                                             "held": f"hub:{ga['anchors']}/held.json", "scales": [0.0, 1.0],
                                                             "batch": entry["keep_batch"]}}
                s["waits_for"] = [f"{ga['anchors']}/held.json"]
            else:
                _, prefix, suffix = TEXT_READS[kind]
                s = _top(entry, f"{prefix}{v['tag']}{suffix}", "extract", needs)
                extra = {"model_id": model, "adapter": repo, "adapter_subfolder": v["subfolder"], "adapter_scale": 1.0}
                if v["wordings"]:
                    extra["wordings"] = v["wordings"]
                s["extract"] = _text_read(entry, kind, project, {**extra, **_remote(entry)})
            specs.append(s)
        choices.append(f"comparisons of {entry['name']} {v['version']} ({v['line']}; {where}): {', '.join(c['reads'])}")
    return specs, choices


def build_lane(entry: dict, which: str, project: str, files=None, lane: str | None = None, progress_every_min: float | None = 10,
               comparisons: bool | None = None, needs: str | None = None, force: bool = False) -> dict:
    """A model's lane: {"lane", "lines", "specs", "choices", "problems", "missing"}. which: "apache", "nc" or "both". files: the
    project repo's paths (a set or list; None: unknown, every input and adapter taken as present and the general answers made).
    comparisons: None = with the first line unless the lane is an -nc line alone."""
    if which not in WHICH:
        raise LaneError(f"which line: one of {', '.join(WHICH)}, not {which!r}")
    if not project or "/" not in project:
        raise LaneError(f"the project repo is owner/name, not {project!r}")
    lines = WHICH[which]
    off = [ln for ln in lines if ln not in entry["lines"]]
    if off and not force:
        raise LaneError(f"{entry['name']} gets the {' and the '.join(LINE_NAMES[ln] for ln in entry['lines'])} line in the "
                        f"catalogue, not the {' and the '.join(LINE_NAMES[ln] for ln in off)} line (--force builds it anyway)")
    if entry["blockers"] and not force:
        raise LaneError(f"{entry['name']}'s lane cannot run as generated: " + " | ".join(entry["blockers"]) + " (--force builds it anyway)")
    needs = needs or __version__
    files_set = set(files) if files is not None else None
    lane = lane or (entry["lane"] + ("_nc" if which == "nc" else ""))
    with_comparisons = (which != "nc") if comparisons is None else comparisons
    choices = [f"model {entry['key']}: {entry['name']} on {entry['base']}"
               + (f" at {entry['base_revision']}" if entry["base_revision"] else "")
               + f"; lines {', '.join(lines)}; lane {lane}; project repo {project}",
               f"every spec needs {needs} (the first package that honors stop_if, waits_for_runs and progress_every_min)"
               + (f" and transformers {entry['transformers']}" if entry["transformers"] else "")]
    if not entry["scheduled"]:
        choices.append(f"{entry['name']} is not trained in the catalogue (training instructions only): this lane is for anyone "
                       "who trains it")
    if off:
        choices.append(f"FORCED: {', '.join(off)} is not a line of {entry['name']} in the catalogue")
    if entry["blockers"]:
        choices += [f"FORCED past a blocker: {b}" for b in entry["blockers"]]
    specs, general_choices = general_answer_specs(entry, lines, files_set if files_set is not None else set(), needs)
    choices += general_choices
    if files_set is None:
        choices.append("the project repo's files are unknown: its general answers are made in this lane")
    for i, line in enumerate(lines):
        made, line_choices = line_specs(entry, line, project, needs, progress_every_min)
        specs += made
        choices += line_choices
        if i == 0:
            if with_comparisons:
                comp, comp_choices = comparison_specs(entry, project, files_set, needs)
                specs += comp
                choices += comp_choices
            else:
                choices.append("comparisons: not in this lane (an -nc lane alone leaves them to the Apache-2.0 lane; "
                               "--comparisons adds them)")
    problems, missing = check(specs, files_set)
    return {"lane": lane, "lines": lines, "specs": specs, "choices": choices, "problems": problems, "missing": missing}


def _paths(obj):
    if isinstance(obj, str):
        if obj.startswith(("hub:", "repo:")):
            yield obj.split(":", 1)[1]
    elif isinstance(obj, list):
        for x in obj:
            yield from _paths(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _paths(x)


def check(specs: list, files: set | None) -> tuple:
    """(problems, missing): run names used twice or already marked in the project repo; the project-repo inputs the lane reads
    that are neither in it nor made by a run before the one that reads them (paths in another repo are not checked)."""
    every = [s for s in specs] + [a for s in specs for a in s.get("after") or []]
    seen, problems = set(), []
    for s in every:
        if s["name"] in seen:
            problems.append(f"the run name {s['name']} is used twice")
        seen.add(s["name"])
        if files is not None and f"runs/done/{s['name']}.json" in files:
            problems.append(f"the run name {s['name']} already has a marker in the project repo (it would be skipped)")
    if files is None:
        return problems, []
    made, missing = set(), []
    for s in specs:
        for unit in [s] + list(s.get("after") or []):
            body = {k: v for k, v in unit.items() if k not in ("after", "waits_for", "waits_for_runs", "stop_if")}
            for p in list(unit.get("waits_for") or []) + list(_paths(body)):
                p = p.strip("/")
                if p.startswith(PROJECT_FOLDERS) and p not in files and p not in made and f"{p} (read by {unit['name']})" not in missing:
                    missing.append(f"{p} (read by {unit['name']})")
        if s["kind"] == "tool" and s["tool"]["module"] in ("anchors", "filter_anchors"):
            made.update(f"anchors/{s['name']}/{f}" for f in ("anchors.json", "held.json", "summary.json"))
    return problems, missing


# ------------------------------------------------------------------------------------------------------------ the notebook
_INSTALL = '''# C1: install, then Runtime > Restart session
import importlib.util, subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
{pin_note}
pip("git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v{version}", "{transformers}", "accelerate>=1.0", "peft>=0.18")
{fast_path}
try:                # Colab's torchao 0.10 breaks PEFT when it builds a LoRA (PEFT wants 0.16 or newer, or none at all)
    import torchao
    _v = tuple(int(x) for x in torchao.__version__.split(".")[:2])
    if _v < (0, 16):
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
        print("torchao", torchao.__version__, "removed: PEFT cannot build a LoRA beside it")
    else:
        print("torchao", torchao.__version__, "stays")
except ImportError:
    print("no torchao")

print("installed; now Runtime > Restart session, then run C2")'''
_FAST_PATH = '''
FAST_PATH = True   # the linear-attention kernels Qwen3.5 uses; causal-conv1d compiles for about 5 minutes; all or nothing
if FAST_PATH:
    if all(importlib.util.find_spec(m) for m in ("causal_conv1d", "fla", "triton")):
        print("fast path already present")
    else:
        try:
            pip("causal-conv1d", "--no-build-isolation")
            pip("flash-linear-attention", "triton")
            print("fast path on")
        except subprocess.CalledProcessError as e:
            print("fast path off (training still runs, slower):", e)
            subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "causal-conv1d"])
'''
_QUEUE = '''# C2: work through the lane's queue (runs/queue_{lane}.json in the project repo; new runs can be added while it runs)
import os
os.environ["QST_REPO"] = "{project}"   # before the package is imported: every file of this lane lives in this repo
LANE = "{lane}"

from qwen_symbolic_thinking import REPO, __version__
from qwen_symbolic_thinking.queue import run_queue

print("package", __version__, "| project repo", REPO, "| lane", LANE)
ran = run_queue(LANE, wait_hours=12)
print("ran:", ran)
'''
NOTEBOOK_META = {"metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                              "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
                 "nbformat": 4, "nbformat_minor": 0}
BADGE = "https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/{name}"
DATASET = "AbstractPhil/thought-symbolic-image-recaptioned"


def install_cell(entry: dict, version: str = __version__) -> str:
    pinned = entry["install"]["transformers"].startswith("transformers==")
    note = ("\n# The transformers release this family's repository code was built on (its specs ask for it too)." if pinned else "")
    return _INSTALL.format(version=version, transformers=entry["install"]["transformers"], pin_note=note,
                           fast_path=_FAST_PATH if entry["install"]["fast_path"] else "")


def queue_cell(lane: str, project: str) -> str:
    return _QUEUE.format(lane=lane, project=project)


def _source(text: str) -> list:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def notebook(entry: dict, built: dict, project: str, version: str = __version__) -> dict:
    """The lane's notebook: what it trains and in what order (from the built lane), C1 installs the package at this version
    (and the model's transformers), C2 points the package at the project repo and works through the lane's queue."""
    lines, lane = built["lines"], built["lane"]
    title = {("apache", "nc"): "the Apache-2.0 and non-commercial lines", ("apache",): "the Apache-2.0 line",
             ("nc",): "the non-commercial line (-nc)"}[tuple(lines)]
    order = []
    for s in built["specs"]:
        what = ("the smoke " + s["name"] if s["name"].startswith("S_v12_smoke") else
                "the training " + s["name"] + " and its nine reads" if s["kind"] == "train" else s["name"])
        order.append(what)
    hours = "; ".join(f"the {ln} line about {registry.hours(entry, ln)} h of training at {entry['pace'][ln]['s']} s a step"
                      + ("" if entry["pace"][ln]["measured"] else " (estimated)") for ln in lines)
    md = (f"# {entry['name']} v1.2: {title}\n\n"
          f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({BADGE.format(name=lane + '.ipynb')})\n\n"
          f"The v1.2 fine-tune of [{entry['base']}](https://huggingface.co/{entry['base']}) on the rows of the public dataset "
          f"[thought-symbolic-image-recaptioned](https://huggingface.co/datasets/{DATASET}) (release 5), held near the base model's "
          "behavior while it learns them (the general-row KL at weight 1, on general requests the base model answered itself), "
          "in one pass over the rows. The Apache-2.0 line trains on the dataset's Apache-2.0 part; the non-commercial line (-nc) "
          "on every row of the release; the held-out pictures are the same for both, so their reads compare.\n\n"
          f"The queue runs, in order: {'; '.join(order)}. A smoke that runs slower or holds more memory than its limits fails, and "
          f"its training does not start. {hours[0].upper() + hours[1:]}.\n\n"
          "The training runs with resume on: its whole state saves to the project repo every 2 hours and once more after the "
          "runtime's 23rd hour. If the runtime ends mid-run, open this notebook on a fresh runtime and run it again: the lane "
          "continues the training from its saved state. Its progress file (runs/progress/<run>.json in the project repo) says "
          "where it stands: `python -m qwen_symbolic_thinking.tools.progress <project repo>`.\n\n"
          f"Runtime: a GPU (the RTX PRO 6000, 95 GB) and the `HF_TOKEN` secret with write access to the project repo `{project}`, "
          "where every file of this lane lives (its training sets and inputs first: see the package's docs/registry_and_lanes.md). "
          "Run C1, restart the session (Runtime > Restart session), then run C2.\n")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": _source(md)},
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(install_cell(entry, version))},
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(queue_cell(lane, project))}]
    return {"cells": cells, **copy.deepcopy(NOTEBOOK_META)}


# ------------------------------------------------------------------------------------------------------------ the command
def _list_files(project: str) -> list:
    from huggingface_hub import HfApi

    from .hub import token, with_retries
    return with_retries(lambda: HfApi(token=token()).list_repo_files(project), "list the project repo's files")


def write(built: dict, nb: dict, out: str) -> tuple:
    os.makedirs(out, exist_ok=True)
    q = os.path.join(out, f"queue_{built['lane']}.json")
    with open(q, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(built["specs"], fh, indent=1)
        fh.write("\n")
    n = os.path.join(out, f"{built['lane']}.ipynb")
    with open(n, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return q, n


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build a model's v1.2 lane (its queue and notebook) from the registry.")
    ap.add_argument("model", help=f"a registry key ({', '.join(registry.keys())}), a base model id or a release repo")
    ap.add_argument("which", choices=list(WHICH), help="the line(s): apache, nc or both (one lane, Apache first)")
    ap.add_argument("--project-repo", required=True, help="the Hub repo the lane reads and writes (owner/name)")
    ap.add_argument("--out", default=".", help="where the queue and the notebook are written (default: here)")
    ap.add_argument("--files", help="a JSON list of the project repo's paths (default: listed from the Hub)")
    ap.add_argument("--lane", help="the lane's name (default: the registry's)")
    ap.add_argument("--progress-every-min", type=float, default=10.0, help="the training's progress file, 0 for none")
    comp = ap.add_mutually_exclusive_group()
    comp.add_argument("--comparisons", dest="comparisons", action="store_true", default=None)
    comp.add_argument("--no-comparisons", dest="comparisons", action="store_false")
    ap.add_argument("--force", action="store_true", help="build a line the catalogue does not give, or past a blocker")
    args = ap.parse_args(argv)
    entry = registry.get(args.model)
    if args.files:
        with open(args.files, encoding="utf-8") as fh:
            data = json.load(fh)
        files = data["files"] if isinstance(data, dict) else data
    else:
        try:
            files = _list_files(args.project_repo)
        except Exception as e:
            print(f"cannot list {args.project_repo}'s files ({type(e).__name__}: {str(e)[:200]}): set HF_TOKEN (read access) "
                  "or pass --files", file=sys.stderr)
            return 2
    try:
        built = build_lane(entry, args.which, args.project_repo, files, args.lane, args.progress_every_min or None,
                           args.comparisons, force=args.force)
    except LaneError as e:
        print(f"not built: {e}", file=sys.stderr)
        return 2
    nb = notebook(entry, built, args.project_repo)
    q, n = write(built, nb, args.out)
    print("\n".join(built["choices"]))
    print(f"\nthe queue ({len(built['specs'])} specs), in order:")
    for s in built["specs"]:
        print(f"  {s['name']}" + (f"  (after: {', '.join(a['name'] for a in s['after'])})" if s.get("after") else ""))
    print("\nthe project repo:")
    for m in built["missing"]:
        print(f"  MISSING {m}")
    for p in built["problems"]:
        print(f"  PROBLEM {p}")
    if not built["missing"] and not built["problems"]:
        print("  every input is there or made earlier in the lane; every run name is free")
    print(f"\nwrote {q}\nwrote {n}\nnothing uploaded: put the queue in {args.project_repo} as runs/queue_{built['lane']}.json to "
          "start the lane")
    return 1 if built["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
