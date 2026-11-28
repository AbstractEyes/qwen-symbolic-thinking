"""A model repo's TRAINING.md: how to train the model's v1.2 from a fresh account, written from the registry (registry.py) and
the one template below.

    python -m qwen_symbolic_thinking.training_md <model key> --out <file> [--version <package version>]

The page says what v1.2 is (release 5's rows, one pass, the keeping objective on the model's own general answers), the lines the
model gets and their licenses, the settings, the pace, peak memory and hours on a G4 (each measured or estimated, with its
source), the files a project repo must hold before its lane starts and where each comes from publicly, the steps from a fresh
account (the package at its tag, the lane generator against the reader's own project repo, the notebook in Colab, the general
answers, the smoke and its stop rule, the training, the reads), and the bars with their pass marks and our numbers where we
have them. For a model trained only by those who choose to (the registry's "scheduled": False) it says so, with what stops its
lane today.

Nothing on the page points to a repo that is not public. A file with no public source is printed as "**not published**" with
what it is and how it was made, and listed at the end of the command's output. Nothing is uploaded.

2026-11-27: the v1.2 Apache lanes' inputs are public in the record repo (seven as our runs read them; the two v1.1 row files in
picture form; the cleaned text requests less one request the public rows release's word rule holds back), so SOURCES names
them; the files table lists only the files the model's own lines read, the requests file included where its general answers
take their text requests from it, and the general answers' counts are the model's own (500 + 40 from the requests file, where
it has one).

2026-11-27 (0.1.77): the two Gemma 4 models take their text requests from the Qwen-written requests file their lanes asked
(text/requests_qwen_v1.json, 2,000 + 200; public in the record repo less two requests the word rule holds back); a model whose
own version's lane differed from the guide's says so after the general answers (its general_requests "as_run"), and our bars
name the preservation read they were read on where it is not the lane's own (registry.BARS "keep")."""
from __future__ import annotations

import os
import sys

from . import __version__, lines, registry
from .registry import V12_SETS
from .resume import LATE_SAVE_H

NOT_PUBLISHED = "**not published**: {what}"
GITHUB = "https://github.com/AbstractEyes/qwen-symbolic-thinking"
PICTURES = lines.PICTURES
RECORD = registry.RECORD
G4 = "a G4, an RTX PRO 6000 Blackwell with 95 GB"
SHORT = {"apache": "Apache-2.0", "nc": "-nc"}
KIND_WORDS = {"closer": "closer look", "deep": "long look", "lettering": "closer look with the lettering sentence",
              "control_lettering": "control with the sentence", "control": "control", "keep": "preservation read",
              "keep_text": "text preservation read"}
UNITS_AN_HOUR = 10          # Colab compute units an hour of a G4 when the v1.2 lanes ran (November 2026)

# Where each file a project repo must hold comes from publicly: a public source, or None and what it is.
_AS_READ = f"the same path in the public record repo {RECORD}, as our runs read it"
REQUESTS = "text/requests_qwen_v1_clean.json"
REQUESTS_GEMMA = "text/requests_qwen_v1.json"
SOURCES = {
    "sft/v12_rows_a_by.json": (_AS_READ, None),
    "sft/v12_rows_nc.json": (None, "the -nc training set, every row of release 5 of the public dataset as a training file (so "
                                   "its rows are public there, in both parts of the release)"),
    "sft/v11_rows_a.json": (f"the same path in the public record repo {RECORD}, in picture form: every item with its picture "
                            "fields only, which is all the general answers read (the repo's README says why)", None),
    "sft/v11_rows_a_held.json": (f"the same path in the public record repo {RECORD}, in picture form (as above)", None),
    "screens/inputs/screen_v1.json": (_AS_READ, None),
    "licenses/coco_photo_licenses.json": (_AS_READ, None),
    REQUESTS: (f"the same path in the public record repo {RECORD}, less one request the public rows release's word rule holds "
               "back (the repo's README says so; a lane built from it answers one different request among its 500)", None),
    REQUESTS_GEMMA: (f"the same path in the public record repo {RECORD}, less two requests the public rows release's word rule "
                     "holds back (the repo's README says so; a lane built from it answers one different request among its "
                     "2,000, and the same 200 held)", None),
    "text/anchor_pool_v1.json": (_AS_READ, None),
    "text/anchor_held_v1.json": (_AS_READ, None),
    "text/text_held_v1.jsonl": (_AS_READ, None),
    "text/text_held_v1_deep.jsonl": (_AS_READ, None),
    "text/notext_control_v1.jsonl": (_AS_READ, None),
}
REQUESTS_WHAT = {REQUESTS: "the text requests of the general answers: requests the Qwen3.5 base models wrote themselves, "
                           "filtered by rule",
                 REQUESTS_GEMMA: "the text requests of the general answers: requests Qwen3.5-9B and Qwen3.5-4B wrote "
                                 "themselves (Gemma 4 seldom ends a request it writes, so its lanes ask these)"}
# What the reads need beyond the lane's own files.
READ_SOURCES = {
    "text/text_held_v1_names.jsonl": (None, "the held manifest with each picture's name screen (its flags mark shown words that "
                                            "may hold a person's name, a credit or a web address); without it the text reads "
                                            "run with --manifest repo:text/text_held_v1.jsonl, the same numbers without the "
                                            "name split"),
    "the yardstick's records": (None, "our yardstick arm's records (its markers, screen and reads); its numbers are in the "
                                      "table below, so a line's bars can be set beside them, and --yardstick-repo reads a "
                                      "yardstick from any repo that holds one"),
}

TEMPLATE = """# Training {name} v1.2

{status}

This page is written from the trainer package's model registry ({github}); its steps install version {version}. Every
number says whether it was measured or estimated, and where it comes from.

## What v1.2 is

{name} v1.2 is {what_is} on the rows of the public dataset
[thought-symbolic-image-recaptioned](https://huggingface.co/datasets/{dataset}), release 5, in one pass:

- the rows: release 5, split by license into two training sets: the Apache-2.0 set ({a_items} items, {a_examples} examples) and
  the non-commercial set ({n_items} items, {n_examples} examples, every row of the release);
{recipe_bullet}
- the keeping objective: while it learns the rows, the model is held near its base model by a KL term (weight
  {general_kl}) on the base model's own answers to ordinary requests, with and without a picture ({anchor_batch} beside every
  batch). So the lane first asks the base model those requests and keeps its answers (the general answers below).{extra_bullets}

## {lines_heading}

{lines_table}

## The settings

{settings_table}

## Time and memory on {g4}

{pace_table}

A G4 cost about {units} Colab compute units an hour when we ran these lanes (November 2026). Beside the training: the general
answers (once per model), a smoke before each training (a few minutes), and the screen and reads after each training (about an
hour a line, an estimate). A Colab runtime ends after about 24 hours; a longer training continues on a fresh runtime from its
saved state.

## What your project repo must hold before the lane starts

The lane reads and writes one Hub repo of yours, the project repo (a model repo; it may be private). Before the first run it must
hold these files at these paths:

{files_table}

A file in the public record repo is copied to the same path of your project repo, for example:

    huggingface-cli download {record} <path> --local-dir inputs
    huggingface-cli upload <you>/<project> inputs/<path> <path>

The probe and museum reads draw their pictures from the public picture lists ([{pictures}](https://huggingface.co/datasets/{pictures})),
and the version-1 adapter its comparison reads load lives in the public record repo
([{record}](https://huggingface.co/{record})). Earlier versions we trained in our own project repo are not public: the lane
generator leaves their comparison reads out of your lane and says so.

## The steps, from a fresh account

1. A Hugging Face account, a token with write access (`HF_TOKEN`), and a project repo of your own (`<you>/<project>`).
2. The files above in the project repo, at those paths.
3. The trainer package at its tag:

       pip install "git+{github}@v{version}"

4. The lane, generated against your project repo:

       python -m qwen_symbolic_thinking.lines {key} {which} --project-repo <you>/<project> --out lane

   It prints every choice it makes, names every input your project repo lacks and every run name already taken, and writes
   `lane/queue_{lane}.json` and `lane/{lane}.ipynb`. Nothing is uploaded.{force_note}
5. The queue in your project repo, at `runs/queue_{lane}.json`:

       huggingface-cli upload <you>/<project> lane/queue_{lane}.json runs/queue_{lane}.json

6. The notebook in Colab: open `lane/{lane}.ipynb` there (File > Upload notebook), or keep it in a GitHub repo of yours and open
   it with `https://colab.research.google.com/github/<you>/<repo>/blob/main/{lane}.ipynb`. Pick a GPU runtime (we ran on
   {g4}), give the notebook your `HF_TOKEN` as a Colab secret, run C1 (it installs the package at its tag{install_note}), restart the
   session (Runtime > Restart session), then run C2. C2 works through the queue; on a fresh runtime it picks up where the last
   one stopped.

## What the lane does

{prepare_note}1. **The general answers**, only when the project repo lacks them: `{ga_anchors}` asks the base model {n_picture:,} requests
   about pictures and {n_text:,} plain requests{text_source} ({n_held_picture} + {n_held_text} more held out) and keeps its answers;
   `{ga_by}` keeps those whose pictures the Apache-2.0 line may use; `{ga_text}` asks it about {n_text_pictures:,} pictures with
   lettering ({n_text_held} held). {general_time}
2. **The smoke**, before each training: {smoke_steps} steps on the set's first {smoke_items:,} items, nothing shipped. It stops
   itself: {stop_rule}. A smoke past its limits marks itself failed, and its training does not start.
3. **The training**: the recipe above.{train_note} {train_state} saves to the project repo every 2 hours, and once more after
   {late_save:g} hours of the runtime; if the runtime ends mid-run, run the notebook again on a fresh runtime and the training
   continues from its saved state. Its progress file, `runs/progress/<run>.json`, says where it stands every 10 minutes:

       python -m qwen_symbolic_thinking.tools.progress <you>/<project>

4. **The reads after each training**: the screen, then nine reads of what it shipped: the held-out probe, the museum read, the
   preservation read (its drift from the base on the held general answers), the closer and long text reads, the text
   preservation read, the lettering read and two controls (photos without lettering, with and without the lettering sentence).
5. **The comparison reads**, after the first line: {comparisons}.

## The reads and the bars

When a line's reads are done, two commands read them from the project repo (they run no model and write nothing to it):

    python -m qwen_symbolic_thinking.reads {key} {read_line} bars --project-repo <you>/<project> --out reads
    python -m qwen_symbolic_thinking.reads {key} {read_line} text --project-repo <you>/<project> --out reads
{reads_note}
The bars compare the line's training with a yardstick: an arm of the same line trained without the keeping objective
({yardstick_runs}). {yardstick_source}

{bars_table}

The text reads (seven tables: how often the rows name the text a picture shows, how often they write lettering, whether the
quoted words are the picture's own) carry no pass mark; they are read beside the base model and the earlier versions.
{names_source}
"""


class Page:
    """The page being written and the placeholders it printed."""

    def __init__(self):
        self.placeholders = []

    def missing(self, where: str, what: str) -> str:
        self.placeholders.append((where, what))
        return NOT_PUBLISHED.format(what=what)


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def _lr(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-")          # 2e-4, 5e-5


def _table(head: list, rows: list) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _release(entry: dict, line: str) -> str:
    for v in entry["versions"]:
        if v["version"] == "v1.2" and v["line"] == line:
            if v["release"]:
                return f"[{v['release']['repo']}](https://huggingface.co/{v['release']['repo']}) at the tag {v['release']['tag']}"
            return "trained, not released yet" if v["ready"] else "in training"
    return "not trained yet" if entry["scheduled"] else "not trained by us"


def status(entry: dict) -> str:
    lines_ = " and ".join(lines.LINE_NAMES[ln] for ln in entry["lines"])
    if entry["scheduled"]:
        return (f"We train {entry['name']} at v1.2 on the {lines_} line{'s' if len(entry['lines']) > 1 else ''}. These are the "
                "steps we take, written so that anyone can take them on a Hub repo of their own.")
    out = (f"We do not train {entry['name']} at v1.2: this page is for anyone who wants to. It has the {lines_} line only, "
           "and we have not run its v1.2 lane.")
    if entry["blockers"]:
        out += " Its lane cannot run as generated yet: " + "; ".join(entry["blockers"]) + "."
    else:
        out += " Nothing in the package stops its lane; its smoke is the first measurement of its pace and memory."
    return out


def lines_table(entry: dict) -> str:
    rows = []
    for ln in entry["lines"]:
        s = V12_SETS[ln]
        rows.append([lines.LINE_NAMES[ln], f"`{s['path']}`", f"{_n(s['items'])} items, {_n(s['examples'])} examples, "
                     f"{_n(s['steps'])} steps", entry["licenses"][ln], _release(entry, ln)])
    note = ""
    if entry["licenses"].get("apache") not in (None, "Apache-2.0"):
        note = (f"\n\nThe Apache-2.0 line names the rows it trains on (those whose pictures may be used that way); the trained "
                f"adapter follows its base model's license, {entry['licenses']['apache']}.")
    return _table(["line", "training set (a path in your project repo)", "rows", "license of the result", "our release"], rows) + note


def base_text(entry: dict) -> str:
    """The base model as a link, or, for a starting model the lane builds (a "hub:" path), what it is and which run builds it."""
    if entry["base"].startswith("hub:"):
        return f"built by the lane's first run (`{entry['prepare']['name']}`): {entry['about']}"
    rev = f" at the revision `{entry['base_revision']}`" if entry["base_revision"] else ""
    return f"[{entry['base']}](https://huggingface.co/{entry['base']}){rev}"


def settings_table(entry: dict) -> str:
    fe = registry.full_entry(entry)
    lora = f"rank {entry['lora']['rank']}, alpha {entry['lora']['alpha']}"
    rows = [["base model", base_text(entry)],
            ["family, model type", f"{entry['family']}, {entry['model_type']}"],
            ["its repository's code", "runs (trust_remote_code), pinned to that revision" if entry["trust_remote_code"] else "not needed"],
            ["transformers", f"{entry['transformers'][2:]} exactly (its repository code was built on it)" if entry["transformers"]
             else "5 or later"],
            ["what trains", f"the full fine-tune: every weight but those its family keeps frozen (below); the LoRA beside it: {lora}"]
            if fe else ["LoRA", lora],
            ["learning rate", f"{_lr(fe['lr'])} (the full fine-tune); {_lr(entry['lr'])} (the LoRA)" if fe else _lr(entry["lr"])],
            ["batch", f"{entry['batch']} x {entry['grad_accum']} (sequences a step: {entry['batch'] * entry['grad_accum']})"],
            ["max length", f"{_n(entry['max_len'])} tokens"],
            ["pictures", ", ".join(f"{k} {_n(v)}" for k, v in entry["image_settings"].items()) if entry["image_settings"]
             else "the processor's default"],
            ["the reads' batch", f"{entry['read_batch']} (the preservation reads {entry['keep_batch']})"],
            ["what the package does for its family", entry["template"]]]
    return _table(["setting", "value"], rows)


def pace_table(entry: dict) -> str:
    rows = []
    fe = registry.full_entry(entry)
    trainings = [(fe, " (the full fine-tune)"), (entry, " (the LoRA)")] if fe else [(entry, "")]
    for ln in entry["lines"]:
        for e, which in trainings:
            p, pk, stop = e["pace"][ln], e["peak"][ln], registry.smoke_stop(e, ln)
            peak = f"{pk['gb']} GB; {pk['source']}" if pk["gb"] is not None else f"not measured yet; {pk['source']}"
            rows.append([lines.LINE_NAMES[ln] + which, f"{p['s']} s; {p['source']}",
                         f"about {registry.hours(e, ln)} ({_n(V12_SETS[ln]['steps'])} steps x {p['s']} s; "
                         f"{'measured pace' if p['measured'] else 'estimated'})", peak,
                         f"above {stop['steady_s_per_step']} s a step after step {stop['after_step']}, or above {stop['peak_gb']} GB"])
    return _table(["line", "seconds a step", "training hours", "peak GPU memory", "the smoke stops at"], rows)


def project_files(entry: dict) -> dict:
    """The files the model's own lines read: the training set of each of its lines, the general answers' inputs, the requests
    file where its general answers take their text requests from one, and the text reads' inputs."""
    sets = {V12_SETS[ln]["path"] for ln in ("apache", "nc")}
    out = {}
    for path, what in lines.PROJECT_FILES.items():
        if path in sets and path not in {V12_SETS[ln]["path"] for ln in entry["lines"]}:
            continue
        out[path] = what
        req = (entry.get("general_requests") or {}).get("requests", "")
        if path == "sft/v11_rows_a_held.json" and req.startswith("hub:"):
            assert req[4:] in REQUESTS_WHAT, req
            out[req[4:]] = REQUESTS_WHAT[req[4:]]
    return out


def files_table(entry: dict, page: Page) -> str:
    rows = []
    for path, what in project_files(entry).items():
        public, fill = SOURCES[path]
        rows.append([f"`{path}`", what, public or page.missing(path, fill)])
    return _table(["path", "what it holds", "where it comes from publicly"], rows)


def general_time(entry: dict) -> str:
    m = registry.GENERAL_ANSWERS_MINUTES
    if entry["key"] in m:
        t = m[entry["key"]]
        return f"On a G4 the three took {t['minutes']} minutes (measured: {t['source']})."
    # 2026-11-27: the measured times do not grow with the model's size (the 1B Anima eyes 139.1 against the 12B's 103.2; the
    # requests asked differ by model), so the sentence gives them as they are and claims no trend
    known = ", ".join(f"the {registry.get(k)['name']}'s {v['minutes']}" for k, v in m.items())
    return (f"Not measured for this model; on a G4 they took {known} minutes (measured), which follow neither the model's size "
            "nor one another closely.")


def comparisons(entry: dict) -> str:
    plan = lines.comparison_plan(entry)
    out = []
    for c in plan:
        kinds = ", ".join(KIND_WORDS[k] for k in c["reads"])
        if c["of"] == "base":
            out.append(f"the base model's {kinds} reads")
            continue
        v = registry.version(entry, c["of"], c.get("line", "apache"))
        if not v["ready"]:
            continue
        where = ("its adapter is in the public record repo" if v["adapter"] == registry.RECORD else
                 "its adapter is in our own project repo, which is not public, so the generator leaves these out of your lane")
        out.append(f"{entry['name']} {v['version']} on the {SHORT[v['line']]} line, its {kinds} reads ({where})")
    note = ". These are the generator's default set (no other is recorded for this model)" if entry["comparisons"] is None else ""
    return "; ".join(out) + note


def _misses(b: dict, y: dict) -> list:
    """The bars a BARS entry misses against its yardstick's values, each with its value and the line it falls short of."""
    out = [f"T1 {w} ({b[k]:.3f}, the mark 0.95)" for k, w in (("format", "full-form format"), ("full_rows", "full rows"))
           if b[k] < 0.95]
    out += [f"T2 {k} ({b[k]:.3f}, the band's floor {y[k] - 0.03:.3f})" for k in ("said", "backed") if b[k] < y[k] - 0.03 - 1e-9]
    out += [f"T3 probe ({b['probe']}, the floor {y['probe'] - 10})"] if b["probe"] < y["probe"] - 10 else []
    out += [f"K1 drift {w} ({b[k]:.3f})" for k, w in (("kl_picture", "picture"), ("kl_text", "text")) if b[k] > 0.5 * y[k]]
    out += [f"K3 ended ({b['ended']:.3f}, the base's {b['base_ended']:.3f} - 0.02)"] if b["ended"] < b["base_ended"] - 0.02 else []
    return out


def _and(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def bars_table(entry: dict) -> str:
    ls = entry["lines"]
    y = {ln: registry.YARDSTICK[ln]["values"] for ln in ls}
    fe = registry.full_entry(entry)
    ours = {ln: registry.BARS.get(((fe or entry)["key"], ln)) for ln in ls}     # a full fine-tune's: the release's

    def both(f):
        return " / ".join(f(ln) for ln in ls)

    def our(ln, f):
        b = ours[ln]
        return f(b) if b else ("not read yet" if entry["scheduled"] else "not trained by us")
    marks = [("T1 full-form format", ">= 0.95", lambda ln: "-", lambda b: f"{b['format']:.3f}"),
             ("T1 full rows", ">= 0.95", lambda ln: "-", lambda b: f"{b['full_rows']:.3f}"),
             ("T2 said", "within 0.03 of the yardstick (above the band is never a miss)", lambda ln: f"{y[ln]['said']:.3f}",
              lambda b: f"{b['said']:.3f}"),
             ("T2 backed", "the same", lambda ln: f"{y[ln]['backed']:.3f}", lambda b: f"{b['backed']:.3f}"),
             ("T3 probe (of 500 held-out pictures)", "within 10 of the yardstick (above the band is never a miss)",
              lambda ln: f"{y[ln]['probe']}", lambda b: f"{b['probe']}"),
             ("K1 drift, picture asks", "at most half the yardstick's", lambda ln: f"{y[ln]['kl_picture']:.3f}",
              lambda b: f"{b['kl_picture']:.3f}"),
             ("K1 drift, text asks", "at most half the yardstick's", lambda ln: f"{y[ln]['kl_text']:.3f}",
              lambda b: f"{b['kl_text']:.3f}"),
             ("K2 row form", "its excess over the base at most 0.02 and at most half the yardstick's excess",
              lambda ln: f"excess {y[ln]['form_excess']:.3f}", lambda b: f"{b['form']:.3f}"),
             ("K3 ended", "at least the base's - 0.02", lambda ln: "-", lambda b: f"{b['ended']:.3f} (base {b['base_ended']:.3f})"),
             ("the verdict", "every bar passes", lambda ln: "-", lambda b: "PASS" if b["pass"] else "NOT READY")]
    names = " / ".join(SHORT[ln] for ln in ls)
    rows = [[bar, mark, both(yv), both(lambda ln, f=f: our(ln, f))] for bar, mark, yv, f in marks]
    sources = [f"{SHORT[ln]}: {b['run']}, read {b['read']}" + (
        f" on the preservation read {b['keep']} (its held text requests asked apart, since the held set of its training's "
        f"general answers holds none; `--keep {b['keep']}` reads it)" if b.get("keep") else "")
        for ln in ls for b in [ours[ln]] if b]
    if fe:
        sources = []
        for ln in ls:
            for which, b in (("the full fine-tune", ours[ln]), ("the LoRA beside it", registry.BARS.get((entry["key"], ln)))):
                if b:
                    m = _misses(b, y[ln])
                    sources.append(f"{which}, {b['run']}, read {b['read']} ({b['note']}): "
                                   + ("every bar passes" if not m else "it misses " + _and(m)))
        out = _table(["bar", "pass mark", f"the yardstick ({names})", f"our {entry['name']} v1.2, the full fine-tune ({names})"],
                     rows)
        if sources:
            out += "\n\nOur numbers: the bars of " + ". The bars of ".join(sources) + "."
        return out
    out = _table(["bar", "pass mark", f"the yardstick ({names})", f"our {entry['name']} v1.2 ({names})"], rows)
    if sources:
        out += "\n\nOur numbers: the bars of " + "; ".join(sources) + "."
    return out


def render(entry: dict, version: str = __version__) -> tuple:
    """(the page's text, its placeholders as (where, what would fill it))."""
    page = Page()
    files = files_table(entry, page)            # first: the placeholders are listed in the page's order
    ga, which = entry["general_answers"], "both" if len(entry["lines"]) > 1 else entry["lines"][0]
    lane = entry["lane"] + ("_nc" if which == "nc" else "")
    stop = {ln: registry.smoke_stop(entry, ln) for ln in entry["lines"]}
    stop_rule = "; ".join(f"on the {lines.LINE_NAMES[ln]} line, a steady pace above {stop[ln]['steady_s_per_step']} s a step "
                          f"after step {stop[ln]['after_step']} or a peak above {stop[ln]['peak_gb']} GB" for ln in entry["lines"])
    fe0 = registry.full_entry(entry)
    if fe0:
        stop_f = {ln: registry.smoke_stop(fe0, ln) for ln in entry["lines"]}
        stop_rule += "; the full fine-tune's smoke, " + "; ".join(
            f"a steady pace above {stop_f[ln]['steady_s_per_step']} s a step after step {stop_f[ln]['after_step']} or a peak above "
            f"{stop_f[ln]['peak_gb']} GB" for ln in entry["lines"])
    stop_rule += (" (1.25 x the pace a v1.2 run measured, or 1.5 x the estimate where none has; 85 GB)")
    yruns = "; ".join(f"{SHORT[ln]}: {registry.YARDSTICK[ln]['run']}" for ln in entry["lines"])
    where, fill = READ_SOURCES["the yardstick's records"]
    yardstick_source = ("They read the yardstick's records by run name from your project repo (or --yardstick-repo). "
                        + page.missing("the yardstick's records", fill))
    where, fill = READ_SOURCES["text/text_held_v1_names.jsonl"]
    names_source = ("They read the held pictures' manifest with the name screen from your project repo "
                    "(`text/text_held_v1_names.jsonl`). " + page.missing("text/text_held_v1_names.jsonl", fill))
    force_note = ""
    if entry["blockers"]:
        force_note = ("\n   For this model it refuses until the package can run its lane (see the top of this page); `--force` "
                      "writes it anyway.")
    req = entry.get("general_requests")
    n_text = req["n_text"] if req else lines.GENERAL["n_text"]
    n_held_text = req["n_held_text"] if req else lines.GENERAL["n_held_text"]
    text_source = f" from `{req['requests'].split(':', 1)[1]}`" if req else ""
    as_run = f" {req['as_run']}" if req and req.get("as_run") else ""
    fe = registry.full_entry(entry)
    steps = f"{_n(V12_SETS['apache']['steps'])} steps on the Apache-2.0 set, {_n(V12_SETS['nc']['steps'])} on the\n  non-commercial set"
    if fe:
        ln = entry["lines"][0]
        what_is = "a full fine-tune of " + (entry["about"] if entry["base"].startswith("hub:") else base_text(entry)) + ","
        recipe_bullet = (f"- one pass over the set at {fe['batch']} x {fe['grad_accum']} sequences a step ({steps}), learning rate "
                         f"{_lr(fe['lr'])}, every weight trained but those its family keeps frozen (below); beside it the lane "
                         f"trains a LoRA on the same rows (rank {entry['lora']['rank']}, alpha {entry['lora']['alpha']}, "
                         f"learning rate {_lr(entry['lr'])}), which we have not released (its bars are below);")
        train_note = (f" The full fine-tune runs first, right after the two smokes: it cannot resume (its state is the whole "
                      f"model and its optimizer), so it trains in the runtime's first hours (about {registry.hours(fe, ln)} at "
                      f"{fe['pace'][ln]['s']} s a step) and ships once at the end. Then the LoRA trains.")
        train_state = "The LoRA's whole state"
        reads_note = (f"\nFor the full fine-tune, name its run key in place of the model's (`{entry['key']}` reads the LoRA "
                      f"beside it):\n\n    python -m qwen_symbolic_thinking.reads {fe['key']} {ln} bars --project-repo "
                      "<you>/<project> --out reads\n")
    else:
        what_is = (f"a LoRA fine-tune of {base_text(entry)}" if entry["base"].startswith("hub:") else
                   f"a LoRA fine-tune of [{entry['base']}](https://huggingface.co/{entry['base']})")   # the revision: the settings
        recipe_bullet = (f"- one pass over the set at {entry['batch']} x {entry['grad_accum']} sequences a step ({steps}), LoRA rank "
                         f"{entry['lora']['rank']}, alpha {entry['lora']['alpha']}, learning rate {_lr(entry['lr'])};")
        train_note, train_state, reads_note = "", "Its whole state", ""
    extra_bullets = f"\n- {entry['about_training']}" if entry.get("about_training") else ""
    prepare_note = ""
    if entry.get("prepare"):
        p = entry["prepare"]
        prepare_note = (f"First, only when the project repo lacks it, `{p['name']}` builds the starting model and writes it to "
                        f"`{os.path.dirname(p['makes'][0])}/` in the project repo; every run after it waits for it.\n\n")
    install_note = ""
    if entry["install"]["fast_path"]:
        install_note = " and builds the linear-attention kernels, about 5 minutes"
    elif entry["transformers"]:
        install_note = f" with transformers {entry['transformers'][2:]}"
    text = TEMPLATE.format(
        name=entry["name"], status=status(entry), github=GITHUB, version=version, dataset=lines.DATASET,
        a_items=_n(V12_SETS["apache"]["items"]), a_examples=_n(V12_SETS["apache"]["examples"]),
        n_items=_n(V12_SETS["nc"]["items"]), n_examples=_n(V12_SETS["nc"]["examples"]), what_is=what_is,
        recipe_bullet=recipe_bullet, extra_bullets=extra_bullets, prepare_note=prepare_note, train_note=train_note,
        train_state=train_state, reads_note=reads_note,
        general_kl=registry.V12_RECIPE["general_kl"], anchor_batch=registry.V12_RECIPE["anchor_batch"],
        lines_table=lines_table(entry), settings_table=settings_table(entry), g4=G4, pace_table=pace_table(entry),
        units=UNITS_AN_HOUR, files_table=files, pictures=PICTURES, record=RECORD, key=entry["key"],
        which=which, lane=lane, force_note=force_note, install_note=install_note, ga_anchors=ga["anchors"].split("/")[-1],
        ga_by=ga["anchors_by"].split("/")[-1], ga_text=ga["text"].split("/")[-1], n_picture=lines.GENERAL["n_picture"],
        n_text=n_text, n_held_picture=lines.GENERAL["n_held_picture"], n_held_text=n_held_text, text_source=text_source,
        n_text_pictures=lines.GENERAL_TEXT["n_picture"], n_text_held=lines.GENERAL_TEXT["n_held_picture"],
        general_time=general_time(entry) + as_run, smoke_steps=registry.SMOKE["max_steps"], smoke_items=registry.SMOKE["n_items"],
        stop_rule=stop_rule, late_save=LATE_SAVE_H, comparisons=comparisons(entry), yardstick_runs=yruns,
        yardstick_source=yardstick_source, bars_table=bars_table(entry), names_source=names_source,
        lines_heading="Its lines and their licenses" if len(entry["lines"]) > 1 else "Its line and its license",
        read_line="<" + "|".join(entry["lines"]) + ">" if len(entry["lines"]) > 1 else entry["lines"][0])
    return text, page.placeholders


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Write a model repo's TRAINING.md from the registry.")
    ap.add_argument("model", help=f"a registry key ({', '.join(registry.keys())}), a base model id or a release repo")
    ap.add_argument("--out", required=True, help="the file to write (TRAINING.md)")
    ap.add_argument("--version", default=__version__, help="the package version (tag) the page installs (default: this one)")
    a = ap.parse_args(argv)
    entry = registry.get(a.model)
    text, placeholders = render(entry, a.version)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {a.out} ({entry['name']}; {len(text.splitlines())} lines; nothing uploaded)")
    if placeholders:
        print(f"not published ({len(placeholders)}), each with what it is:")
        for where, what in placeholders:
            print(f"  NOT PUBLISHED {where}: {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
