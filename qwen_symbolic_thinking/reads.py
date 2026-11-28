"""The reads of a model's v1.2 line, one command per line (record-only: they read what the lane's runs wrote to the project
repo; no model runs and nothing is written to any repo):

    python -m qwen_symbolic_thinking.reads <model key> <apache|nc> <bars|text> --project-repo <owner/name> --out <folder>
        [--manifest <held manifest>] [--probe-manifest <local copy>] [--yardstick-run <run> --yardstick-tag <tag>]
        [--yardstick-repo <owner/name>] [--keep <run>] [--yardstick-keep <run>] [--records <local folder>] [--dry-run]

The run names are the lane generator's (lines.py): the line's training and the nine reads after it, the comparison reads of
the base model and of the older versions the registry compares (or the generator's default set), with the reads earlier lanes
made under other names (the registry's comparisons, "earlier"), and for the -nc line its Apache-2.0 sibling's reads.

`bars` (after the line's preservation read K_v12_A1_<key>_<by|nc> is done): tools.bars, the line's training against its
line's yardstick (registry.YARDSTICK: an arm of the same line trained without the keeping objective; --yardstick-run and
--yardstick-tag name another, --yardstick-repo reads it from another repo). Writes bars_<read tag>.txt and .json. --keep
names another preservation read for the line (and --yardstick-keep for the yardstick): our Ulisse E4B Apache-2.0 line was read
on K_v12_A1_ge4b_by_t, its held text requests asked apart, because its registered held set holds none (registry.BARS).

`text` (after the line's last read and its comparison reads are done): seven reads, each <stem>.json, .md and .txt:

| output | read | runs, in order |
|---|---|---|
| text_read_v12_<key>_<by,nc> | tools.text_names: rows that name the shown text | the closer look of the base, each older version, the Apache-2.0 sibling (-nc), the line |
| text_read_v12_<key>_<by,nc>_deep | the same | the long look of each |
| text_read_lettering_ask_<key>_<by,nc> | the same | the closer look with the lettering sentence, of each |
| keep_text_read_v12_<key>_<by,nc> | tools.keep_text_names: answers to "what does the text say" | the text preservation read of each that has one |
| lettering_ask_read_text_<key>_<by,nc> | tools.lettering_ask: rows writing lettering | the closer looks of the base, the older versions of the line's own license line and the line; the line's long look; the lettering-sentence looks of the base, those older versions, the sibling and the line |
| lettering_ask_read_ctl_<key>_<by,nc> | the same, on photos without lettering | the two controls (without, with the sentence) of the base, those older versions, the sibling and the line |
| lettering_quote_precision_<key>_<by,nc> | tools.quote_precision: are the quoted words the picture's own | the line's lettering-sentence, closer and long looks; the base's lettering-sentence look |

An older version of "the line's own license line" is one released on that line, or one released as a single line before the
two lines split (version 1 counts for both). These are the sets read by hand for the 2B, 9B and 0.8B v1.2 lines.

The held manifest (--manifest; default repo:text/text_held_v1_names.jsonl in the project repo) is the 300 held pictures with
lettering (text/text_held_v1.jsonl, which every lane's project repo holds) with each picture's name screen added
("name_screen": "flagged" or "clear"), which splits every text read by it. Without it, --manifest repo:text/text_held_v1.jsonl
reads the same numbers without the name split. A comparison read with no marker in the project repo is left out (and said); a
read of the line or its sibling that is missing, or any read whose marker is not done, stops the command before it reads.
The bars' probe kinds come from the picture manifest the probe ran on (its marker names it, a public dataset file of about 75 MB,
kept in the Hub cache); --probe-manifest takes a local copy instead. The token is HF_TOKEN (hub.token()), never printed.
Exit 0 when the reads were written (the bars' verdict is in their table), 2 when they could not be read."""
from __future__ import annotations

import os
import sys

from . import lines, registry
from .tools import bars, keep_text_names, lettering_ask, quote_precision, text_names
from .tools import records as rec

NAMES_MANIFEST = "repo:text/text_held_v1_names.jsonl"
PLAIN_MANIFEST = "repo:text/text_held_v1.jsonl"
OWN = ("line", "sibling")          # the reads a set cannot do without


def on_line(entry: dict, version: dict, line: str) -> bool:
    """Whether an older version belongs to the line's own license line: released on it, or released as a single line (no
    version of the same name on another line)."""
    if version["line"] == line:
        return True
    return not any(v["version"] == version["version"] and v["line"] != version["line"] for v in entry["versions"])


def subjects(entry: dict, line: str) -> list:
    """What a line's reads compare, in order: the base, the older versions (registry order), the Apache-2.0 sibling of an -nc
    line, the line; each {"who", "what", "reads": {kind: run name}, "on_line"}."""
    out = []
    for c in lines.comparison_plan(entry):
        if c["of"] == "base":
            reads = {kind: lines.comparison_name(entry, kind) for kind in c["reads"]}
            what, on = f"the base {entry['base']}", True
        else:
            v = registry.version(entry, c["of"], c.get("line", "apache"))
            if not v["ready"] or not v["subfolder"]:
                continue
            reads = {kind: lines.comparison_name(entry, kind, v) for kind in c["reads"]}
            what, on = f"{entry['name']} {v['version']} ({v['line']})", on_line(entry, v, line)
        reads.update(c.get("earlier") or {})
        out.append({"who": "base" if c["of"] == "base" else "older", "what": what, "reads": reads, "on_line": on})
    if line == "nc" and "apache" in entry["lines"]:
        out.append({"who": "sibling", "what": f"{entry['name']} v1.2 (apache)", "reads": lines.names(entry, "apache")["by_kind"],
                    "on_line": True})
    out.append({"who": "line", "what": f"{entry['name']} v1.2 ({line})", "reads": lines.names(entry, line)["by_kind"], "on_line": True})
    return out


def text_sets(entry: dict, line: str) -> list:
    """The text set of a line: [(output stem, tool, [(run name, who)])], in the order they are read."""
    k, t = entry["key"], lines.TAG[line]
    every = subjects(entry, line)

    def of(kind, subs):
        return [(s["reads"][kind], s["who"]) for s in subs if kind in s["reads"]]

    def who(*names):
        return [s for s in every if s["who"] in names and (s["who"] != "older" or s["on_line"])]
    mine = who("line")
    lettering = who("base", "older", "sibling", "line")        # the older versions of the line's own license line only
    return [(f"text_read_v12_{k}_{t}", "text_names", of("closer", every)),
            (f"text_read_v12_{k}_{t}_deep", "text_names", of("deep", every)),
            (f"text_read_lettering_ask_{k}_{t}", "text_names", of("lettering", every)),
            (f"keep_text_read_v12_{k}_{t}", "keep_text_names", of("keep_text", every)),
            (f"lettering_ask_read_text_{k}_{t}", "lettering_ask",
             of("closer", who("base", "older", "line")) + of("deep", mine) + of("lettering", lettering)),
            (f"lettering_ask_read_ctl_{k}_{t}", "lettering_ask",
             [x for s in lettering for x in of("control", [s]) + of("control_lettering", [s])]),
            (f"lettering_quote_precision_{k}_{t}", "quote_precision",
             of("lettering", mine) + of("closer", mine) + of("deep", mine) + of("lettering", who("base")))]


def bars_set(entry: dict, line: str, yardstick_run: str | None = None, yardstick_tag: str | None = None) -> tuple:
    """(the training's run, its read tag, the yardstick's run, its read tag)."""
    n = lines.names(entry, line)
    probe = n["by_kind"]["probe"]
    tag = probe[len("P_probe_"):]
    assert n["by_kind"]["keep"] == f"K_{tag}", (probe, n["by_kind"]["keep"])
    y = registry.YARDSTICK[line]
    return n["train"], tag, yardstick_run or y["run"], yardstick_tag or y["tag"]


def plan(entry: dict, line: str, which: str, yardstick_run=None, yardstick_tag=None, keep=None, yardstick_keep=None) -> list:
    """The command's plan as lines of text."""
    if which == "bars":
        run, tag, a0_run, a0_tag = bars_set(entry, line, yardstick_run, yardstick_tag)
        return [f"bars_{tag}: {run} (its reads P_probe_{tag}, {keep or 'K_' + tag}, its screen) against the yardstick {a0_run} "
                f"(P_probe_{a0_tag}, {yardstick_keep or 'K_' + a0_tag}, its screen)"]
    out = []
    for stem, tool, runs in text_sets(entry, line):
        out.append(f"{stem} (tools.{tool}): " + " ".join(n for n, _ in runs))
    return out


def run_text(records: rec.Records, entry: dict, line: str, out: str, manifest: str, say=print) -> list:
    """The text set of a line, written under out; returns the files written."""
    sets = text_sets(entry, line)
    present = records.runs()
    own = sorted({n for _, _, runs in sets for n, w in runs if w in OWN and n not in present})
    if own:
        raise rec.RecordError(f"the line's own reads have no marker in the project repo yet: {', '.join(own)}")
    for n in sorted({n for _, _, runs in sets for n, w in runs if w not in OWN and n not in present}):
        say(f"left out: {n} (a comparison read with no marker in the project repo)")
    wanted = [n for _, _, runs in sets for n, _ in runs if n in present]
    for n in dict.fromkeys(wanted):
        records.done(n)                  # every read named must be done before anything is read
    try:
        held = rec.manifest_file(manifest, records)
    except Exception as e:
        raise rec.RecordError(f"the held manifest {manifest} could not be read ({type(e).__name__}). Put text_held_v1_names.jsonl "
                              "(text/text_held_v1.jsonl with each picture's name screen) in the project repo, pass --manifest "
                              f"<its local path>, or read without the name split: --manifest {PLAIN_MANIFEST}") from None
    written = []
    for stem, tool, runs in sets:
        names = [n for n, _ in runs if n in present]
        if not names:
            say(f"\n== {stem}: nothing to read")
            continue
        path = os.path.join(out, stem)
        if tool == "text_names":
            _, table = text_names.run(records, held, names, path)
        elif tool == "keep_text_names":
            _, table = keep_text_names.run(records, held, names, path)
        elif tool == "lettering_ask":
            _, table = lettering_ask.run(records, names, path)
        else:
            _, table = quote_precision.run(records, held, names, path)
        say(f"\n== {stem}\n" + "\n".join(table))
        written += [path + ext for ext in (".json", ".md", ".txt")]
    return written


def run_bars(records: rec.Records, entry: dict, line: str, out: str, yardstick_run=None, yardstick_tag=None,
             yardstick_repo: str | None = None, probe_manifest: str | None = None, say=print, keep: str | None = None,
             yardstick_keep: str | None = None) -> tuple:
    """The bars of a line, written under out; returns (the JSON record, the files written)."""
    run, tag, a0_run, a0_tag = bars_set(entry, line, yardstick_run, yardstick_tag)
    other = rec.HubRecords(yardstick_repo) if yardstick_repo and yardstick_repo != records.repo else None
    try:
        data, table = bars.run(records, run, tag, a0_run, a0_tag, os.path.join(out, f"bars_{tag}"), other, probe_manifest,
                               keep, yardstick_keep)
    finally:
        if other is not None:
            other.close()
    say("\n".join(table))
    return data, [os.path.join(out, f"bars_{tag}{ext}") for ext in (".txt", ".json")]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The reads of a model's v1.2 line (record-only), from the registry's run names.")
    ap.add_argument("model", help=f"a registry key ({', '.join(registry.keys())}), a base model id or a release repo")
    ap.add_argument("line", choices=list(registry.LINES), help="the line: apache or nc")
    ap.add_argument("set", choices=["bars", "text"], help="bars (the pass marks) or text (the seven text and lettering reads)")
    rec.add_source(ap)
    ap.add_argument("--out", help="the folder the outputs are written to")
    ap.add_argument("--manifest", default=NAMES_MANIFEST,
                    help=f"the held pictures' manifest: repo:<path>, a local path or hub:<owner>/<name>/<path> (default {NAMES_MANIFEST})")
    ap.add_argument("--probe-manifest", help="a local copy of the picture manifest the probe ran on (default: fetched)")
    ap.add_argument("--yardstick-run", help="the bars' yardstick run (default: the registry's, by line)")
    ap.add_argument("--yardstick-tag", help="its read tag (default: the registry's, by line)")
    ap.add_argument("--yardstick-repo", help="the repo the yardstick's runs are in (default: the project repo)")
    ap.add_argument("--keep", help="the bars: the line's preservation read (default: the lane's K_<read tag>)")
    ap.add_argument("--yardstick-keep", help="the bars: the yardstick's preservation read (default: K_<its tag>)")
    ap.add_argument("--dry-run", action="store_true", help="print the runs each read takes; read nothing")
    a = ap.parse_args(argv)
    rec.utf8_stdout()
    entry = registry.get(a.model)
    if a.line not in entry["lines"]:
        print(f"{entry['name']} has no {a.line} line (its lines: {', '.join(entry['lines'])})", file=sys.stderr)
        return 2
    print(f"{entry['name']} v1.2, the {a.line} line: the {a.set} reads" + ("" if a.dry_run else f", into {a.out}"))
    for p in plan(entry, a.line, a.set, a.yardstick_run, a.yardstick_tag, a.keep, a.yardstick_keep):
        print("  " + p)
    if a.dry_run:
        return 0
    if not a.out:
        print("name the output folder (--out)", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    try:
        with rec.open_records(a.project_repo, a.records) as records:
            if a.set == "bars":
                _, written = run_bars(records, entry, a.line, a.out, a.yardstick_run, a.yardstick_tag, a.yardstick_repo,
                                      a.probe_manifest, keep=a.keep, yardstick_keep=a.yardstick_keep)
            else:
                written = run_text(records, entry, a.line, a.out, a.manifest)
    except rec.RecordError as e:
        print(f"not read: {e}", file=sys.stderr)
        return 2
    print("\nwrote " + ", ".join(os.path.basename(w) for w in written) + f" in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
