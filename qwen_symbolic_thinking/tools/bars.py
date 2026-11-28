"""The bars (record-only): a trained version's reads against the registered pass marks, beside a yardstick run (A0: an arm of
the same line trained without the keeping objective; the registry's YARDSTICK names the v1.2 lines'):

| bar | pass mark |
|---|---|
| T1 full-form format, full rows | the screen's full-form answers: the share in the row's format, and the share whole (the format, the wordings each depth asks for in the v1.1 rows, the questions in range, each question saying what settles it) >= 0.95 |
| T2 said, backed | the screen's full-form hint said / backed within 0.03 of the yardstick's (a value above the band is reported, never a miss) |
| T3 probe | the held-out probe's prompt lines that name the picture's kind (tools.medium), summed over the five kinds, within 10 of the yardstick's (above the band likewise) |
| K1 drift picture, text | the preservation read's drift per answer token at strength 1 (picture asks, text asks) <= 0.5 x the yardstick's |
| K2 row form | the share of general answers that take the row's form, its excess over the base's <= 0.02 and <= 0.5 x the yardstick's excess |
| K3 ended | the share of general answers that end on their own >= the base's - 0.02 |

What it reads, for the version and for the yardstick: the training's marker (its screen's full-form entry, "<model id> | full",
read on the run's own base), the screen's results (screens/<stamp>_<run>/results.jsonl), the preservation read's marker
K_<tag>, and the probe's rows (P_probe_<tag>'s extraction) with the probe split's kinds from the picture manifest the probe ran
on (its marker names it; a local copy may be given instead). Prints a table and PASS / NOT READY; writes <stem>.txt (the table
as printed) and <stem>.json (every value read, the bars, the verdict).

The preservation read is K_<tag> unless another is named (--keep, --yardstick-keep): a line whose registered held set holds no
text request was read on a second preservation read over held text requests asked apart (Ulisse E4B's Apache-2.0 line,
K_v12_A1_ge4b_by_t; the registry's BARS says which). A preservation read without text asks stops with what to pass.

    python -m qwen_symbolic_thinking.tools.bars --project-repo <owner/name> --out <stem> <version run> <read tag>
        <yardstick run> <yardstick tag> [--yardstick-repo <owner/name>] [--probe-manifest <path>] [--keep <run>]
        [--yardstick-keep <run>]"""
from __future__ import annotations

import os
from collections import Counter

from . import medium, probe_read
from . import records as rec

# The wordings a whole row carries at each depth in the v1.1 rows (chain.WORDINGS_V11), kept here as the bars registered them.
SAFE = {"simple": {"caption"}, "medium": {"caption", "prompt", "tags"}, "deep": {"caption", "critique", "prompt", "tags"}}


def t1(records: rec.Records, run: str) -> tuple:
    """(the share of full-form answers in the row's format, the share whole) of the run's screen."""
    found = [f for f in records.listing("screens") if f.startswith("screens/") and f.endswith(f"_{run}/results.jsonl")]
    if len(found) != 1:
        raise rec.RecordError(f"{run}: {len(found)} screens found ({found}), one wanted")
    rows = records.jsonl(found[0])
    full = [r for r in rows if r["form"] == "full"]
    fmt = sum(bool((r.get("scores") or {}).get("format_ok")) for r in full) / len(full)
    ok = 0
    for r in full:
        sc, parsed = r.get("scores") or {}, r.get("parsed") or {}
        words = set((parsed.get("wordings") or {}).keys())
        qs = parsed.get("questions") or []
        ok += bool(sc.get("format_ok")) and words == SAFE.get(r.get("depth"), {"caption"}) and \
            bool(sc.get("questions_in_range")) and all("needs" in q for q in qs)
    return fmt, ok / len(full)


def probe_manifest(records: rec.Records, tag: str) -> str:
    """The picture manifest P_probe_<tag> ran on, as its marker names it."""
    return records.done(f"P_probe_{tag}")["extract"]["manifest"]


def probe_kinds(path: str) -> dict:
    """sample_id -> kind for the probe split of a local manifest (the museum's own classification)."""
    return probe_read.probe_kinds(os.path.dirname(os.path.abspath(path)), path, offline=True)


def probe(records: rec.Records, tag: str, kinds: dict) -> int:
    """The probe rows whose prompt line's first term names the picture's kind, over the five kinds."""
    p = records.done(f"P_probe_{tag}")
    rows = records.jsonl(f"{p['extract']['path_in_repo']}/rows.jsonl")
    first = Counter()
    for r in rows:
        kind = kinds.get(r["sample_id"])
        if kind:
            first[kind] += medium.names(medium.first_term(r.get("text") or ""), kind)
    return sum(first.values())


def values(records: rec.Records, run: str, tag: str, kinds: dict, keep: str | None = None) -> dict:
    """Every value the bars read for one run (the version, or the yardstick); keep: its preservation read (default K_<tag>)."""
    m = records.done(run)
    fulls = [v for key, v in m["screen"].items() if key.endswith(" | full")]     # the run's own base: "<model id> | full"
    if len(fulls) != 1:
        raise rec.RecordError(f"{run}: {len(fulls)} full-form screen entries ({list(m['screen'])}), one wanted")
    full = fulls[0]
    keep = keep or f"K_{tag}"
    k = records.done(keep)["tool"]
    s1 = k["by_scale"]["1.0"]
    if "text-held" not in s1["drift"]:
        raise rec.RecordError(f"{keep}: its held set holds no text request, so K1's text bar has nothing to read; name a "
                              "preservation read over held text requests (--keep <run>)")
    return {"t1": list(t1(records, run)), "said": full["hint_said"], "backed": full["hint_backed"], "probe": probe(records, tag, kinds),
            "kl_picture": s1["drift"]["picture-held"]["kl_mean"], "kl_text": s1["drift"]["text-held"]["kl_mean"],
            "kl_all": s1["drift"]["all"]["kl_mean"], "form": s1["takeover"]["all"]["takes_row_form"],
            "ended": s1["takeover"]["all"]["ended"], "base_form": k["base"]["takeover"]["all"]["takes_row_form"],
            "base_ended": k["base"]["takeover"]["all"]["ended"]}


def check(v: dict, a0: dict) -> tuple:
    """The bars: ([(bar, value, pass rule, read, note)], the bars missed)."""
    rows, misses = [], []

    def bar(name, value, rule, ok, note=""):
        rows.append((name, value, rule, "pass" if ok else "MISS", note))
        if not ok:
            misses.append(name)

    bar("T1 full-form format", f"{v['t1'][0]:.3f}", ">= 0.95", v["t1"][0] >= 0.95)
    bar("T1 full rows", f"{v['t1'][1]:.3f}", ">= 0.95", v["t1"][1] >= 0.95)
    for key in ("said", "backed"):
        d = v[key] - a0[key]
        bar(f"T2 {key}", f"{v[key]:.3f}", f"within 0.03 of A0 {a0[key]:.3f}", d >= -0.03,
            "outside the band upward" if d > 0.03 else "")
    dp = v["probe"] - a0["probe"]
    bar("T3 probe", str(v["probe"]), f"within 10 of A0 {a0['probe']}", dp >= -10, "outside the band upward" if dp > 10 else "")
    for part in ("picture", "text"):
        bar(f"K1 drift {part}", f"{v['kl_' + part]:.3f}", f"<= 0.5 x A0 {a0['kl_' + part]:.3f}",
            v["kl_" + part] <= 0.5 * a0["kl_" + part])
    excess, a0_excess = v["form"] - v["base_form"], a0["form"] - a0["base_form"]
    bar("K2 row form", f"{v['form']:.3f}", f"excess <= 0.02 and <= 0.5 x A0's {a0_excess:.3f}",
        excess <= 0.02 and excess <= 0.5 * a0_excess)
    bar("K3 ended", f"{v['ended']:.3f}", f">= base {v['base_ended']:.3f} - 0.02", v["ended"] >= v["base_ended"] - 0.02)
    return rows, misses


def table(run: str, a0_run: str, v: dict, a0: dict, rows: list, misses: list) -> list:
    out = [f"{run} against {a0_run} (drift all {v['kl_all']:.3f} against {a0['kl_all']:.3f})",
           "| bar | value | pass rule | read | note |", "|---|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("PASS: every bar" if not misses else f"NOT READY: {', '.join(misses)}")
    return out


def run(records: rec.Records, version_run: str, tag: str, a0_run: str, a0_tag: str, out_stem: str | None = None,
        yardstick: rec.Records | None = None, probe_manifest_path: str | None = None, keep: str | None = None,
        a0_keep: str | None = None) -> tuple:
    """The bars of a version against its yardstick: (the JSON record, the table's lines); written to out_stem.{txt,json}.
    yardstick: the records the yardstick's runs are in (default: the version's). probe_manifest_path: a local copy of the
    picture manifest the version's probe ran on (default: the manifest its marker names, fetched). keep, a0_keep: the
    preservation reads to read (default K_<tag>, K_<yardstick tag>)."""
    yardstick = yardstick or records
    spec = probe_manifest_path or probe_manifest(records, tag)
    kinds = probe_kinds(rec.manifest_file(spec, records))
    keep, a0_keep = keep or f"K_{tag}", a0_keep or f"K_{a0_tag}"
    v, a0 = values(records, version_run, tag, kinds, keep), values(yardstick, a0_run, a0_tag, kinds, a0_keep)
    rows, misses = check(v, a0)
    lines = table(version_run, a0_run, v, a0, rows, misses)
    data = {"version": {"run": version_run, "tag": tag, "keep": keep, **v},
            "yardstick": {"run": a0_run, "tag": a0_tag, "keep": a0_keep, **a0},
            "probe_manifest": spec, "bars": [dict(zip(("bar", "value", "rule", "read", "note"), r)) for r in rows],
            "pass": not misses, "misses": misses}
    if out_stem:
        rec.write(out_stem, data, lines, md=False)
    return data, lines


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The bars: a version's reads against the registered pass marks, beside a yardstick.")
    ap.add_argument("version_run")
    ap.add_argument("tag", help="the version's read tag: its probe is P_probe_<tag>, its preservation read K_<tag>")
    ap.add_argument("yardstick_run")
    ap.add_argument("yardstick_tag")
    rec.add_source(ap)
    ap.add_argument("--yardstick-repo", help="the repo the yardstick's runs are in (default: the project repo)")
    ap.add_argument("--probe-manifest", help="a local copy of the picture manifest the probe ran on (default: fetched)")
    ap.add_argument("--keep", help="the version's preservation read (default: K_<tag>)")
    ap.add_argument("--yardstick-keep", help="the yardstick's preservation read (default: K_<yardstick tag>)")
    ap.add_argument("--out", required=True, help="the output stem: <stem>.txt and <stem>.json")
    a = ap.parse_args(argv)
    rec.utf8_stdout()
    with rec.open_records(a.project_repo, a.records) as records:
        other = rec.HubRecords(a.yardstick_repo) if a.yardstick_repo and a.yardstick_repo != a.project_repo else None
        try:
            _, lines = run(records, a.version_run, a.tag, a.yardstick_run, a.yardstick_tag, a.out, other, a.probe_manifest,
                           a.keep, a.yardstick_keep)
        finally:
            if other is not None:
                other.close()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
