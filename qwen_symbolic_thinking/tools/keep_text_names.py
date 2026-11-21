"""The direct text read: on the held asks about pictures with lettering ("what does the text say"; a text preservation read of
tools.keep_read against the held answers anchors/v11txt_<key>/held.json), how often a model's answer names the shown words,
beside the base model's own answer to the same ask (the preservation read stores both), record-only.

The same test as tools.text_names (a picture's words as whole words of the answer; the chance control against another
picture's words, a seeded derangement) on answers to the asks instead of the trained row: whether a model still reads text when
asked, apart from whether its row carries it. Read at each strength the preservation read ran (1.0 and 0.5); with a manifest
whose lines carry "name_screen", each strength is also read on the pictures the name screen flags and on the others
("<strength> name_flagged", "<strength> name_clear"). The answers are matched to the manifest by picture: a manifest line
"txt-coco-<id>" is the answer's seed id "coco:<id>".

    python -m qwen_symbolic_thinking.tools.keep_text_names --project-repo <owner/name> --manifest <manifest> --out <stem>
        <preservation read run name> [...]

(a run name is a queue run's, e.g. KT_v12_2b_by; its rows are keep/<stamp>_<run>/rows.jsonl). Writes <stem>.json, <stem>.md and
<stem>.txt (the table as printed)."""
from __future__ import annotations

import random
import re
from collections import defaultdict

from . import records as rec

SEED = 20261113


def truth_by_seed(rows: list) -> dict:
    """The manifest by the answers' seed id: txt-coco-<id> -> coco:<id>."""
    return {"coco:" + m["sample_id"].split("-")[-1]: m for m in rows}


def other_words(truth: dict, seed: int = SEED) -> dict:
    order = sorted(truth)
    random.Random(seed).shuffle(order)
    return {sid: set(truth[order[(k + 1) % len(order)]]["text_words"]) for k, sid in enumerate(order)}


def tokens(s) -> set:
    return set(re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).split())


HEAD = ["| keep read | strength | answers | adapter names the text | base names the text | adapter words named | "
        "base words named | chance (adapter) | answers like the base (f1) |", "|---|---|---|---|---|---|---|---|---|"]


def read(records: rec.Records, truth: dict, names: list) -> tuple:
    """({name: its read by strength}, the table's lines)."""
    other = other_words(truth)
    results, lines = {}, list(HEAD)
    for name in names:
        marker = records.done(name)
        path = marker["tool"]["path_in_repo"]
        rows = records.jsonl(f"{path}/rows.jsonl")
        by = defaultdict(lambda: defaultdict(float))
        for r in rows:
            m = truth.get(r.get("seed_id"))
            if not m:
                continue
            words = set(m["text_words"])
            t, bt = tokens(r.get("text")), tokens(r.get("base_text"))
            # THE NAME SPLIT: with a manifest that carries name_screen, each answer is also counted under
            # "<strength> name_flagged" or "<strength> name_clear" (the stored strength keys unchanged)
            for key in [str(r["scale"])] + ([f"{r['scale']} name_{m['name_screen']}"] if m.get("name_screen") else []):
                b = by[key]
                b["n"] += 1
                b["tuned"] += bool(words & t)
                b["base"] += bool(words & bt)
                b["tuned_recall"] += len(words & t) / len(words)
                b["base_recall"] += len(words & bt) / len(words)
                b["chance"] += bool(other[r["seed_id"]] & t)
                b["f1"] += r.get("f1") or 0.0
        results[name] = {"path": path, "adapter_subfolder": marker["tool"].get("adapter_subfolder"),
                         "by_scale": {s: {k: (round(v / b["n"], 3) if k != "n" else int(v)) for k, v in b.items()}
                                      for s, b in by.items()}}
        for s, b in results[name]["by_scale"].items():
            lines.append(f"| {name} | {s} | {b['n']} | {b['tuned']:.3f} | {b['base']:.3f} | {b['tuned_recall']:.3f} | "
                         f"{b['base_recall']:.3f} | {b['chance']:.3f} | {b['f1']:.3f} |")
    return results, lines


def run(records: rec.Records, manifest, names: list, out_stem: str | None = None) -> tuple:
    """The read of the named preservation reads against the manifest: (results, lines); written to out_stem.{json,md,txt}."""
    truth = truth_by_seed(rec.manifest_rows(manifest, records))
    results, lines = read(records, truth, names)
    if out_stem:
        rec.write(out_stem, results, lines)
    return results, lines


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The direct text read: how often an answer to 'what does the text say' names it.")
    ap.add_argument("names", nargs="+", help="preservation read run names (their markers are under runs/done/)")
    rec.add_source(ap)
    ap.add_argument("--manifest", required=True, help="the held pictures' manifest: repo:<path> or a local path")
    ap.add_argument("--out", required=True, help="the output stem: <stem>.json, .md and .txt")
    a = ap.parse_args(argv)
    rec.utf8_stdout()
    with rec.open_records(a.project_repo, a.records) as records:
        _, lines = run(records, a.manifest, a.names, a.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
