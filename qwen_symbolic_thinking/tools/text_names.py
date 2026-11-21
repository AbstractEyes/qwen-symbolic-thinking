"""The text read: for each extraction of the held pictures with lettering, the share of pictures whose row NAMES the text the
picture shows, read from the extraction's rows in the project repo and the words in the manifest (record-only: no model runs).

A row names the text when one of the picture's words (the manifest's "text_words": COCO-Text's legible, machine-printed,
English words, letters and digits only, lowercased, three characters or more) appears in it as a whole word (the row lowercased,
every run of other characters a space). Read for the whole row, the chain (the thoughts), the caption, the prompt and the
questions, with the share of each picture's words named (word recall over the whole row), by group of pictures (the manifest's
"group") and over all. Rows are parsed by the package's parser (chain.parse).

THE CHANCE CONTROL: the same whole-row test against ANOTHER picture's words (a seeded derangement: the ids in a seeded order,
each taking its successor's words), the rate at which a row "names" text it was never shown by sharing ordinary words.

THE NAME SPLIT: a manifest line may carry "name_screen": "flagged" (the picture's shown words may hold a person's name, a
credit or a web address, which the rows' writers are not given and may not quote) or "clear"; every row is then also counted
in the group name_flagged or name_clear, and a second table gives each extraction's read on the two.

LOCAL ROWS: a name that is a local file is read as rows, one JSON object a line, {"sample_id", "text"} (the whole row as
written; "chain", "caption", "prompt", "questions" may be given beside it, else they are parsed from the text). THE HINT SPLIT:
a local row may carry "hint_words" (the strings its writer's hint gave; empty for an unaided row); its rows are then also
grouped aided / unaided, and a third table gives the share of the hinted shown words the row names (the hint passed through)
apart from the share of the other shown words (the writer's own reading).

Each extraction's adapter and strength are printed from its own summary ("project" when the adapter lives in the project
repo the records come from). The shown words most often named are listed per extraction; words named equally often are listed
in the order they were first named (rows in their order; within a row, alphabetically).

    python -m qwen_symbolic_thinking.tools.text_names --project-repo <owner/name> --manifest <manifest>[,<manifest>...]
        --out <stem> <extraction or rows file> [...]

writes <stem>.json, <stem>.md and <stem>.txt (the table as printed). A manifest is "repo:<path>" (the project repo), a local
path or "hub:<owner>/<name>/<path>"; every line holds "sample_id", "text_words" and optionally "group" and "name_screen"."""
from __future__ import annotations

import os
import random
import re
from collections import Counter

from .. import chain
from . import records as rec

SEED = 20261113
PARTS = ("row", "chain", "caption", "prompt", "questions")


def load_truth(rows: list) -> dict:
    """sample_id -> manifest line; a sample id twice, or a line without words, stops the read."""
    truth = {}
    for m in rows:
        assert m["sample_id"] not in truth, f"sample_id twice across the manifests: {m['sample_id']}"
        assert m["text_words"], f"no words for {m['sample_id']}"
        truth[m["sample_id"]] = m
    return truth


def other_words(truth: dict, seed: int = SEED) -> dict:
    """THE CHANCE CONTROL's words: each picture's successor's words in a seeded order of the ids."""
    order = sorted(truth)
    random.Random(seed).shuffle(order)
    return {sid: set(truth[order[(k + 1) % len(order)]]["text_words"]) for k, sid in enumerate(order)}


def tokens(s: str) -> set:
    return set(re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).split())


def hint_forms(strings) -> set:
    """The words a hint gave, in both cleanings a manifest may use: each string split on spaces (the DOCCI rule) and each whole
    string joined (COCO-Text's per-annotation cleaning: 'Security Camera' -> securitycamera); letters and digits, lowercase, 3+."""
    out = set()
    for s in strings:
        for w in str(s).split() + [str(s)]:
            c = re.sub(r"[^0-9A-Za-z]+", "", w).lower()
            if len(c) >= 3:
                out.add(c)
    return out


def label(ex: dict, project: str | None) -> str:
    """The adapter an extraction ran with, from its summary: 'none', or '<project | repo name>/<run folder or (root)> x<strength>'."""
    if not ex.get("adapter"):
        return "none"
    repo = "project" if project and ex["adapter"] == project else ex["adapter"].split("/")[-1]
    sub = (ex.get("adapter_subfolder") or "").strip("/").split("/")
    run = sub[-2] if len(sub) >= 2 and sub[-1] == "final" else (sub[-1] or "(root)")
    return f"{repo}/{run} x{ex.get('adapter_scale', 1.0):g}"


def _joined(v) -> str:
    return " ".join(f"{q.get('key', '')} {q.get('value', '')}" if isinstance(q, dict) else str(q) for q in v)


def read(records: rec.Records, truth: dict, names: list) -> tuple:
    """{name: its read} for every extraction (or local rows file) named, and the local files that carry the hint split."""
    other = other_words(truth)
    results, hint_files = {}, set()
    for name in names:
        local = os.path.isfile(name)
        if local:                       # a local rows file: {"sample_id", "text"} a line
            ex, adapter = {"path_in_repo": name, "model": None}, "local file"
            rows = rec.FolderRecords(os.path.dirname(os.path.abspath(name))).jsonl(os.path.basename(name))
        else:
            ex = records.done(name)["extract"]
            adapter = label(ex, records.repo)
            rows = records.jsonl(f"{ex['path_in_repo']}/rows.jsonl")
        by, matched = {}, Counter()
        for r in rows:
            m = truth.get(r["sample_id"])
            if not m:
                continue
            words = set(m["text_words"])
            p = chain.parse(r.get("text") or "")
            parts = {"row": r.get("text") or "",
                     "chain": " ".join(t["text"] for t in p["thoughts"]),
                     "caption": p["wordings"].get("caption", ""),
                     "prompt": p["wordings"].get("prompt", ""),
                     "questions": _joined(p["questions"])}
            for k in (PARTS[1:] if local else ()):    # a LOCAL row may give its parts itself; an extraction's are parsed
                v = r.get(k)
                if v is not None:
                    parts[k] = _joined(v) if isinstance(v, list) else str(v)
            hit = {k: bool(words & tokens(v)) for k, v in parts.items()}
            row_tokens = tokens(parts["row"])
            recall = len(words & row_tokens) / len(words)
            chance = bool(other[r["sample_id"]] & row_tokens)
            matched.update(sorted(words & row_tokens))
            groups = [m["group"].replace("text_held_", ""), "all"] if m.get("group") else ["all"]
            if m.get("name_screen"):                     # THE NAME SPLIT: the picture's name screen, flagged or clear
                groups.append("name_" + m["name_screen"])
            split = local and "hint_words" in r          # THE HINT SPLIT: a local row that says which words its hint gave
            if split:
                hint = hint_forms(r.get("hint_words") or [])
                hinted, own = words & hint, words - hint
                groups += ["aided" if hint else "unaided"]
                hint_files.add(name)
            for g in groups:
                b = by.setdefault(g, {"n": 0, "recall": 0.0, "chance": 0, **{k: 0 for k in PARTS},
                                      "split_n": 0, "hinted_n": 0, "hinted_named": 0.0, "own_n": 0, "own_named": 0.0,
                                      "names_own": 0})
                b["n"] += 1
                b["recall"] += recall
                b["chance"] += chance
                for k in PARTS:
                    b[k] += hit[k]
                if split:
                    b["split_n"] += 1
                    if hinted:
                        b["hinted_n"] += 1
                        b["hinted_named"] += len(hinted & row_tokens) / len(hinted)
                    if own:
                        b["own_n"] += 1
                        b["own_named"] += len(own & row_tokens) / len(own)
                        b["names_own"] += bool(own & row_tokens)
        results[name] = {"path": ex["path_in_repo"], "rows": len(rows), "model": ex.get("model"), "adapter": adapter,
                         "top_matched": matched.most_common(15),
                         "by_line": {g: {"n": b["n"], "recall": round(b["recall"] / b["n"], 3),
                                         "chance": round(b["chance"] / b["n"], 3),
                                         **{k: round(b[k] / b["n"], 3) for k in PARTS},
                                         **({"hint_split": {
                                             "rows": b["split_n"],
                                             "hinted_words_named": round(b["hinted_named"] / b["hinted_n"], 3) if b["hinted_n"] else None,
                                             "rows_with_hinted_words": b["hinted_n"],
                                             "unhinted_words_named": round(b["own_named"] / b["own_n"], 3) if b["own_n"] else None,
                                             "names_an_unhinted_word": round(b["names_own"] / b["own_n"], 3) if b["own_n"] else None,
                                             "rows_with_unhinted_words": b["own_n"]}} if b["split_n"] else {})}
                                     for g, b in by.items()}}
    return results, hint_files


def table(results: dict, hint_files: set) -> list:
    """The read as markdown lines: the table over all pictures, the name split, the hint split, the words most often named."""
    lines = ["| extraction | model | adapter (run) x strength | pictures | names the text: row | chance (another picture's words) "
             "| chain | caption | prompt | questions | words named (recall) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["by_line"]["all"]
        lines.append(f"| {name} | {(r['model'] or '').split('/')[-1]} | {r['adapter']} | {a['n']} | {a['row']:.3f} | {a['chance']:.3f} | "
                     f"{a['chain']:.3f} | {a['caption']:.3f} | {a['prompt']:.3f} | {a['questions']:.3f} | {a['recall']:.3f} |")
    if any(g.startswith("name_") for r in results.values() for g in r["by_line"]):
        lines += ["", "THE NAME SPLIT (name_flagged: the shown words may hold a person's name, a credit or a web address, which the "
                  "rows' writers may not quote; name_clear: the other pictures)",
                  "| extraction | adapter (run) x strength | group | pictures | names the text: row | chance | chain | caption | prompt "
                  "| words named (recall) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            for g in ("name_clear", "name_flagged"):
                b = r["by_line"].get(g)
                if b:
                    lines.append(f"| {name} | {r['adapter']} | {g} | {b['n']} | {b['row']:.3f} | {b['chance']:.3f} | {b['chain']:.3f} | "
                                 f"{b['caption']:.3f} | {b['prompt']:.3f} | {b['recall']:.3f} |")
    if hint_files:
        lines += ["", "THE HINT SPLIT (local rows with hint_words; 'aided' = a hint gave words, 'unaided' = none): of the shown words the "
                  "hint gave, the share the row names (the hint passed through); of the shown words the hint did not give, the share "
                  "named (the writer's own reading) and the share of rows naming at least one",
                  "| rows file | group | rows | names the text | words named | chance | hinted words named | unhinted words named | "
                  "rows naming an unhinted word |", "|---|---|---|---|---|---|---|---|---|"]
        for name in sorted(hint_files):
            for g, b in sorted(results[name]["by_line"].items()):
                h = b.get("hint_split")
                if not h:
                    continue

                def fmt(v):
                    return "-" if v is None else f"{v:.3f}"
                lines.append(f"| {name} | {g} | {h['rows']} | {b['row']:.3f} | {b['recall']:.3f} | {b['chance']:.3f} | "
                             f"{fmt(h['hinted_words_named'])} | {fmt(h['unhinted_words_named'])} | {fmt(h['names_an_unhinted_word'])} |")
    lines += ["", "the shown words most often named (pictures whose row names the word):"]
    for name, r in results.items():
        lines.append(f"- {name}: " + ", ".join(f"{w} {c}" for w, c in r["top_matched"]))
    return lines


def run(records: rec.Records, manifest, names: list, out_stem: str | None = None) -> tuple:
    """The read of the named extractions against the manifest (a spec or a list of specs): (results, lines); written to
    out_stem.{json,md,txt} when given."""
    truth = load_truth(rec.manifest_rows(manifest, records))
    results, hint_files = read(records, truth, names)
    lines = table(results, hint_files)
    if out_stem:
        rec.write(out_stem, results, lines)
    return results, lines


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The text read: how often an extraction's rows name the text a picture shows.")
    ap.add_argument("names", nargs="+", help="extraction run names (their markers are under runs/done/) or local rows files")
    rec.add_source(ap)
    ap.add_argument("--manifest", required=True, help="the held pictures' manifest(s): repo:<path>, a local path, or several joined by commas")
    ap.add_argument("--out", required=True, help="the output stem: <stem>.json, .md and .txt")
    a = ap.parse_args(argv)
    rec.utf8_stdout()
    with rec.open_records(a.project_repo, a.records) as records:
        _, lines = run(records, a.manifest, a.names, a.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
