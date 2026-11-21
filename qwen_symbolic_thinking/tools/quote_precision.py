"""Are the words a row writes as lettering the picture's own? (record-only). For each named extraction of the held pictures with
lettering: the rows that write lettering (a `say` thought, or a quoted pair in a thought, the caption or the prompt); their
QUOTED words (the text inside the quote marks, cleaned as the manifest cleans the shown words: lowercased, every character but
letters and digits a space, words of 3+ characters); against the picture's shown words (the manifest's "text_words"). Per row:
every quoted word is the picture's ("all"), some are ("some"), none are ("none"); a row whose lettering carries no quoted word
of 3+ characters is "unquoted". Beside it the same test against ANOTHER picture's words (a seeded derangement), the rate a quote
matches lettering it was never shown. Split by the manifest's name screen (clear / flagged). Up to 12 examples per extraction,
verbatim, with the picture's shown words ("text_shown").

    python -m qwen_symbolic_thinking.tools.quote_precision --project-repo <owner/name> --manifest <manifest> --out <stem>
        <extraction> [...]

writes <stem>.json, <stem>.md and <stem>.txt (the table as printed)."""
from __future__ import annotations

import random
import re

from . import records as rec

SEED = 20261119
QUOTE = re.compile(r"\"([^\"\n]{1,80})\"|“([^”\n]{1,80})”")
HEAD = ["| extraction | group | pictures | rows writing lettering | quoted words all the picture's | some | none | unquoted | "
        "all + some, against another picture's words |", "|---|---|---|---|---|---|---|---|---|"]


def other_words(truth: dict, seed: int = SEED) -> dict:
    order = sorted(truth)
    random.Random(seed).shuffle(order)
    return {sid: set(truth[order[(k + 1) % len(order)]]["text_words"]) for k, sid in enumerate(order)}


def words(s: str) -> set:
    return {w for w in re.sub(r"[^0-9a-z]+", " ", s.lower()).split() if len(w) >= 3}


def lettering(r: dict) -> tuple:
    """A row's `say` thoughts and the text inside its quoted pairs (the thoughts, the caption, the prompt)."""
    thoughts = [t for t in (r.get("thoughts") or []) if isinstance(t, dict)]
    say = [str(t.get("text", "")) for t in thoughts if t.get("step") == "say"]
    w = r.get("wordings") or {}
    texts = [str(t.get("text", "")) for t in thoughts] + [str(w.get(k) or "") for k in ("caption", "prompt")]
    quotes = [m.group(1) or m.group(2) for x in texts for m in QUOTE.finditer(x)]
    return say, quotes


def cls(q: set, ref: set) -> str:
    if not q:
        return "unquoted"
    found = q & ref
    return "all" if found == q else "some" if found else "none"


def read(records: rec.Records, truth: dict, names: list) -> tuple:
    """({name: {group: counts}}, the table's lines and the examples)."""
    other = other_words(truth)
    report, md, ex_md = {}, list(HEAD), []
    for name in names:
        path = records.done(name)["extract"]["path_in_repo"]
        rows = records.jsonl(f"{path}/rows.jsonl")
        assert {r["sample_id"] for r in rows} <= set(truth), f"{name}: rows outside the manifest"
        rep, shown = {}, []
        for group in ("all", "clear", "flagged"):
            sel = [r for r in rows if group == "all" or truth[r["sample_id"]].get("name_screen") == group]
            c = {"all": 0, "some": 0, "none": 0, "unquoted": 0}
            chance = writing = 0
            for r in sel:
                say, quotes = lettering(r)
                if not (say or quotes):
                    continue
                writing += 1
                q = set().union(*[words(x) for x in quotes]) if quotes else set()
                c[cls(q, set(truth[r["sample_id"]]["text_words"]))] += 1
                chance += cls(q, other[r["sample_id"]]) in ("all", "some")
                if group == "all" and len(shown) < 12:
                    shown.append(f"- {r['sample_id']} (shown: {' | '.join(truth[r['sample_id']]['text_shown'][:8])}): "
                                 + "; ".join([f"say: {x[:160]}" for x in say] + [f"quote \"{x}\"" for x in quotes[:4]]))
            rep[group] = {"pictures": len(sel), "writing": writing, **c, "chance_all_or_some": chance}
            n = len(sel) or 1
            md.append(f"| {name} | {group} | {len(sel)} | {writing} ({writing / n:.3f}) | {c['all']} | {c['some']} | {c['none']} | "
                      f"{c['unquoted']} | {chance} |")
        report[name] = rep
        ex_md += ["", f"{name}: rows that write lettering (first 12)", ""] + shown
    return report, md + ex_md


def run(records: rec.Records, manifest, names: list, out_stem: str | None = None) -> tuple:
    truth = {m["sample_id"]: m for m in rec.manifest_rows(manifest, records)}
    report, lines = read(records, truth, names)
    if out_stem:
        rec.write(out_stem, report, lines, ensure_ascii=False)
    return report, lines


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Quote precision: are the words a row quotes as lettering the picture's own?")
    ap.add_argument("names", nargs="+", help="extraction run names (their markers are under runs/done/)")
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
