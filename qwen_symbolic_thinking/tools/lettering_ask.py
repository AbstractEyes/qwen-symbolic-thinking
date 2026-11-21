"""The lettering read, its row side (record-only): for each named extraction, how many rows carry a `say` thought (the step
for the words a picture shows), a quoted pair (in a thought, the caption or the prompt), either; how many keep the row's whole
shape (the extraction's own scores: full_ok) and how many the extraction kept. Each extraction is marked by its ask: "trained"
(the trained ask) or "lettering sentence" (the trained ask with the lettering sentence after it). Beside it, for the control
extractions (pictures COCO-Text marks as holding no text), up to eight rows that write lettering anyway, verbatim, to look at:
lettering a picture does not have, or small lettering COCO-Text did not mark. The shown-word read itself is tools.text_names.

    python -m qwen_symbolic_thinking.tools.lettering_ask --project-repo <owner/name> --out <stem> <extraction> [...]

writes <stem>.json, <stem>.md and <stem>.txt (the table as printed)."""
from __future__ import annotations

import re

from . import records as rec

QUOTE = re.compile(r"\"[^\"\n]{1,80}\"|“[^”\n]{1,80}”")
SENTENCE, TRAINED = "lettering sentence", "trained"
HEAD = ["| extraction | ask | rows | kept | whole shape (full_ok) | a `say` thought | a quoted pair | either |",
        "|---|---|---|---|---|---|---|---|"]


def flags(r: dict) -> tuple:
    """A row's `say` thoughts and its quoted pairs (in the thoughts, the caption and the prompt)."""
    thoughts = r.get("thoughts") or []
    say = [t.get("text", "") for t in thoughts if isinstance(t, dict) and t.get("step") == "say"]
    w = r.get("wordings") or {}
    texts = [str(t.get("text", "")) for t in thoughts if isinstance(t, dict)] + [str(w.get(k) or "") for k in ("caption", "prompt")]
    quotes = [m.group(0) for x in texts for m in QUOTE.finditer(x)]
    return say, quotes


def read(records: rec.Records, names: list) -> tuple:
    """({name: its counts and up to eight examples}, the table's lines and the control's examples)."""
    report, md, examples = {}, list(HEAD), []
    for name in names:
        path = records.done(name)["extract"]["path_in_repo"]
        summary = records.json(f"{path}/summary.json")
        ask = SENTENCE if summary.get("ask_suffix") else TRAINED
        if not summary.get("rows"):     # an extraction that selected no picture writes no rows file
            report[name] = {"rows": 0, "manifest": summary.get("manifest"), "split": summary.get("split"), "empty": True}
            md.append(f"| {name} | {ask} | 0 (selected no picture: split {summary.get('split')}) | | | | | |")
            continue
        rows = records.jsonl(f"{path}/rows.jsonl")
        n = len(rows)
        kept = sum(1 for r in rows if not r.get("rejected"))
        whole = sum(1 for r in rows if (r.get("scores") or {}).get("full_ok"))
        say = quoted = either = 0
        shown = []
        for r in rows:
            s, q = flags(r)
            say += bool(s)
            quoted += bool(q)
            either += bool(s or q)
            if (s or q) and len(shown) < 8:
                shown.append({"sample_id": r["sample_id"], "say": s, "quotes": q[:4]})
        report[name] = {"rows": n, "kept": kept, "full_ok": whole, "say": say, "quoted": quoted, "either": either,
                        "ask_suffix": summary.get("ask_suffix"), "adapter": summary.get("adapter"),
                        "adapter_subfolder": summary.get("adapter_subfolder"), "manifest": summary.get("manifest"),
                        "package": summary.get("package"), "examples": shown}
        md.append(f"| {name} | {ask} | {n} | {kept} ({kept / n:.3f}) | {whole} ({whole / n:.3f}) | {say} ({say / n:.3f}) | "
                  f"{quoted} ({quoted / n:.3f}) | {either} ({either / n:.3f}) |")
        if "notext_control" in (summary.get("manifest") or "") and shown:
            examples += ["", f"{name} ({ask}), rows that write lettering on a picture COCO-Text marks as holding none:", ""]
            examples += [f"- {e['sample_id']}: " + "; ".join([f"say: {x}" for x in e["say"]] + [f"quote {x}" for x in e["quotes"]])
                         for e in shown]
    return report, md + examples


def run(records: rec.Records, names: list, out_stem: str | None = None) -> tuple:
    report, lines = read(records, names)
    if out_stem:
        rec.write(out_stem, report, lines, ensure_ascii=False)
    return report, lines


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The lettering read's row side: rows writing lettering, the whole shape kept.")
    ap.add_argument("names", nargs="+", help="extraction run names (their markers are under runs/done/)")
    rec.add_source(ap)
    ap.add_argument("--out", required=True, help="the output stem: <stem>.json, .md and .txt")
    a = ap.parse_args(argv)
    rec.utf8_stdout()
    with rec.open_records(a.project_repo, a.records) as records:
        _, lines = run(records, a.names, a.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
