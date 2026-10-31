"""A screen's conflicts and length, for the picture rows (`full`) and the specification rows (`spec_full`) of each screen named: the
summary's rulers (thoughts, said / backed, items named, glosses) beside the four conflicts counted on the answers and the first term of
the prompt line. Usage: python -m qwen_symbolic_thinking.tools.screen_conflicts <stamp> [<stamp> ...] [--fetch]"""
import argparse
import collections
import json
import os
import statistics

from . import medium, store

FORMS = ("full", "spec_full")


def read(rows: list) -> dict:
    c = collections.Counter()
    terms = collections.Counter()
    for r in rows:
        text = r.get("text") or ""
        for k, v in medium.conflicts(text).items():
            c[k] += bool(v)
        term = medium.first_term(text)
        if term:
            terms["photograph" if "photograph" in term else ("painting" if "painting" in term else "other")] += 1
    words = statistics.mean(len((r.get("text") or "").split()) for r in rows) if rows else 0.0
    return {"rows": len(rows), "conflicts": dict(c), "terms": terms, "words": words}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("stamps", nargs="+", help="screen stamps (screens/<stamp>/ on the Hub)")
    ap.add_argument("--fetch", action="store_true", help="fetch the screens' files again")
    store.add_common(ap, manifest=False)
    a = ap.parse_args(argv)
    root_dir = store.root(a.data)
    print("| screen | form | rows | thoughts | words | said / backed | items named | unknown q | light | stock mood | mood lines | "
          "definitions | first term photograph / painting / other | glosses |")
    print("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for stamp in a.stamps:
        d = store.screen_dir(stamp, root_dir, fetch=a.fetch and not a.offline)
        with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        for form in FORMS:
            rs = [r for r in rows if r.get("form") == form]
            if not rs:
                continue
            keys = [k for k in summary if k.endswith("| " + form)]
            v = summary[keys[0]] if keys else {}
            s = read(rs)
            n, c = max(1, s["rows"]), s["conflicts"]
            moods = c.get("mood", 0)
            sb = f"{v['hint_said']:.2f} / {v['hint_backed']:.2f}" if v.get("hint_said") is not None else "-"
            tot = max(1, sum(s["terms"].values()))
            print(f"| {stamp} | {form} | {n} | {v.get('mean_thoughts', 0):.2f} | {s['words']:.0f} | {sb} | {v.get('spec_recall') or 0:.2f} | "
                  f"{c.get('unknown', 0)/n:.0%} | {c.get('light', 0)/n:.0%} | {c.get('stock', 0)/max(1, moods):.0%} of {moods} | {moods/n:.0%} | "
                  f"{c.get('defining', 0)/n:.0%} | {s['terms']['photograph']/tot:.0%} / {s['terms']['painting']/tot:.0%} / {s['terms']['other']/tot:.0%} | "
                  f"{v.get('rows_with_gloss', 0):.2f} |")


if __name__ == "__main__":
    main()
