"""The museum pictures: for each extraction run named over museum sources (AIC, NGA, Cleveland, the Met), the prompt line's first term
against the kind the catalogue line describes (`medium.expected`), per kind, with the first terms written and the four conflicts.
The painting row understates a model: watercolours and gouaches on paper are named drawings, as the National Gallery's own catalogue
does, and a few object or print lines match the painting pattern ("painted", "this is a painting of"); `museum_mismatch` lists them.
Usage: python -m qwen_symbolic_thinking.tools.museum_read <run> [<run> ...]"""
import argparse
import collections

from . import medium, store

ORDER = ("painting", "print", "drawing", "photograph", "object", "unlabelled")


def read(rows: list, captions: dict) -> dict:
    by = collections.defaultdict(collections.Counter)
    terms = collections.Counter()
    c = collections.Counter()
    for r in rows:
        text = r.get("text") or ""
        term = medium.first_term(text)
        kind = medium.expected(captions.get(r["sample_id"], "")) or "unlabelled"
        if term:
            terms[term] += 1
            by[kind][term] += 1
        for k, v in medium.conflicts(text).items():
            c[k] += bool(v)
    return {"rows": len(rows), "by_kind": by, "terms": terms, "conflicts": dict(c)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("runs", nargs="+", help="extraction run names over museum sources")
    ap.add_argument("--sources", default=",".join(store.MUSEUMS), help="the museum sources, comma-separated")
    store.add_common(ap)
    a = ap.parse_args(argv)
    root_dir = store.root(a.data)
    captions = store.museum_captions(root_dir, a.manifest, a.offline, tuple(a.sources.split(",")))
    summary = []
    for run in a.runs:
        m, rows = store.rows_of(run, root_dir, a.offline)
        if rows is None:
            print(f"== {run}: status {m.get('status')}; {m.get('error', '')}")
            continue
        x = m.get("extract") or {}
        s = read(rows, captions)
        n = max(1, s["rows"])
        c = s["conflicts"]
        print(f"\n== {run}: {s['rows']} rows; {x.get('s_per_row')} s a row")
        print(f"   rows with an unknown question {c.get('unknown', 0)/n:.0%} | a thought about light {c.get('light', 0)/n:.0%} | "
              f"a stock mood word {c.get('stock', 0)/n:.0%} | a defining pattern {c.get('defining', 0)/n:.0%}")
        print("   first terms overall: " + ", ".join(f"{k} {v/n:.0%}" for k, v in s["terms"].most_common(6)))
        cells = []
        for kind in ORDER:
            counter = s["by_kind"].get(kind)
            if not counter:
                continue
            tot = sum(counter.values())
            ok = sum(v for term, v in counter.items() if medium.names(term, kind))
            print(f"   expected {kind:10s} n={tot:3d}: named {ok/tot:.0%}; terms " + ", ".join(f"{k} {v}" for k, v in counter.most_common(4)))
            if kind != "unlabelled":
                cells.append(f"{ok} of {tot}")
        summary.append(f"| {run} | " + " | ".join(cells) + f" | {c.get('defining', 0)/n:.0%} |")
    if summary:
        print("\n| run | painting | print | drawing | photograph | object | defining |")
        print("|---|---|---|---|---|---|---:|")
        print("\n".join(summary))


if __name__ == "__main__":
    main()
