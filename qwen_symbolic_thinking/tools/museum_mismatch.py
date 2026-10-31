"""One kind of the museum read, row by row: every row whose catalogue line describes the kind, with the term the model wrote and the
line itself, so that works on paper (watercolour, gouache, ink) can be told from true misses and from lines the pattern mislabelled.
Usage: python -m qwen_symbolic_thinking.tools.museum_mismatch <run> [--kind painting]"""
import argparse
import re

from . import medium, store


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("run", help="an extraction run name over museum sources")
    ap.add_argument("--kind", default="painting", choices=sorted(medium.ACCEPT))
    ap.add_argument("--width", type=int, default=150, help="characters of the catalogue line to show")
    store.add_common(ap)
    a = ap.parse_args(argv)
    root_dir = store.root(a.data)
    captions = store.museum_captions(root_dir, a.manifest, a.offline)
    m, rows = store.rows_of(a.run, root_dir, a.offline)
    if rows is None:
        raise SystemExit(f"{a.run}: status {m.get('status')}")
    n = named = blank = paper_miss = other_miss = 0
    for r in rows:
        cap = captions.get(r["sample_id"], "")
        if medium.expected(cap) != a.kind:
            continue
        n += 1
        term = medium.first_term(r.get("text") or "")
        if not term:
            blank += 1
            flag = "BLANK"
        elif medium.names(term, a.kind):
            named += 1
            flag = "ok   "
        elif medium.on_paper(cap):
            paper_miss += 1
            flag = "PAPER"
        else:
            other_miss += 1
            flag = "MISS "
        print(f"{flag} {r.get('source', ''):9s} {term:28s} | {re.sub(r'\s+', ' ', cap)[:a.width]}")
    print(f"\n{a.kind}: {n} catalogue rows; named {named}; no first term {blank}; missed on works on paper {paper_miss}; missed otherwise {other_miss}")


if __name__ == "__main__":
    main()
