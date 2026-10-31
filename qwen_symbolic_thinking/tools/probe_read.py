"""The held-out probe: the manifest's `probe` split (500 National Gallery of Art pictures, 100 of each kind: painting, print, drawing,
photograph, object), never in any training set. For each extraction run named, how many rows' prompt lines name the kind, and the
four conflicts. Usage: python -m qwen_symbolic_thinking.tools.probe_read <run> [<run> ...]"""
import argparse
import collections

from . import medium, store

ORDER = ("painting", "print", "drawing", "photograph", "object")


def probe_kinds(root_dir: str, manifest: str | None, offline: bool) -> dict:
    """sample_id -> kind for the probe split (the museum's own classification)."""
    out = {}
    for m in store.manifest_rows(root_dir, manifest, offline):
        if m.get("split") == "probe":
            kind = medium.kind_of_class((m.get("labels") or {}).get("genre") or "")
            if kind:
                out[m["sample_id"]] = kind
    return out


def read(rows: list, probe: dict) -> dict:
    """Per kind: rows named rightly of rows; and the conflicts' counts."""
    by = {k: [0, 0] for k in ORDER}
    c = collections.Counter()
    n = 0
    for r in rows:
        kind = probe.get(r["sample_id"])
        if not kind:
            continue
        n += 1
        by[kind][1] += 1
        by[kind][0] += medium.names(medium.first_term(r.get("text") or ""), kind)
        for k, v in medium.conflicts(r.get("text") or "").items():
            c[k] += bool(v)
    return {"rows": n, "by_kind": by, "conflicts": dict(c)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("runs", nargs="+", help="extraction run names (their markers are under runs/done/)")
    store.add_common(ap)
    a = ap.parse_args(argv)
    root_dir = store.root(a.data)
    probe = probe_kinds(root_dir, a.manifest, a.offline)
    print(f"probe: {len(probe)} rows; by kind {dict(collections.Counter(probe.values()))}")
    print("\n| run | rows | painting | print | drawing | photograph | object | unknown q | light | stock mood | defining |")
    print("|---|---:|---|---|---|---|---|---:|---:|---:|---:|")
    for run in a.runs:
        m, rows = store.rows_of(run, root_dir, a.offline)
        if rows is None:
            print(f"| {run} | {m.get('status')} |")
            continue
        s = read(rows, probe)
        n = max(1, s["rows"])
        cells = " | ".join(f"{s['by_kind'][k][0]} of {s['by_kind'][k][1]}" for k in ORDER)
        c = s["conflicts"]
        print(f"| {run} | {s['rows']} | {cells} | {c.get('unknown', 0)/n:.0%} | {c.get('light', 0)/n:.0%} | {c.get('stock', 0)/n:.0%} | {c.get('defining', 0)/n:.0%} |")


if __name__ == "__main__":
    main()
