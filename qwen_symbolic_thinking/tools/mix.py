"""Mixing the lookers' rows. Every looker writes the same pictures (one extraction per looker, in parts); per picture the pool
holds each looker's row, and the mixes are built from the pool with no reference text:

  - the agreement pick: among the rows that hold the row's form, the one whose content words (its thoughts and wordings) overlap
    the other lookers' rows most, as a weighted mean Jaccard (a minimum-Bayes-risk choice);
  - the field picks: the caption, and the prompt line, that the other lookers' captions / prompt lines agree with most;
  - the kind vote: each row's prompt-line first term as one class (painting, print, drawing, photograph, an object word, other);
  - the pools: every distinct caption and prompt line, with the generator word rule applied (the whole words boy / boys / girl /
    girls become young man / young men / young woman / young women, counted), for caption sets that feed a picture generator.

The lookers are weighed by family (scheme "family"): each family the same share, each size an equal part of its family's share,
an adapter and its full fine-tune halves of their size's part; "size" (one share a size) and "each" (one share a looker) are the
alternatives. Agreement among same-family lookers that learned from the same rows is not independent evidence, so the family
scheme keeps one family from outvoting the other by its member count.

The read compares every looker, the picks, a random looker and the oracle on the pictures whose source carries captions (said /
backed of the whole row and of the caption alone, by source, with standard errors), the kind named on the held-out probe split,
and the picked row's backing by agreement decile.

    python -m qwen_symbolic_thinking.tools.mix lookers.json --manifest <path> [--out <dir>] [--data <root>] [--offline] [--ship]
    from qwen_symbolic_thinking.tools import mix; mix.run(lookers=[...], manifest="hub:...")     (the queue's tool kind)

A looker is {"key": ..., "family": ..., "size": ..., "runs": [extraction run names], "rows": true}; "rows": false leaves its
per-looker table out of the output (a looker whose rows already have a split of their own)."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re

from .. import chain
from . import medium, store

DATA_REPO = store.DATA_REPO
WORD_RULE = re.compile(r"\b(boys|boy|girls|girl)\b", re.IGNORECASE)
REPLACEMENT = {"boy": "young man", "boys": "young men", "girl": "young woman", "girls": "young women"}
SCORE_INTS = ("full_ok", "in_range", "wordings_ok", "questions_in_range", "n_thoughts", "questions_n", "gloss_hits", "remark_hits",
              "stray_lines", "chars")
ROWS_COLUMNS = (["sample_id", "source", "group", "depth", "text", "thoughts", "wordings", "questions", "rejected", "hinted",
                 "training_item", "extraction"] + list(SCORE_INTS) + ["hint_said", "hint_backed"])
KIND_ORDER = ("painting", "print", "drawing", "photograph", "object")


def word_rule(text: str) -> tuple:
    """The text with the whole words boy / boys / girl / girls replaced (case kept), and how many were replaced."""
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        w = m.group(0)
        r = REPLACEMENT[w.lower()]
        if len(w) > 1 and w.isupper():
            return r.upper()
        return r[0].upper() + r[1:] if w[0].isupper() else r
    return WORD_RULE.sub(sub, text or ""), n


def weights(lookers: list, scheme: str = "family") -> dict:
    """Each looker's weight in the agreement and the vote."""
    keys = [lk["key"] for lk in lookers]
    if scheme == "each":
        return {k: 1.0 for k in keys}
    fam = {lk["key"]: lk.get("family") or lk["key"] for lk in lookers}
    size = {lk["key"]: (fam[lk["key"]], lk.get("size") or lk["key"]) for lk in lookers}
    members = collections.Counter(size.values())
    if scheme == "size":
        return {k: 1.0 / members[size[k]] for k in keys}
    if scheme != "family":
        raise ValueError(f"unknown scheme {scheme!r}")
    sizes_of = collections.defaultdict(set)
    for k in keys:
        sizes_of[fam[k]].add(size[k])
    return {k: 1.0 / len(sizes_of) / len(sizes_of[fam[k]]) / members[size[k]] for k in keys}


def holds_form(row: dict) -> bool:
    """A row that holds the whole row's form (thoughts, wordings by depth, formed questions) and is not an error."""
    return bool((row.get("scores") or {}).get("full_ok")) and not (row.get("text") or "").startswith("<error")


def wordings_of(row: dict) -> dict:
    w = row.get("wordings") or {}
    return json.loads(w) if isinstance(w, str) else w


def thoughts_of(row: dict) -> list:
    t = row.get("thoughts") or []
    return json.loads(t) if isinstance(t, str) else t


def row_words(row: dict) -> set:
    """The content words the agreement compares: the thoughts and the wordings."""
    return chain.words(" ".join([t.get("text", "") for t in thoughts_of(row)] + list(wordings_of(row).values())))


def prompt_of(row: dict) -> str:
    w = wordings_of(row)
    return w.get("prompt") or w.get("tags") or ""


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def agreement(words: dict, w: dict) -> dict:
    """Each key's weighted mean Jaccard against the other keys present."""
    out = {}
    for k, wk in words.items():
        num = den = 0.0
        for j, wj in words.items():
            if j != k:
                num += w[j] * jaccard(wk, wj)
                den += w[j]
        out[k] = num / den if den else None
    return out


def pick(scores: dict, order: list):
    """The key with the highest score; ties go to the earlier looker in order; a lone key is picked."""
    best = None
    for k in order:
        if k in scores and (best is None or (scores[k] or 0.0) > (scores[best] or 0.0)):
            best = k
    return best


def kind_class(term: str) -> str:
    for kind in ("painting", "print", "drawing", "photograph"):
        if medium.names(term, kind):
            return kind
    return "objectish" if medium.names(term, "object") else "other"


def kind_right(cls: str, kind: str) -> bool:
    return cls in ("photograph", "objectish") if kind == "object" else cls == kind


def kind_vote(terms: dict, w: dict, order: list) -> tuple:
    """The winning class of the first terms and its share of the weight; ties go to the earliest looker's class."""
    tally = collections.Counter()
    for k, t in terms.items():
        tally[kind_class(t)] += w[k]
    if not tally:
        return "", 0.0
    top = max(tally.values())
    for k in order:
        if k in terms and tally[kind_class(terms[k])] == top:
            return kind_class(terms[k]), top / sum(tally.values())
    return "", 0.0


def _ruled_pool(texts: list) -> tuple:
    out, seen, n = [], set(), 0
    for t in texts:
        r, c = word_rule(t)
        n += c
        key = r.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(r.strip())
    return out, n


def mix_picture(rows: dict, w: dict, order: list) -> dict:
    """One picture's mix from its rows (looker key -> row)."""
    form = {k: rows[k] for k in order if k in rows and holds_form(rows[k])}
    agree = agreement({k: row_words(r) for k, r in form.items()}, w)
    pk = pick(agree, order)
    caps = {k: wordings_of(r).get("caption", "").strip() for k, r in form.items() if wordings_of(r).get("caption", "").strip()}
    cap_agree = agreement({k: chain.words(v) for k, v in caps.items()}, w)
    cap_pk = pick(cap_agree, order)
    prompts = {k: prompt_of(r).strip() for k, r in form.items() if prompt_of(r).strip()}
    pr_agree = agreement({k: chain.words(v) for k, v in prompts.items()}, w)
    pr_pk = pick(pr_agree, order)
    terms = {k: medium.first_term(rows[k].get("text") or "") for k in order if k in rows}
    terms = {k: t for k, t in terms.items() if t}
    cls, share = kind_vote(terms, w, order)
    pool_caps, n_caps = _ruled_pool([caps[k] for k in order if k in caps])
    pool_prompts, n_prompts = _ruled_pool([prompts[k] for k in order if k in prompts])
    cap_text, n_cap = word_rule(caps[cap_pk]) if cap_pk else (None, 0)
    pr_text, n_pr = word_rule(prompts[pr_pk]) if pr_pk else (None, 0)
    return {"n_rows": len(rows), "n_form": len(form), "pick": pk, "pick_score": agree.get(pk) if pk else None,
            "agreement": {k: (round(v, 4) if v is not None else None) for k, v in agree.items()},
            "caption_pick": cap_text, "caption_pick_looker": cap_pk, "prompt_pick": pr_text, "prompt_pick_looker": pr_pk,
            "kind_vote": cls, "kind_share": round(share, 4), "captions": pool_caps, "prompts": pool_prompts,
            "words_replaced": n_caps + n_prompts + n_cap + n_pr}


def caption_scores(caption: str, hint_words: set) -> tuple:
    """Said / backed of a caption alone against the source caption's content words; a missing caption says and backs nothing."""
    if not hint_words:
        return None, None
    cw = chain.words(caption or "")
    return len(cw & hint_words) / len(hint_words), (len(cw & hint_words) / len(cw) if cw else 0.0)


def mean_se(vals: list) -> tuple:
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return 0, None, None
    m = sum(vals) / n
    se = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1) / n) if n > 1 else None
    return n, m, se


class Read:
    """The accumulators of the read; fed one picture at a time."""

    def __init__(self, order: list):
        self.order = order
        self.vol = collections.defaultdict(lambda: collections.defaultdict(list))   # source -> method -> [(said, backed)]
        self.probe = collections.defaultdict(lambda: collections.Counter())          # method -> kind -> right
        self.probe_n = collections.Counter()
        self.deciles = []                                                             # (pick score, pick backed)

    def add(self, source: str, hint: str, rows: dict, m: dict, probe_kind: str | None):
        hw = chain.words(hint or "")
        if hw:
            for src in (source, "all"):
                v = self.vol[src]
                said_b = {}
                for k in self.order:
                    if k in rows:
                        s = rows[k].get("scores") or {}
                        said_b[k] = (s.get("hint_said"), s.get("hint_backed") or 0.0)
                        v[k].append(said_b[k])
                        v[f"caption: {k}"].append(caption_scores(wordings_of(rows[k]).get("caption", ""), hw))
                if said_b:
                    v["random looker"].append((sum(a for a, _ in said_b.values() if a is not None) / len(said_b),
                                               sum(b for _, b in said_b.values()) / len(said_b)))
                    v["oracle (best-backed row)"].append((None, max(b for _, b in said_b.values())))
                    caps = [caption_scores(wordings_of(rows[k]).get("caption", ""), hw) for k in said_b]
                    v["caption: random looker"].append((sum(a for a, _ in caps) / len(caps), sum(b for _, b in caps) / len(caps)))
                if m["pick"]:
                    v["agreement pick"].append(said_b[m["pick"]])
                if m["caption_pick"] is not None:
                    v["caption: agreement pick"].append(caption_scores(m["caption_pick"], hw))
            if m["pick"] and m["pick_score"] is not None:
                self.deciles.append((m["pick_score"], said_b[m["pick"]][1]))
        if probe_kind:
            self.probe_n[probe_kind] += 1
            for k in self.order:
                if k in rows:
                    self.probe[k][probe_kind] += kind_right(kind_class(medium.first_term(rows[k].get("text") or "")), probe_kind)
            self.probe["kind vote"][probe_kind] += kind_right(m["kind_vote"], probe_kind)
            if m["pick"]:
                self.probe["agreement pick"][probe_kind] += kind_right(kind_class(medium.first_term(rows[m["pick"]].get("text") or "")), probe_kind)

    def result(self) -> dict:
        vol = {}
        for src, methods in self.vol.items():
            vol[src] = {}
            for meth, pairs in methods.items():
                n, s, s_se = mean_se([a for a, _ in pairs])
                _, b, b_se = mean_se([b for _, b in pairs])
                vol[src][meth] = {"n": len(pairs), "said": s, "said_se": s_se, "backed": b, "backed_se": b_se}
        probe = {meth: {kd: c[kd] for kd in KIND_ORDER} for meth, c in self.probe.items()}
        dec = []
        if self.deciles:
            xs = sorted(self.deciles)
            for i in range(10):
                part = xs[i * len(xs) // 10:(i + 1) * len(xs) // 10]
                if part:
                    dec.append({"decile": i + 1, "n": len(part), "score_from": round(part[0][0], 4), "score_to": round(part[-1][0], 4),
                                "backed": sum(b for _, b in part) / len(part)})
        return {"volume": vol, "probe": probe, "probe_n": dict(self.probe_n), "deciles": dec}


def _fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def read_markdown(res: dict, order: list) -> str:
    """The read as Markdown tables."""
    out = []
    allv = res["volume"].get("all", {})
    if allv:
        out.append("### Said / backed on every captioned picture (whole row; caption alone)\n")
        out.append("| method | pictures | said | backed | caption said | caption backed |")
        out.append("|---|---:|---|---|---|---|")
        for meth in order + ["random looker", "agreement pick", "oracle (best-backed row)"]:
            r = allv.get(meth)
            if not r:
                continue
            c = allv.get(f"caption: {meth}") or {}
            out.append(f"| {meth} | {r['n']} | {_fmt(r['said'])} ± {_fmt(r['said_se'])} | {_fmt(r['backed'])} ± {_fmt(r['backed_se'])} | "
                       f"{_fmt(c.get('said'))} | {_fmt(c.get('backed'))} |")
        c = allv.get("caption: agreement pick")
        if c:
            out.append(f"| caption pick (the caption the others agree with most) | {c['n']} | - | - | {_fmt(c['said'])} | {_fmt(c['backed'])} |")
        out.append("\n### Backed by source (whole row)\n")
        heads = order + ["random looker", "agreement pick"]
        out.append("| source | pictures | " + " | ".join(heads) + " |")
        out.append("|---|---:|" + "---|" * len(heads))
        for src in sorted(s for s in res["volume"] if s != "all"):
            v = res["volume"][src]
            n = max((x["n"] for x in v.values()), default=0)
            out.append(f"| {src} | {n} | " + " | ".join(_fmt((v.get(h) or {}).get("backed")) for h in heads) + " |")
    if res["probe"]:
        out.append(f"\n### The probe: the kind named rightly ({res['probe_n']})\n")
        out.append("| method | " + " | ".join(KIND_ORDER) + " | sum |")
        out.append("|---|" + "---|" * (len(KIND_ORDER) + 1))
        for meth in order + ["kind vote", "agreement pick"]:
            r = res["probe"].get(meth)
            if r:
                out.append(f"| {meth} | " + " | ".join(str(r[k]) for k in KIND_ORDER) + f" | {sum(r.values())} |")
    if res["deciles"]:
        out.append("\n### The agreement pick's backing by agreement decile (captioned pictures)\n")
        out.append("| decile | pictures | agreement | backed |")
        out.append("|---|---:|---|---|")
        for d in res["deciles"]:
            out.append(f"| {d['decile']} | {d['n']} | {d['score_from']:.3f}-{d['score_to']:.3f} | {d['backed']:.3f} |")
    return "\n".join(out) + "\n"


def rows_record(row: dict, extraction: str) -> dict:
    """A row in the dataset's `rows` config columns (the columns of rows/v1 and rows/v2)."""
    s = row.get("scores") or {}
    rec = {"sample_id": row["sample_id"], "source": row.get("source"), "group": row.get("group"), "depth": row.get("depth"),
           "text": row.get("text"),
           "thoughts": json.dumps(thoughts_of(row), ensure_ascii=False), "wordings": json.dumps(wordings_of(row), ensure_ascii=False),
           "questions": json.dumps(row.get("questions") or [], ensure_ascii=False) if not isinstance(row.get("questions"), str) else row["questions"],
           "rejected": row.get("rejected"), "hinted": bool(row.get("hinted")), "training_item": False, "extraction": extraction}
    for k in SCORE_INTS:
        rec[k] = int(s[k]) if s.get(k) is not None else None
    for k in ("hint_said", "hint_backed"):
        rec[k] = float(s[k]) if s.get(k) is not None else None
    return rec


def rows_table(records: list):
    """The `rows` config's table (the exact schema of rows/v2: strings, bools, int64, double)."""
    import pyarrow as pa
    types = {"hinted": pa.bool_(), "training_item": pa.bool_(), "hint_said": pa.float64(), "hint_backed": pa.float64()}
    types.update({k: pa.int64() for k in SCORE_INTS})
    schema = pa.schema([(c, types.get(c, pa.string())) for c in ROWS_COLUMNS])
    return pa.table({c: [r[c] for r in records] for c in ROWS_COLUMNS}, schema=schema)


def mixed_table(records: list):
    import pyarrow as pa
    schema = pa.schema([("sample_id", pa.string()), ("source", pa.string()), ("group", pa.string()), ("depth", pa.string()),
                        ("n_rows", pa.int64()), ("n_form", pa.int64()), ("pick_looker", pa.string()), ("pick_score", pa.float64()),
                        ("pick_text", pa.string()), ("pick_rejected", pa.string()), ("caption_pick", pa.string()),
                        ("caption_pick_looker", pa.string()), ("prompt_pick", pa.string()), ("prompt_pick_looker", pa.string()),
                        ("kind_vote", pa.string()), ("kind_share", pa.float64()), ("captions", pa.list_(pa.string())),
                        ("prompts", pa.list_(pa.string())), ("words_replaced", pa.int64()), ("qwen_image_render", pa.bool_()),
                        ("agreement", pa.string())])
    return pa.table({f.name: [r[f.name] for r in records] for f in schema}, schema=schema)


def load_rows(looker: dict, root: str, offline: bool) -> dict:
    """sample_id -> row for every run of a looker, each row carrying `_extraction` (its run's folder)."""
    out = {}
    for name in looker["runs"]:
        marker, rows = store.rows_of(name, root, offline)
        if rows is None:
            raise RuntimeError(f"{name}: status {marker.get('status')}")
        folder = ((marker.get("extract") or {}).get("path_in_repo") or "").split("/")[-1]
        for r in rows:
            r["_extraction"] = folder
            out[r["sample_id"]] = r
    return out


def run(lookers: list, manifest: str, scheme: str = "family", data: str = "/content/data", out_dir: str = "/content/mix",
        out_repo: str = DATA_REPO, generator_excluded: tuple = ("sdxl_qwen_phase0",), offline: bool = False, ship: bool = True,
        prefix: str = "mixed/v1", tag: str = "", repo: str | None = None) -> dict:
    """Mix the lookers' rows over the manifest's pictures; write the per-looker tables, the agreement split, the mixed table and the
    read; ship them to the dataset repo (rows/<key>/extracted_rows.parquet, rows/agreement/extracted_rows.parquet,
    <prefix>/mixed.parquet, <prefix>/reads.json, <prefix>/reads.md) in one commit."""
    from ..extract import hint_of, load_manifest
    from ..hub import token
    if not os.environ.get("HF_TOKEN") and not offline:
        tok = token()
        if tok:
            os.environ["HF_TOKEN"] = tok
    os.makedirs(out_dir, exist_ok=True)
    order = [lk["key"] for lk in lookers]
    w = weights(lookers, scheme)
    man = load_manifest(manifest)
    pools = {}
    for lk in lookers:
        pools[lk["key"]] = load_rows(lk, data, offline)
        print(f"  {lk['key']}: {len(pools[lk['key']])} rows", flush=True)
    rd = Read(order)
    per_looker = collections.defaultdict(list)
    tabled = {lk["key"] for lk in lookers if lk.get("rows", True)}
    agree_rows, mixed = [], []
    excluded = set(generator_excluded or ())
    for m in man:
        sid = m["sample_id"]
        rows = {k: pools[k][sid] for k in order if sid in pools[k]}
        if not rows:
            continue
        mx = mix_picture(rows, w, order)
        probe_kind = medium.kind_of_class((m.get("labels") or {}).get("genre") or "") if m.get("split") == "probe" else None
        rd.add(m["source"], hint_of(m), rows, mx, probe_kind)
        for k, r in rows.items():
            if k in tabled:
                per_looker[k].append(rows_record(r, r.get("_extraction", "")))
        first = next(iter(rows.values()))
        pr = rows[mx["pick"]] if mx["pick"] else None
        if pr is not None:
            agree_rows.append(rows_record(pr, pr.get("_extraction", "")))
        mixed.append({"sample_id": sid, "source": m["source"], "group": first.get("group"), "depth": first.get("depth"),
                      "n_rows": mx["n_rows"], "n_form": mx["n_form"], "pick_looker": mx["pick"], "pick_score": mx["pick_score"],
                      "pick_text": pr.get("text") if pr is not None else None, "pick_rejected": pr.get("rejected") if pr is not None else None,
                      "caption_pick": mx["caption_pick"], "caption_pick_looker": mx["caption_pick_looker"],
                      "prompt_pick": mx["prompt_pick"], "prompt_pick_looker": mx["prompt_pick_looker"],
                      "kind_vote": mx["kind_vote"], "kind_share": mx["kind_share"], "captions": mx["captions"], "prompts": mx["prompts"],
                      "words_replaced": mx["words_replaced"], "qwen_image_render": m["source"] in excluded,
                      "agreement": json.dumps(mx["agreement"])})
    res = rd.result()
    res["pictures"] = len(mixed)
    res["picked_from"] = dict(collections.Counter(x["pick_looker"] for x in mixed if x["pick_looker"]))
    res["scheme"], res["weights"], res["lookers"] = scheme, w, lookers
    res["words_replaced"] = sum(x["words_replaced"] for x in mixed)
    md = read_markdown(res, order)
    import pyarrow.parquet as pq
    files = []
    for lk in lookers:
        if lk.get("rows", True) and per_looker[lk["key"]]:
            p = os.path.join(out_dir, f"rows_{lk['key']}.parquet")
            pq.write_table(rows_table(per_looker[lk["key"]]), p, compression="zstd")
            files.append((p, f"rows/{lk['key']}/extracted_rows.parquet"))
    p = os.path.join(out_dir, "agreement.parquet")
    pq.write_table(rows_table(agree_rows), p, compression="zstd")
    files.append((p, "rows/agreement/extracted_rows.parquet"))
    p = os.path.join(out_dir, "mixed.parquet")
    pq.write_table(mixed_table(mixed), p, compression="zstd")
    files.append((p, f"{prefix}/mixed.parquet"))
    for name, body in (("reads.json", json.dumps(res, indent=1, default=str)), ("reads.md", md)):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        files.append((p, f"{prefix}/{name}"))
    print(md, flush=True)
    summary = {"pictures": len(mixed), "agreement_rows": len(agree_rows), "picked_from": res["picked_from"],
               "words_replaced": res["words_replaced"], "files": [f[1] for f in files]}
    allv = res["volume"].get("all", {})
    summary["headline"] = {k: {"said": allv[k]["said"], "backed": allv[k]["backed"], "n": allv[k]["n"]}
                           for k in order + ["random looker", "agreement pick"] if k in allv}
    if ship:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        info = api.create_commit(out_repo, repo_type="dataset", operations=[CommitOperationAdd(dst, src) for src, dst in files],
                                 commit_message=f"the lookers' rows mixed over {len(mixed)} pictures ({tag or 'mix'}): one split per looker, "
                                                "the agreement pick, the mixed table and its read")
        summary["commit"] = getattr(info, "oid", None)
        print("  shipped:", out_repo, summary["commit"], flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lookers", help="a JSON file: the list of lookers")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme", default="family", choices=("family", "size", "each"))
    ap.add_argument("--out", default="mix_out")
    ap.add_argument("--ship", action="store_true", help="ship the outputs to the dataset repo (default: write them locally only)")
    store.add_common(ap, manifest=False)
    a = ap.parse_args(argv)
    lookers = json.load(open(a.lookers, encoding="utf-8"))
    run(lookers, a.manifest, scheme=a.scheme, data=store.root(a.data), out_dir=a.out, offline=a.offline, ship=a.ship)


if __name__ == "__main__":
    main()
