"""The chain of thought as text.

A row's thoughts are rendered the way the model is asked to write them, one line each:

    see: One man in an apron stands in front of two ovens.
    place: A working kitchen; pans hang on the wall behind him.
    caption: A man in an apron standing by two ovens in a kitchen.

The model's text is parsed back into thoughts and wordings, and a parsed chain is scored against the row it answers: the number
of thoughts against the depth's counts, the steps used, the specification's facts present in the text, and the curation
detectors (a thought that defines a word instead of looking at the picture; a remark about the task instead of the picture)."""
from __future__ import annotations

import re

from thought_seeds.curate import gloss, remark
from thought_seeds.thought_tools import COUNTS, STEPS

WORDINGS = ("caption", "prompt", "tags", "critique")
LABELS = set(STEPS) | set(WORDINGS)
NUMBERS = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
           "10": "ten", "20s": "twenties", "30s": "thirties", "40s": "forties", "50s": "fifties", "60s": "sixties",
           "70s": "seventies"}
SKIP_KEYS = {"from"}                          # a record that points at another record, not a fact of the picture
_LINE = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?[*_#\s]*([A-Za-z]+)[*_\s]*(?::|[—–-]\s)\s*(.*\S)\s*$")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def render(row: dict, wordings: tuple = ("caption",)) -> str:
    """A row's thoughts, then the wordings asked for, one `label: text` line each."""
    lines = [f"{t['step']}: {str(t['text']).strip()}" for t in row.get("thoughts", [])]
    w = row.get("wordings") or {}
    lines += [f"{k}: {str(w[k]).strip()}" for k in wordings if w.get(k)]
    return "\n".join(lines)


def parse(text: str) -> dict:
    """The model's text as {thoughts: [{step, text}], wordings: {label: text}, stray: [lines], thinking: str}.

    A labelled line starts a thought (a step name) or a wording; a bare line continues the thought before it; a bare line before
    any thought is stray (an introduction, a heading). Markdown bold, list markers and numbering around a label are tolerated."""
    thinking = " ".join(m.group(0)[7:-8].strip() for m in _THINK.finditer(text))
    text = _THINK.sub("", text)
    thoughts, wordings, stray = [], {}, []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        label = m.group(1).lower() if m else None
        body = m.group(2).strip().lstrip("*_ ").strip() if m else raw.strip()
        if label in STEPS:
            thoughts.append({"step": label, "text": body})
        elif label in WORDINGS:
            wordings[label] = body
        elif thoughts and not wordings:
            thoughts[-1]["text"] = (thoughts[-1]["text"] + " " + body).strip()
        else:
            stray.append(raw.strip())
    return {"thoughts": thoughts, "wordings": wordings, "stray": stray, "thinking": thinking}


def leaves(spec: dict) -> list:
    """The facts a specification states: every string value, except the records that point at other records."""
    out = []

    def walk(node, key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and key not in SKIP_KEYS and node.strip():
            out.append(node.strip())
    walk(spec)
    return out


def present(value: str, text: str) -> bool:
    """Is a fact's value in the text, as itself or as the number word it stands for?"""
    v, t = value.lower(), text.lower()
    if v in NUMBERS:
        return NUMBERS[v] in t or re.search(rf"\b{re.escape(v)}\b", t) is not None
    return v in t


def score(parsed: dict, row: dict) -> dict:
    """What a screen reads off one parsed chain, against the row it answers."""
    depth = row.get("depth", "medium")
    lo, hi = COUNTS[depth]["thoughts"]
    thoughts = parsed["thoughts"]
    text = " ".join([t["text"] for t in thoughts] + list(parsed["wordings"].values()))
    facts = leaves(row.get("spec") or {})
    hits = [f for f in facts if present(f, text)]
    row_steps, out_steps = {t["step"] for t in row.get("thoughts", [])}, {t["step"] for t in thoughts}
    union = row_steps | out_steps
    return {"n_thoughts": len(thoughts),
            "in_range": lo <= len(thoughts) <= hi,
            "has_caption": bool(parsed["wordings"].get("caption")),
            "format_ok": bool(thoughts) and bool(parsed["wordings"].get("caption")),
            "stray_lines": len(parsed["stray"]),
            "spec_n": len(facts),
            "spec_recall": (len(hits) / len(facts)) if facts else None,
            "spec_missed": [f for f in facts if f not in hits],
            "gloss_hits": sum(1 for t in thoughts if gloss(t["text"])),
            "remark_hits": sum(1 for t in thoughts if remark(t["text"])),
            "chars": sum(len(t["text"]) for t in thoughts),
            "steps": [t["step"] for t in thoughts],
            "step_jaccard": (len(row_steps & out_steps) / len(union)) if union else None,
            "thinking_chars": len(parsed.get("thinking", ""))}


def reference(row: dict) -> dict:
    """The row's own chain, parsed and scored: the ceiling the model's chain is read against."""
    return score(parse(render(row)), row)
