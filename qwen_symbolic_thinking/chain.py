"""The chain of thought as text.

A row's thoughts are rendered the way the model is asked to write them, one line each, then the wordings, then (for the whole
row) the questions a careful looker would still ask:

    see: One man in an apron stands in front of two ovens.
    place: A working kitchen; pans hang on the wall behind him.
    caption: A man in an apron standing by two ovens in a kitchen.
    tags: 1boy, solo, apron, kitchen, oven
    question: person age -> unknown [unknown]

The model's text is parsed back into thoughts, wordings and questions, and a parsed chain is scored against the row it answers:
the number of thoughts against the depth's counts, the steps used, the specification's facts present in the text, the wordings
and questions the depth asks for, the grounding against the picture's own captions when the row carries them, and the curation
detectors (a thought that defines a word instead of looking at the picture; a remark about the task instead of the picture)."""
from __future__ import annotations

import re

from thought_seeds.curate import gloss, remark
from thought_seeds.thought_tools import COUNTS, STEPS

WORDINGS = ("caption", "prompt", "tags", "critique")
WORDINGS_BY_DEPTH = {"simple": 1, "medium": 2, "deep": 4}     # a caption; a caption and a prompt or tags; all four
NEEDS = ("record", "chain", "unknown")
LABELS = set(STEPS) | set(WORDINGS) | {"question"}
NUMBERS = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
           "10": "ten", "20s": "twenties", "30s": "thirties", "40s": "forties", "50s": "fifties", "60s": "sixties",
           "70s": "seventies"}
SPELLINGS = {"centre": "center", "colour": "color", "grey": "gray", "theatre": "theater", "harbour": "harbor", "armour": "armor",
             "moustache": "mustache", "jewellery": "jewelry", "pyjamas": "pajamas", "mould": "mold", "plough": "plow"}
SKIP_KEYS = {"from"}                          # a record that points at another record, not a fact of the picture
UNNAMED = "unnamed"                           # a thought written without a step name
STOP = set("""the a an and or of in on at to with from by for into onto over under near beside behind front back left right there this
that these those is are was were be being been has have had it its they them their he she his her him some one two three four five
several many few more most other another same also very just only while where when which who what how than then here out up down
off about around across along through against between among above below picture image photo frame scene shows showing shown seen
see sees looks looking like likely seems appears appear suggests suggest reads read set toward towards side part shot view""".split())
_LINE = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?[*_#\s]*([A-Za-z]+)[*_\s]*(?::|[—–-]\s)\s*(.*\S)\s*$")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_THINK_TOKEN = re.compile(r"</?think>")        # a stray opening or closing token without its pair
_STEP_HEAD = re.compile(r"^(" + "|".join(STEPS) + r")\b[:,.\s—–-]*(.*)$", re.I | re.S)
_QUESTION = re.compile(r"^(.*?)\s*->\s*(.*?)\s*\[(record|chain|unknown)\]\s*$", re.I)
GENERIC_LABELS = ("step", "thought")          # a label copied from the form line instead of a step name
ECHOES = ("what you see or work out",)        # the form line's own body, echoed


def render(row: dict, wordings: tuple = ("caption",), full: bool = False) -> str:
    """A row as `label: text` lines: the thoughts, then the wordings asked for (every one the row has, when full), then, when
    full, the questions as `question: what -> answer [record, chain or unknown]`."""
    lines = [f"{t['step']}: {str(t['text']).strip()}" for t in row.get("thoughts", [])]
    w = row.get("wordings") or {}
    for k in (WORDINGS if full else wordings):
        if w.get(k):
            v = w[k]
            lines.append(f"{k}: {(', '.join(v) if isinstance(v, list) else str(v)).strip()}")
    if full:
        for q in row.get("questions") or []:
            if isinstance(q, dict):
                if q.get("key"):
                    lines.append(f"question: {q['key']} -> {q.get('value')} [{q.get('needs') or 'unknown'}]")
            elif q:
                lines.append(f"question: {q}")
    return "\n".join(lines)


def parse(text: str) -> dict:
    """The model's text as {thoughts: [{step, text}], wordings: {label: text}, questions: [...], stray: [lines], thinking: str}.

    A labelled line starts a thought (a step name), a wording or a question; a bare line continues the thought before it; a bare
    line before any thought is stray (an introduction, a heading). Markdown bold, list markers and numbering around a label are
    tolerated. A line labelled with the form's own word (`step: count three zebras`) is read by its first word: a step name names
    the thought, anything else leaves it unnamed; an echo of the form line itself is stray. A question is read as its key, its
    answer and what settles it when written in that form, else kept as text."""
    thinking = " ".join(m.group(0)[7:-8].strip() for m in _THINK.finditer(text))
    text = _THINK_TOKEN.sub("", _THINK.sub("", text))
    thoughts, wordings, questions, stray = [], {}, [], []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        label = m.group(1).lower() if m else None
        body = m.group(2).strip().lstrip("*_ ").strip() if m else raw.strip()
        if label in GENERIC_LABELS:
            if body.lower().rstrip(".") in ECHOES:
                stray.append(raw.strip())
                continue
            h = _STEP_HEAD.match(body)
            label, body = (h.group(1).lower(), h.group(2).strip() or body) if h else (UNNAMED, body)
        if label in STEPS or label == UNNAMED:
            thoughts.append({"step": label, "text": body})
        elif label == "question":
            q = _QUESTION.match(body)
            questions.append({"key": q.group(1).strip(), "value": q.group(2).strip(), "needs": q.group(3).lower()} if q else {"text": body})
        elif label in WORDINGS:
            wordings[label] = body
        elif thoughts and not wordings and not questions:
            thoughts[-1]["text"] = (thoughts[-1]["text"] + " " + body).strip()
        else:
            stray.append(raw.strip())
    return {"thoughts": thoughts, "wordings": wordings, "questions": questions, "stray": stray, "thinking": thinking}


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
    """Is a fact's value in the text, as itself, as the number word it stands for, or in the other English spelling?"""
    v, t = value.lower(), text.lower()
    if v in NUMBERS:
        return NUMBERS[v] in t or re.search(rf"\b{re.escape(v)}\b", t) is not None
    if v in t:
        return True
    other = v
    for british, american in SPELLINGS.items():
        other = other.replace(british, american)
    return other != v and other in t


def words(text: str) -> set:
    """The content words of a text: letters only, three or more, not a stop word."""
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) >= 3 and w not in STOP}


def score(parsed: dict, row: dict) -> dict:
    """What a screen reads off one parsed chain, against the row it answers."""
    depth = row.get("depth", "medium")
    lo, hi = COUNTS[depth]["thoughts"]
    qlo, qhi = COUNTS[depth]["questions"]
    thoughts = parsed["thoughts"]
    named = [t for t in thoughts if t["step"] != UNNAMED]
    answer = " ".join([t["text"] for t in thoughts] + list(parsed["wordings"].values()))
    text = " ".join([answer] + parsed["stray"])                       # every fact anywhere
    facts = leaves(row.get("spec") or {})
    hits = [f for f in facts if present(f, text)]
    row_steps, out_steps = {t["step"] for t in row.get("thoughts", [])}, {t["step"] for t in named}
    union = row_steps | out_steps
    questions = parsed.get("questions", [])
    grounded = " ".join([t["text"] for t in thoughts] + [parsed["wordings"].get("caption", "")])   # the chain and its caption
    hint_words, chain_words = words(row.get("hint") or ""), words(grounded)
    wordings_ok = len(parsed["wordings"]) == WORDINGS_BY_DEPTH.get(depth, 1) and bool(parsed["wordings"].get("caption"))
    questions_in_range = qlo <= len(questions) <= qhi
    return {"n_thoughts": len(thoughts),
            "in_range": lo <= len(thoughts) <= hi,
            "has_caption": bool(parsed["wordings"].get("caption")),
            "format_ok": bool(named) and bool(parsed["wordings"].get("caption")),
            "named_frac": (len(named) / len(thoughts)) if thoughts else None,
            "wordings_n": len(parsed["wordings"]),
            "wordings_ok": wordings_ok,
            "questions_n": len(questions),
            "questions_in_range": questions_in_range,
            "questions_formed": sum(1 for q in questions if "needs" in q),
            "full_ok": bool(named) and wordings_ok and questions_in_range and all("needs" in q for q in questions),
            "stray_lines": len(parsed["stray"]),
            "spec_n": len(facts),
            "spec_recall": (len(hits) / len(facts)) if facts else None,
            "spec_missed": [f for f in facts if f not in hits],
            "hint_said": (len(chain_words & hint_words) / len(hint_words)) if hint_words else None,
            "hint_backed": (len(chain_words & hint_words) / len(chain_words)) if (hint_words and chain_words) else None,
            "gloss_hits": sum(1 for t in thoughts if gloss(t["text"])),
            "remark_hits": sum(1 for t in thoughts if remark(t["text"])),
            "chars": sum(len(t["text"]) for t in thoughts),
            "steps": [t["step"] for t in thoughts],
            "step_jaccard": (len(row_steps & out_steps) / len(union)) if union else None,
            "thinking_chars": len(parsed.get("thinking", ""))}


def reference(row: dict) -> dict:
    """The row's own chain (the whole row), parsed and scored: the ceiling the model's chain is read against."""
    return score(parse(render(row, full=True)), row)
