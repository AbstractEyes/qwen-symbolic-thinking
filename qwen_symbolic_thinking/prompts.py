"""The prompt forms.

    bare        the picture and the instructions: the line form, the steps, the depth's counts, the rules
    shown       the same, with two worked examples written for other pictures (described in a line each, not shown)
    spec        no picture: a specification of one, as records, to be thought through as if seen
    short       the trained form: the picture and one line, for the chain and a caption
    full        the trained form for the whole row: the chain, the wordings and the questions
    spec_short, spec_full   the two trained forms for a specification instead of a picture

The screen forms ask for one thought per line as `step: text`, then one `caption:` line."""
from __future__ import annotations

from thought_seeds.thought_tools import COUNTS, STEPS

from .chain import render

LOOK = {"simple": "a quick look", "medium": "a closer look", "deep": "a long look"}
WORDS = {"simple": "a caption", "medium": "a caption and a prompt or tags", "deep": "a caption, a prompt, tags and a critique"}

RULES = """Rules:
- Every thought is about THIS picture: what is in it, where, how many, in what light, and what that suggests. Never define a word or explain a term in general.
- Counts, colours and places are exact to the picture; say "likely" or "suggests" when you infer.
- No remarks about the task, the prompt, the examples or your own writing.
- End with one line, `caption:` and a one-sentence caption of the picture."""


def _form(depth: str) -> str:
    lo, hi = COUNTS[depth]["thoughts"]
    return (f"Think it through in writing, one thought per line. Each line is the step's name, a colon, then the thought, "
            f"like `see: two riders on grey horses` or `count: three windows, one of them lit`.\n\n"
            f"The steps a careful looker can take, in their usual order: {', '.join(STEPS)}. "
            f"Pick only the few the picture needs: {lo} to {hi} thoughts in all for {LOOK[depth]}, never the whole list.\n\n{RULES}")


def bare(depth: str) -> str:
    return "Look at this picture. " + _form(depth)


def shown(depth: str, examples: list) -> str:
    shown_examples = "\n\n".join(f"Picture: {ex.get('hint') or ex.get('wordings', {}).get('caption', '')}\n{render(ex)}"
                                 for ex in examples)
    return (f"Look at this picture. {_form(depth)}\n\n"
            f"Two examples of this kind of thinking, written for other pictures (each begins with a short description of its "
            f"picture, which is not shown):\n\n{shown_examples}\n\nNow the picture above.")


def records(spec: dict) -> str:
    """A specification as one record per line: `system attribute: value`, relations as `relation: a rel b`."""
    lines = []
    for system, body in spec.items():
        if system == "relation":
            lines += [f"relation: {r.get('a')} {r.get('rel')} {r.get('b')}" for r in body]
        elif isinstance(body, dict):
            lines += [f"{system} {k}: {v}" for k, v in body.items() if k != "from"]
    return "\n".join(lines)


def _records_preamble(specification: dict) -> str:
    return (f"There is no picture to see. Here is a specification of one, as records:\n\n{records(specification)}\n\n"
            f"Treat the records as what the picture shows and ")


def spec(depth: str, specification: dict) -> str:
    return _records_preamble(specification) + _form(depth)[0].lower() + _form(depth)[1:]


def short(depth: str) -> str:
    """The trained form: the picture and one line; the form of the answer is the model's to know."""
    return f"Look at this picture and think it through in writing, {LOOK[depth]}, ending with a caption."


def spec_short(depth: str, specification: dict) -> str:
    return _records_preamble(specification) + f"think it through in writing, {LOOK[depth]}, ending with a caption."


def _full_tail(depth: str) -> str:
    return (f"think it through in writing, {LOOK[depth]}; then write {WORDS[depth]}, and the questions a careful looker would "
            f"still ask, each as `question: what -> answer [record, chain or unknown]`.")


def full(depth: str) -> str:
    """The trained form for the whole row: the chain, the wordings and the questions."""
    return "Look at this picture and " + _full_tail(depth)


def spec_full(depth: str, specification: dict) -> str:
    return _records_preamble(specification) + _full_tail(depth)


# The system prompt (v1, 2026-10-30): the stable guarantees of the row, sent once before the user line, in training and at
# inference alike. Its rules are the ones the human rows obey; the user line keeps the per-row variables (the depth, the form).
SYSTEM_V1 = """You are a careful looker writing a thought row about one picture, or about a specification standing in for one.

Form: thoughts first, one per line as `step: text`, only the steps this picture needs, in this order: see, name, place, count, relate, form, light, colour, mood, mean, intend, story, compare, change, check, say; the depth sets how many (a quick look 1-2, a closer look 3-5, a long look 6-10). Then `caption:` one sentence; `prompt:` comma-separated, its first term the kind of picture (photograph, painting, drawing, print, sculpture, illustration, render), then what is in it; `tags:` and `critique:` when asked. Then the open questions, `question: what -> answer [record, chain or unknown]`.

Looking: every line fits this picture and few others. Counts, colours and places are exact; inference says "likely" or "suggests". A mood line names the feeling and what in the picture gives it, never a stock mood. Mark a question unknown only for what no picture can show; counts, light, colours and positions are there to be looked at. Never define a word or explain a term in general; never remark on the task or your own writing."""

SYSTEMS = {"v1": SYSTEM_V1}


def system(spec: str | None) -> str | None:
    """The system prompt a spec names: None or '' -> none (the trained convention so far); a version name ('v1') -> that text;
    'hub:<path>' -> the text of that file in the project repo; anything else -> the text itself."""
    if not spec:
        return None
    if spec in SYSTEMS:
        return SYSTEMS[spec]
    if spec.startswith("hub:"):
        from .hub import fetch
        with open(fetch(spec[4:]), encoding="utf-8") as fh:
            return fh.read().strip()
    return spec
