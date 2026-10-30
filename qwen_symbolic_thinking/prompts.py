"""The prompt forms of the behaviour screen.

    bare    the picture and the instructions: the line form, the steps, the depth's counts, the rules
    shown   the same, with two worked examples written for other pictures (described in a line each, not shown)
    spec    no picture: a specification of one, as records, to be thought through as if seen
    short   the trained form: the picture and one line (spec_short: the records and one line); the form is the model's to know

Every form asks for the same output: one thought per line as `step: text`, then one `caption:` line."""
from __future__ import annotations

from thought_seeds.thought_tools import COUNTS, STEPS

from .chain import render

LOOK = {"simple": "a quick look", "medium": "a closer look", "deep": "a long look"}

RULES = """Rules:
- Every thought is about THIS picture: what is in it, where, how many, in what light, and what that suggests. Never define a word or explain a term in general.
- Counts, colours and places are exact to the picture; say "likely" or "suggests" when you infer.
- No remarks about the task, the prompt, the examples or your own writing.
- End with one line, `caption:` and a one-sentence caption of the picture."""


def _form(depth: str) -> str:
    lo, hi = COUNTS[depth]["thoughts"]
    return (f"Think it through in writing, one thought per line, in this form:\n\n"
            f"step: what you see or work out\n\n"
            f"The steps, in the order a careful looker takes them: {', '.join(STEPS)}. "
            f"Use only the steps the picture needs: {lo} to {hi} thoughts for {LOOK[depth]}.\n\n{RULES}")


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


def spec(depth: str, specification: dict) -> str:
    return (f"There is no picture to see. Here is a specification of one, as records:\n\n{records(specification)}\n\n"
            f"Treat the records as what the picture shows and " + _form(depth)[0].lower() + _form(depth)[1:])


def short(depth: str) -> str:
    """The trained form: the picture and one line; the form of the answer is the model's to know."""
    return f"Look at this picture and think it through in writing, {LOOK[depth]}, ending with a caption."


def spec_short(depth: str, specification: dict) -> str:
    """The trained form for a specification instead of a picture."""
    return (f"There is no picture to see. Here is a specification of one, as records:\n\n{records(specification)}\n\n"
            f"Treat the records as what the picture shows and think it through in writing, {LOOK[depth]}, ending with a caption.")
