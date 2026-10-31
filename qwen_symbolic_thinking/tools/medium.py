"""The medium term: which kind of picture a catalogue line describes, whether a row's prompt line names it, and the four conflicts a row
can carry (an unknown question, a thought about light, a stock mood word, a defining pattern)."""
import re

# The kind a catalogue line describes, by the first pattern that matches, in this order (photographic processes before the print
# vocabulary, since "platinum print" and "photogravure" are photographs).
KINDS = [("painting", r"\b(oil|tempera|acrylic|watercolou?r|gouache|fresco|painting|painted|encaustic)\b"),
         ("photograph", r"\b(gelatin silver|albumen|photograph|daguerreotype|platinum print|chromogenic|cyanotype|photogravure|silver print)\b"),
         ("print", r"\b(etching|engraving|lithograph|woodcut|aquatint|drypoint|screenprint|mezzotint|linocut|woodblock|print)\b"),
         ("drawing", r"\b(graphite|charcoal|chalk|pastel|pen and ink|drawing|crayon|ink on paper|pencil)\b"),
         ("object", r"\b(porcelain|glass|bronze|marble|wood|silver|gold|ceramic|textile|stoneware|earthenware|ivory|terracotta|iron|steel|"
                    r"brass|jade|cotton|silk|wool|linen|leather|lacquer|enamel|copper|stone|limestone|sandstone|plaster|bone|shell|faience)\b")]
_KINDS = [(kind, re.compile(rx)) for kind, rx in KINDS]

# A museum's own classification (the manifest's `labels.genre` or `spec.form.classification`) as one of the five kinds.
KIND_OF_CLASS = {"drawing": "drawing", "print": "print", "painting": "painting", "photograph": "photograph",
                 "sculpture": "object", "decorative art": "object"}

# The first terms that count as naming each kind (a photograph of an object is a photograph).
ACCEPT = {"painting": {"painting"},
          "print": {"print", "engraving", "etching", "lithograph", "woodcut"},
          "drawing": {"drawing", "sketch"},
          "photograph": {"photograph"},
          "object": {"photograph", "sculpture", "object", "artifact", "artefact", "ceramic", "vessel", "glass", "textile", "statue"}}

STOCK = re.compile(r"\b(gentle|quiet|everyday|unhurried|calm|plain)\b")
DEFINING = re.compile(r"\b(means|is a|is an|refers to|is the term)\b")
PAPER = re.compile(r"\b(watercolou?r|gouache|tempera|paper|parchment|vellum|board|ink)\b")
_LINE = re.compile(r"^([a-z]+):\s*(.*)$")
_NOT_THOUGHTS = ("question", "caption", "prompt", "tags", "critique")


def expected(caption: str):
    """The kind a catalogue line describes, or None when no pattern matches."""
    c = (caption or "").lower()
    for kind, rx in _KINDS:
        if rx.search(c):
            return kind
    return None


def kind_of_class(classification: str):
    """A museum classification as a kind; plurals ("Drawings") are singular first; None when it is not one of the five."""
    c = (classification or "").lower().strip()
    if c.endswith("s") and c[:-1] in KIND_OF_CLASS:
        c = c[:-1]
    return KIND_OF_CLASS.get(c)


def lines_of(text: str) -> list:
    """The row's lines as (step, text) pairs: every line of the form `step: text`."""
    out = []
    for line in (text or "").split("\n"):
        m = _LINE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def first_term(text: str) -> str:
    """The prompt line's first comma-separated term, lower-cased; "" when the row has no prompt line."""
    for step, t in lines_of(text):
        if step == "prompt":
            return t.split(",")[0].strip().lower()
    return ""


def names(term: str, kind: str) -> bool:
    """Whether a first term names the kind ("oil painting" names a painting; "still life photograph" names a photograph)."""
    return bool(term) and any(term.startswith(a) or term.endswith(a) for a in ACCEPT.get(kind, ()))


def conflicts(text: str) -> dict:
    """The four conflicts of one row: an unknown question, a thought about light, a mood line (and a stock word in it), a defining
    pattern in a thought."""
    ls = lines_of(text)
    moods = [t for s, t in ls if s == "mood"]
    return {"unknown": any("[unknown]" in t for s, t in ls if s == "question"),
            "light": any(s == "light" for s, _ in ls),
            "mood": bool(moods),
            "stock": bool(moods) and bool(STOCK.search(moods[0].lower())),
            "defining": any(DEFINING.search(t) for s, t in ls if s not in _NOT_THOUGHTS)}


def on_paper(caption: str) -> bool:
    """Whether a catalogue line describes a work on paper (or parchment, board, silk-like supports named with their medium)."""
    return bool(PAPER.search((caption or "").lower()))
