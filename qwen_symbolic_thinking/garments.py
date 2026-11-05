"""Garment colors two describers agree on, the camera angle from a body fit, and a picture's facts as one helper line.

A caption model and a booru tagger each say what color a garment is, and each is sometimes wrong. A color goes into a helper
line only where both give the same base color to the same kind of garment (the top, the bottom, a one-piece garment, an outer
layer, the shoes): `agreed(caption, tags)`; shoes or an outer layer enter only where both name one: `worn(caption, tags, slot)`.
`angle(turn)` reads the camera angle from the body fit's hips turn (pose.angles), in the bands pose.facing words.
`records_line(spec, colors, angle, worn)` writes a specification's records with all of it as one line, `system attribute: value`
joined by '; ', in US spelling.
"""
import re

SLOTS = ("upper", "lower", "onepiece", "outer", "shoes")

GARMENTS = {           # a garment word (or two) -> its slot; the longest match wins ("tank top" before "top")
    "upper": ["t-shirt", "tshirt", "tee", "shirt", "top", "tank top", "tank", "camisole", "cami", "blouse", "sweater", "pullover",
              "jumper", "hoodie", "sweatshirt", "cardigan", "polo", "crop top", "tube top", "bodysuit", "turtleneck", "henley"],
    "lower": ["pants", "trousers", "jeans", "shorts", "skirt", "miniskirt", "leggings", "joggers", "sweatpants", "chinos",
              "culottes", "slacks"],
    "onepiece": ["dress", "sundress", "jumpsuit", "romper", "playsuit", "overalls", "gown"],
    "outer": ["jacket", "coat", "blazer", "vest", "parka", "windbreaker", "bomber"],
    "shoes": ["shoes", "footwear", "sneakers", "boots", "sandals", "heels", "high heels", "flats", "loafers", "slippers", "pumps",
              "trainers", "mules", "espadrilles", "slides"],
}
WORD_SLOT = {w: slot for slot, words in GARMENTS.items() for w in words}
LONGEST = max(len(w.split()) for w in WORD_SLOT)

BASE = {               # a color word -> its base color; light / dark / pale and other shade words are skipped
    "white": "white", "ivory": "white", "cream": "white", "off-white": "white",
    "black": "black",
    "gray": "gray", "grey": "gray", "charcoal": "gray", "silver": "gray",
    "red": "red", "burgundy": "red", "maroon": "red", "crimson": "red", "wine": "red", "scarlet": "red",
    "blue": "blue", "navy": "blue", "teal": "blue", "turquoise": "blue", "aqua": "blue", "cobalt": "blue", "indigo": "blue",
    "green": "green", "olive": "green", "mint": "green", "emerald": "green", "sage": "green",
    "yellow": "yellow", "mustard": "yellow", "gold": "yellow", "golden": "yellow", "lemon": "yellow",
    "pink": "pink", "magenta": "pink", "fuchsia": "pink", "rose": "pink", "blush": "pink",
    "purple": "purple", "lavender": "purple", "violet": "purple", "lilac": "purple", "plum": "purple", "mauve": "purple",
    "brown": "brown", "tan": "brown", "camel": "brown", "chocolate": "brown", "rust": "brown", "cognac": "brown",
    "orange": "orange", "coral": "orange", "peach": "orange", "salmon": "orange",
    "beige": "beige", "khaki": "beige", "nude": "beige", "sand": "beige", "taupe": "beige",
}
BREAKS = {",", ";", ".", ":", "with", "over", "under", "underneath", "beneath", "and", "while", "who", "which", "that", "her",
          "his", "their", "holding", "carrying"}
_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*|[,;.:]")
SPELLINGS = {"colour": "color", "grey": "gray"}


def _tokens(text: str) -> list:
    return _TOKEN.findall((text or "").lower())


def _garment_at(tokens: list, i: int):
    """The garment word starting at tokens[i] (the longest one), as (slot, word, length), or None."""
    for n in range(min(LONGEST, len(tokens) - i), 0, -1):
        word = " ".join(tokens[i:i + n])
        if word in WORD_SLOT:
            return WORD_SLOT[word], word, n
    return None


def caption_colors(caption: str) -> dict:
    """{slot: [base colors]} from a caption: the color words since the last break (a comma, 'and' before a non-color word,
    'with', ...) go to the garment word that follows them ('a white and blue striped shirt' -> upper: white, blue)."""
    tokens, out, pending, i = _tokens(caption), {}, [], 0
    while i < len(tokens):
        t = tokens[i]
        found = _garment_at(tokens, i)
        if found:
            slot, _, n = found
            for c in pending:
                if c not in out.setdefault(slot, []):
                    out[slot].append(c)
            pending, i = [], i + n
            continue
        parts = t.split("-")
        if t in BASE:
            pending.append(BASE[t])
        elif len(parts) > 1 and all(p in BASE or p == "and" for p in parts):      # black-and-white
            pending.extend(BASE[p] for p in parts if p in BASE)
        elif t in BREAKS and not (t == "and" and i + 1 < len(tokens) and tokens[i + 1] in BASE):
            pending = []
        i += 1
    return out


def tag_colors(tags) -> dict:
    """{slot: [base colors]} from booru tags ('white shirt', 'black footwear', 'light blue skirt'), a list or one comma line."""
    if isinstance(tags, str):
        tags = tags.split(",")
    out = {}
    for tag in tags:
        if isinstance(tag, (list, tuple)):          # [name, score, category, precision]
            tag = tag[0]
        words = _tokens(str(tag).replace("_", " "))
        for start in range(len(words)):
            found = _garment_at(words, start)
            if found and start + found[2] == len(words):        # the tag ends with the garment word
                for w in words[:start]:
                    if w in BASE and BASE[w] not in out.setdefault(found[0], []):
                        out[found[0]].append(BASE[w])
                break
    return {k: v for k, v in out.items() if v}


def caption_garments(caption: str) -> dict:
    """{slot: [garment words]} a caption names, in order."""
    tokens, out, i = _tokens(caption), {}, 0
    while i < len(tokens):
        found = _garment_at(tokens, i)
        if found:
            if found[1] not in out.setdefault(found[0], []):
                out[found[0]].append(found[1])
            i += found[2]
        else:
            i += 1
    return out


def agreed(caption: str, tags) -> dict:
    """{slot: base color} where the caption and the tags give a garment of that slot the same base color (the caption's first
    such color when they share several); a slot where both name colors and share none is left out."""
    cap, tag = caption_colors(caption), tag_colors(tags)
    out = {}
    for slot in SLOTS:
        shared = [c for c in cap.get(slot, []) if c in tag.get(slot, [])]
        if shared:
            out[slot] = shared[0]
    return out


def angle(turn):
    """The camera angle from the hips' turn (degrees, + toward the subject's left): frontal up to 20, three quarter up to 70,
    profile up to 110; past that the back is to the camera, which no angle names (None). None without a fit."""
    if turn is None:
        return None
    t = abs(float(turn))
    return "frontal" if t <= 20 else "three quarter" if t <= 70 else "profile" if t <= 110 else None


def _us(text: str) -> str:
    for british, american in SPELLINGS.items():
        text = text.replace(british, american)
    return text


def records_line(spec: dict, colors: dict | None = None, camera_angle: str | None = None, worn: dict | None = None) -> str:
    """The specification's records as one line, `system attribute: value; ...`, in US spelling. `worn` adds garments the
    specification lacks ({"shoes": {"name": "sneakers"}}, from `worn()`); the agreed colors are set on the garments it then
    holds (a one-piece garment's color is `wear color`). `camera_angle`: None leaves the specification's angle as it is, a value
    replaces it, "" drops it (the fit reads the back to the camera, which no angle names)."""
    spec = {system: dict(body) for system, body in (spec or {}).items() if isinstance(body, dict)}
    for system, body in (worn or {}).items():
        if system not in spec and body:
            spec[system] = dict(body)
    colors = colors or {}
    for slot in ("upper", "lower", "outer", "shoes"):
        if slot in colors and slot in spec:
            spec[slot]["colour"] = colors[slot]
    if "onepiece" in colors and spec.get("wear", {}).get("clothing"):
        spec["wear"]["colour"] = colors["onepiece"]
    if camera_angle:
        spec.setdefault("shot", {})["angle"] = camera_angle
    elif camera_angle == "" and "shot" in spec:
        spec["shot"].pop("angle", None)
    parts = []
    for system, body in spec.items():
        for attribute, value in body.items():
            if isinstance(value, str) and value.strip():
                parts.append(_us(f"{system} {attribute}: {value.strip()}"))
    return "; ".join(parts)


GENERIC = {"shoes": "shoes", "outer": "jacket"}       # the name when both describers see the slot but name it differently


def worn(caption: str, tags, slot: str) -> dict:
    """A garment of `slot` (shoes or outer) both describers name, as {"name": ...}: the caption's word when the tags name the same
    one, the slot's plain name when they only agree that one is worn ('black footwear' beside 'sneakers'); empty when either
    is silent."""
    cap = caption_garments(caption).get(slot, [])
    if isinstance(tags, str):
        tags = tags.split(",")
    tag_words = set()
    for tag in tags:
        if isinstance(tag, (list, tuple)):
            tag = tag[0]
        words = _tokens(str(tag).replace("_", " "))
        for start in range(len(words)):
            found = _garment_at(words, start)
            if found and found[0] == slot:
                tag_words.add(found[1])
    if not cap or not tag_words:
        return {}
    same = [w for w in cap if w in tag_words]
    return {"name": same[0] if same else GENERIC.get(slot, cap[0])}
