"""Absences in a caption: the stock captioners answer a checklist ask ("any eyewear, headwear or jewelry", "any print or text on
it") by naming what is NOT there ("no eyewear, headwear, or jewelry", "with no visible print or text"). drop() removes those
statements for the two families the asks name (eyewear / headwear / jewelry / accessories; print / text / logos) and leaves
every other negative alone ("no visible smile", "no visible garments": they describe the picture).

    from qwen_symbolic_thinking import absences
    absences.drop("A man with short hair and no eyewear, headwear, or jewelry smiles.")   # "A man with short hair smiles."
    absences.left(text)                                                                   # the absence phrases still in a text
"""
from __future__ import annotations

import re

# One absent item, with the qualifiers the captioners put before it; a list of them as the captioners write it.
_ITEM = (r"(?:(?:visible|discernible|apparent|readable|additional|other|printed|graphic)\s+)*"
         r"(?:eye\s?wear|head\s?wear|jewel(?:le)?ry|eyeglasses|sunglasses|glasses|accessor(?:y|ies)|prints?|text|logos?|graphics?)")
_LATER = r"(?:" + _ITEM + r"|(?:(?:visible|discernible|apparent|additional|other)\s+)*(?:garments?|clothing|clothes))"
_LIST = _ITEM + r"(?:\s*,\s*(?:or\s+|and\s+|nor\s+)?" + _LATER + r"|\s+(?:or|and|nor)\s+" + _LATER + r")*"
# What may follow the list inside its clause: "is visible", "appears on any garment", "on the fabric", "anywhere in the image".
_WHERE = r"(?:\s+(?:on|in|anywhere|elsewhere|beyond)\b[^,.;]*)?"
_PRED = (r"(?:\s+(?:is|are)\s+(?:(?:also|otherwise)\s+)?(?:visible|present|apparent|discernible|evident|worn|noticeable)"
         + _WHERE + r"|\s+(?:is|are)\s+(?:on|in)\b[^,.;]*|\s+(?:appears?|visible|present)" + _WHERE
         + r"|\s+(?:on|anywhere|in)\s+[^,.;]*)?")
_NO = r"\bno\s+" + _LIST + _PRED + r"(?!\s*,?\s*(?:or|nor)\b)"   # never cut before an 'or' that goes on with the list
_SHUT = r",(?!\s*(?:(?:or|and|nor)\s+)?" + _LATER + r")"   # a comma that closes the list
_WHO = r"(?:he|she|they|it|both|each|(?:the|his|her|their|its)\s+(?:[\w-]+\s+)?[\w-]+)"   # "her garment", "the front boy", "the t-shirt"
_HAS = r"(?:wears|has|have|is\s+wearing|are\s+wearing|has\s+on|appears\s+to\s+wear|shows|show|features|feature|bears|bear)"
_VERB = r"(?:has|have|wears|wear|is|are|holds|smiles|looks|gazes|appears|shows|sports|carries|stands|sits|poses|has\s+on)"
# A sentence start: the text's start, or after a sentence end (with or without a closing quote).
_START = r"(?:^|(?<=[.!?]\s)|(?<=[.!?][\"\u201d\u2019']\s))"

_RULES = [
    # "No eyewear, headwear, or jewelry is visible." (a whole sentence)
    (re.compile(_START + r"(?:There\s+(?:is|are)\s+)?" + _NO + r"\s*[.!]", re.I), ""),
    # "No eyewear or headwear is visible, and her garment ..." -> "Her garment ..."
    (re.compile(_START + _NO + r"(?:,\s+(?:and|but|while)\s+|\s*;\s*)", re.I), "\u0000"),
    # "she wears no eyewear or headwear, but has on a necklace" -> "she has on a necklace" (a verb follows)
    (re.compile(r"\b(" + _WHO + r")\s+" + _HAS + r"\s+" + _NO + r"(?:" + _SHUT + r")?\s+(?:but|and|while)\s+(?=" + _VERB + r"\b)",
                re.I), r"\1 "),
    # "The child wears no eyewear or jewelry, and only the top of a garment is visible" -> "Only the top of a garment ..."
    (re.compile(_START + _WHO + r"\s+" + _HAS + r"\s+" + _NO + r"(?:" + _SHUT + r")?\s+(?:but|and|while)\s+", re.I), "\u0000"),
    (re.compile(r"\b" + _WHO + r"\s+" + _HAS + r"\s+" + _NO + r"(?:" + _SHUT + r")?\s+(?:but|and|while)\s+", re.I), ""),
    # "; he wears no eyewear, headwear, or jewelry." / ", and no print or text is visible" (a clause of its own)
    (re.compile(r"\s*[;,]\s*(?:and\s+|but\s+|though\s+|although\s+|while\s+)?(?:" + _WHO + r"\s+" + _HAS
                + r"\s+|there\s+(?:is|are)\s+)?" + _NO
                + r"(?=\s*[.;]|\s*$)", re.I), ""),
    # ", wearing no eyewear or headwear, and a thin necklace" -> ", wearing a thin necklace"
    (re.compile(r",\s+(wearing|with|having)\s+" + _NO + _SHUT + r"\s+and\s+", re.I), r", \1 "),
    # ", wearing no eyewear or headwear," (a parenthetical)
    (re.compile(r",\s+(?:wearing|with|having)\s+" + _NO + _SHUT, re.I), ""),
    # "short dark hair with bangs, no eyewear or headwear, and a wide smile" -> "short dark hair with bangs and a wide smile"
    (re.compile(r",\s+" + _NO + _SHUT + r"\s+and\s+", re.I), " and "),
    # "a mustache, no eyewear or headwear, gazes at the camera" -> "a mustache gazes at the camera"
    (re.compile(r",\s+" + _NO + _SHUT + r"\s+", re.I), " "),
    # "though no text is visible" / "but no logos are visible on the collar"
    (re.compile(r",?\s+(?:though|although|but|while)\s+" + _NO, re.I), ""),
    # "pants that have no visible print or text" / "a top, which has no print or text and features a round neckline"
    (re.compile(r"\b(that|which)\s+(?:also\s+)?(?:has|have)\s+" + _NO + r"\s+and\s+", re.I), r"\1 "),
    (re.compile(r",?\s+(?:that|which)\s+(?:also\s+)?(?:has|have|features?|bears?|shows?)\s+" + _NO + r"(?=\s*[.,;]|\s*$)", re.I), ""),
    # "both featuring no visible print or text" / "- no visible print or text appears on any garment"
    (re.compile(r",?\s+(?:both\s+|all\s+|each\s+)?(?:featuring|showing|bearing|displaying)\s+" + _NO, re.I), ""),
    (re.compile(r"\s*[\u2013\u2014]\s*" + _NO + r"(?=\s*[.;]|\s*$)", re.I), ""),
    (re.compile(r"\s+-\s+" + _NO + r"(?=\s*[.;]|\s*$)", re.I), ""),
    # "and has no eyewear or headwear" / "and wears no jewelry"
    (re.compile(r",?\s+(?:and|but)\s+" + _HAS + r"\s+" + _NO, re.I), ""),
    # "a plain top with no visible print or text" / "wearing no eyewear" / "and no eyewear, headwear, or jewelry"
    (re.compile(r",?\s+(?:and\s+)?(?:with|wearing|having|and)\s+" + _NO, re.I), ""),
    # "He wears no eyewear or headwear." (the subject's only claim) -> nothing
    (re.compile(_START + _WHO + r"\s+" + _HAS + r"\s+" + _NO + r"\s*[.!]", re.I), ""),
    # "He has no headwear or visible jewelry; a red garment shows at the edge" -> "A red garment shows at the edge"
    (re.compile(_START + _WHO + r"\s+" + _HAS + r"\s+" + _NO + r"\s*(?:;|[\u2013\u2014])\s*", re.I), "\u0000"),
]
_LEFT = re.compile(_NO, re.I)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[,;]\s*([.!?])", r"\1", text)
    text = re.sub(r"([.!?])\s*[,;]\s*", r"\1 ", text)
    text = re.sub(r"^[\s,;.]+", "", text)
    text = re.sub(r";\s*$", ".", text.strip())
    # a sentence start left lower-case by a removed lead ("\0" marks the cut) or by a dropped sentence before it
    text = re.sub(r"\u0000\s*(\w)", lambda m: m.group(1).upper(), text)
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text.strip()


def drop(text: str) -> str:
    """The caption without its statements of absent eyewear, headwear, jewelry, accessories, print or text."""
    text = str(text or "")
    out = text
    for _ in range(4):   # to a fixed point: removing one absence can expose the next ("no eyewear, and no hat")
        before = out
        for rule, sub in _RULES:
            out = rule.sub(sub, out)
        if out == before:
            break
    return text if out == text else _tidy(out)   # a caption with nothing to drop comes back as it was, byte for byte


def left(text: str) -> list:
    """The absence phrases of the two families still in a text (empty for a clean caption)."""
    return [m.group(0) for m in _LEFT.finditer(str(text or ""))]
