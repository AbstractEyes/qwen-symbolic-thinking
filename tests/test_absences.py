"""absences.drop: the statements of absent eyewear, headwear, jewelry, accessories, print or text leave the caption; every
other word stays. The cases are the forms the stock Qwen3.5-4B wrote on FFHQ and DeepFashion."""
import pytest

from qwen_symbolic_thinking import absences as A

CASES = [
    # whole sentences
    ("He smiles. No eyewear, headwear, or jewelry is visible on either person. The wall is white.",
     "He smiles. The wall is white."),
    ("The temple arm bears the text “PORSCHE DESIGN.” No headwear or jewelry is visible. The light is soft.",
     "The temple arm bears the text “PORSCHE DESIGN.” The light is soft."),
    ("She stands still. No print or text appears anywhere in the image.", "She stands still."),
    ("He looks up. He wears no eyewear, headwear, or visible jewelry. His shirt is red.", "He looks up. His shirt is red."),
    # a clause of its own
    ("A child with a rounded neckline; no print or text is visible on the garment.", "A child with a rounded neckline."),
    ("He smiles at the camera; he wears no eyewear, headwear, or jewelry. He is dressed in black.",
     "He smiles at the camera. He is dressed in black."),
    ("She wears a white top with a subtle smile; her garment has no visible print or text.", "She wears a white top with a subtle smile."),
    ("Her hair is pulled back, and no visible eyewear, headwear, or jewelry is present.", "Her hair is pulled back."),
    # inside a clause
    ("A young infant with short dark hair and no eyewear, headwear, or jewelry is shown in a close-up.",
     "A young infant with short dark hair is shown in a close-up."),
    ("She is dressed in a plain gray top with no visible print or text.", "She is dressed in a plain gray top."),
    ("She wears large turquoise hoop earrings and has no eyewear or headwear.", "She wears large turquoise hoop earrings."),
    ("She is in her 30s or 40s, with no eyewear, headwear, or visible jewelry. She smiles.", "She is in her 30s or 40s. She smiles."),
    ("He has short dark hair with bangs, no eyewear or headwear, and a wide, toothy smile.",
     "He has short dark hair with bangs and a wide, toothy smile."),
    ("A man with short brown hair and a mustache, no eyewear or headwear, gazes at the camera.",
     "A man with short brown hair and a mustache gazes at the camera."),
    ("An older man with gray, wavy hair, no eyewear, and no headwear or jewelry. He frowns.", "An older man with gray, wavy hair. He frowns."),
    ("Blue jeans that have no visible print or text. She holds a bag.", "Blue jeans. She holds a bag."),
    ("Shorts in the same chiffon, both featuring no visible print or text.", "Shorts in the same chiffon."),
    ("She wears a shirt with no visible text or additional garments.", "She wears a shirt."),
    ("A pink sweater with ribbed cuffs — no visible print or text. Gray jeans below.", "A pink sweater with ribbed cuffs. Gray jeans below."),
    # the subject's absence joined to a clause that goes on
    ("She is in her 30s; she wears no eyewear or headwear, but has on a beaded necklace.", "She is in her 30s; she has on a beaded necklace."),
    ("The child wears no eyewear or jewelry, and only the top of a dark garment is visible.", "Only the top of a dark garment is visible."),
    ("No eyewear or headwear is visible, and her garment is a black strap top.", "Her garment is a black strap top."),
    ("No jewelry or headwear is visible; her expression is warm.", "Her expression is warm."),
    ("He has no headwear or visible jewelry; a red garment shows at the edge.", "A red garment shows at the edge."),
    ("A woman is in the foreground, wearing no eyewear or headwear, and a thin necklace.", "A woman is in the foreground, wearing a thin necklace."),
]
KEPT = [
    "A boy with no visible smile looks on.",                                   # an expression, not an absent item
    "The child has no visible hair peeking out from the beanie.",
    "No visible garments are shown; only the top of her shoulders is visible.",   # garments alone: a description of the crop
    "She wears a shirt with no visible text or patterns on the sleeves.",      # the list goes on past what the rules know: left whole
    "A sign reads “NO TEXTING” behind him.",
    "e.g., a plain wall; nothing else.",                                       # untouched text comes back byte for byte
    "Form-fitting black pants with no visible print or texture.",              # texture is not text (0.1.50 wrote "pantsure")
    "The design has no text presented on it.",                                 # words end at a boundary: presented, not present
]


@pytest.mark.parametrize("raw, clean", CASES)
def test_drop(raw, clean):
    assert A.drop(raw) == clean
    assert A.left(A.drop(raw)) == []
    assert A.drop(A.drop(raw)) == A.drop(raw)          # a fixed point


@pytest.mark.parametrize("text", KEPT)
def test_kept(text):
    assert A.drop(text) == text


@pytest.mark.parametrize("raw, clean", CASES)
def test_only_loses_words(raw, clean):
    import re
    words = lambda s: {w.lower() for w in re.findall(r"[\w'’-]+", s)}
    assert words(A.drop(raw)) <= words(raw)            # a cleaned caption never gains a word


def test_plural_items():
    assert A.drop("A plain tee with no visible prints or text. She smiles.") == "A plain tee. She smiles."


def test_left_names_what_is_there():
    assert A.left("A man with no eyewear, headwear, or jewelry.") == ["no eyewear, headwear, or jewelry"]
    assert A.left("A man in a gray coat.") == [] and A.drop("") == "" and A.drop(None) == ""
