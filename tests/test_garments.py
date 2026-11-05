from qwen_symbolic_thinking import garments as g

CAPTION = ("A young man stands against a plain white background, wearing a dark heathered gray crew-neck t-shirt, slim-fit black "
           "pants, and black high-top sneakers with white laces and soles.")
TAGS = [["1boy", 0.9, "general", 0.9], ["grey shirt", 0.8, "general", 0.7], ["black pants", 0.8, "general", 0.8],
        ["black footwear", 0.7, "general", 0.8], ["sneakers", 0.6, "general", 0.7]]


def test_caption_colors_follow_the_garment_they_precede():
    assert g.caption_colors(CAPTION) == {"upper": ["gray"], "lower": ["black"], "shoes": ["black"]}
    assert g.caption_colors("a white and blue striped shirt and red shorts") == {"upper": ["white", "blue"], "lower": ["red"]}
    assert g.caption_colors("a black-and-white top") == {"upper": ["black", "white"]}
    assert g.caption_colors("a white t-shirt with a red print") == {"upper": ["white"]}      # the print's color stays off
    assert g.caption_colors("light-wash denim shorts") == {}                                 # denim is a fabric, not a color


def test_tag_colors_read_color_plus_garment_tags():
    assert g.tag_colors(TAGS) == {"upper": ["gray"], "lower": ["black"], "shoes": ["black"]}
    assert g.tag_colors("light_blue_skirt, tank_top, red dress") == {"lower": ["blue"], "onepiece": ["red"]}


def test_agreed_keeps_only_colors_both_give():
    assert g.agreed(CAPTION, TAGS) == {"upper": "gray", "lower": "black", "shoes": "black"}
    assert g.agreed("a white shirt and blue jeans", "black shirt, blue pants") == {"lower": "blue"}   # the shirt disagrees
    assert g.agreed("a woman in a top", "white shirt") == {}


def test_worn_needs_both_describers():
    assert g.worn(CAPTION, TAGS, "shoes") == {"name": "sneakers"}
    assert g.worn("black boots", "black footwear", "shoes") == {"name": "shoes"}
    assert g.worn("a denim jacket over a tee", "jacket, open jacket", "outer") == {"name": "jacket"}
    assert g.worn(CAPTION, ["shirt"], "shoes") == {}


def test_angle_bands():
    assert [g.angle(t) for t in (0, -20, 21, 70, -71, 110, 111, 180, None)] == [
        "frontal", "frontal", "three quarter", "three quarter", "profile", "profile", None, None, None]


def test_records_line():
    spec = {"person": {"gender": "man", "count": "1"}, "upper": {"name": "t-shirt", "pattern": "colour block"},
            "lower": {"name": "trousers", "pattern": "plain"}, "shot": {"framing": "full body", "angle": "frontal"}}
    line = g.records_line(spec, g.agreed(CAPTION, TAGS), "three quarter", {"shoes": g.worn(CAPTION, TAGS, "shoes")})
    assert line == ("person gender: man; person count: 1; upper name: t-shirt; upper pattern: color block; upper color: gray; "
                    "lower name: trousers; lower pattern: plain; lower color: black; shot framing: full body; "
                    "shot angle: three quarter; shoes name: sneakers; shoes color: black")
    assert "shot angle" not in g.records_line(spec, camera_angle="")          # the back to the camera: no angle
    assert "shot angle: frontal" in g.records_line(spec)                       # no fit: the specification's own
    dress = {"wear": {"clothing": "dress", "pattern": "floral"}}
    assert g.records_line(dress, {"onepiece": "red", "upper": "white"}) == "wear clothing: dress; wear pattern: floral; wear color: red"
    assert spec["upper"] == {"name": "t-shirt", "pattern": "colour block"}    # the caller's specification is untouched
