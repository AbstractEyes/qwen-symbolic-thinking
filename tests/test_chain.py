from qwen_symbolic_thinking import chain, prompts

ROW = {"id": "art-000001", "depth": "medium", "domain": "photograph", "genre": "genre scene",
       "spec": {"person": {"count": "1", "gender": "man"}, "thing": {"name": "oven", "count": "2", "position": "left"},
                "body": {"pose": "standing"}, "scene": {"setting": "kitchen"},
                "relation": [{"a": "man", "rel": "in front of", "b": "oven"}], "mood": {"feeling": "cozy", "from": "scene setting"}},
       "thoughts": [{"step": "see", "text": "One man, standing, in front of two ovens set to the left."},
                    {"step": "place", "text": "A kitchen: the ovens and the pans on the wall say so."},
                    {"step": "mood", "text": "A kitchen at work reads as cozy."}],
       "wordings": {"caption": "A man standing in front of two ovens in a kitchen."}}


def test_render_parse_round_trip():
    text = chain.render(ROW)
    parsed = chain.parse(text)
    assert parsed["thoughts"] == ROW["thoughts"]
    assert parsed["wordings"] == {"caption": ROW["wordings"]["caption"]}
    assert parsed["stray"] == []


def test_parse_tolerates_markdown_numbering_and_thinking():
    text = ("<think>some private thinking</think>\n"
            "Here is the chain:\n"
            "1. **See:** A woman in a red coat.\n"
            "- *place* : A wet street at night,\n"
            "   with lamps along it.\n"
            "**Mood** - Lonely.\n"
            "Caption: A woman in a red coat on a wet street at night.")
    parsed = chain.parse(text)
    assert [t["step"] for t in parsed["thoughts"]] == ["see", "place", "mood"]
    assert parsed["thoughts"][0]["text"] == "A woman in a red coat."
    assert parsed["thoughts"][1]["text"] == "A wet street at night, with lamps along it."
    assert parsed["wordings"]["caption"].startswith("A woman in a red coat")
    assert parsed["stray"] == ["Here is the chain:"]
    assert parsed["thinking"] == "some private thinking"


def test_parse_reads_the_literal_step_form_and_stray_think_tokens():
    text = ("step: what you see or work out\n\n"
            "step: see three zebras grazing in a dry field.\n\n"
            "step: count three zebras, all facing away.\n\n"
            "step: A line written without a step name.\n"
            "</think>\n"
            "caption: Three zebras graze in a dry field.")
    parsed = chain.parse(text)
    assert [t["step"] for t in parsed["thoughts"]] == ["see", "count", chain.UNNAMED]
    assert parsed["thoughts"][0]["text"] == "three zebras grazing in a dry field."
    assert parsed["thoughts"][1]["text"] == "three zebras, all facing away."
    assert parsed["stray"] == ["step: what you see or work out"]
    assert parsed["wordings"]["caption"].startswith("Three zebras")
    s = chain.score(parsed, dict(ROW, depth="medium", spec={"animal": {"name": "zebra", "count": "3"}}, thoughts=[]))
    assert s["format_ok"] and s["n_thoughts"] == 3 and s["named_frac"] == 2 / 3 and s["in_range"]
    assert s["spec_recall"] == 1.0


def test_unnamed_thoughts_do_not_make_the_format():
    parsed = chain.parse("step: A girl stands in a swamp.\nstep: The light is golden.\ncaption: A girl in a swamp.")
    s = chain.score(parsed, ROW)
    assert s["n_thoughts"] == 2 and s["named_frac"] == 0 and not s["format_ok"] and s["has_caption"]


def test_reference_scores_the_rows_own_chain():
    ref = chain.reference(ROW)
    assert ref["format_ok"] and ref["in_range"] and ref["has_caption"]
    assert ref["n_thoughts"] == 3
    assert ref["spec_recall"] == 1.0, ref["spec_missed"]
    assert ref["gloss_hits"] == 0 and ref["remark_hits"] == 0
    assert ref["step_jaccard"] == 1.0


def test_score_reads_misses_and_detectors():
    parsed = chain.parse("see: A woman holds a cup.\nlight: Low key: a dark picture with few bright areas.\n"
                         "mood: Gentle, an inference from the spec.\ncaption: A woman with a cup.")
    s = chain.score(parsed, ROW)
    assert s["n_thoughts"] == 3 and s["in_range"] and s["format_ok"]
    assert s["spec_recall"] < 0.5 and "kitchen" in s["spec_missed"]
    assert s["gloss_hits"] == 1
    assert s["remark_hits"] == 1
    assert s["step_jaccard"] == 2 / 4        # see, mood shared of see, place, mood, light


def test_counts_read_as_words():
    assert chain.present("2", "two ovens stand there")
    assert chain.present("2", "2 ovens stand there")
    assert not chain.present("2", "twelve ovens stand there")
    assert chain.present("30s", "a woman in her thirties")
    assert chain.present("centre", "zebras in the center of the frame") and chain.present("grey", "a gray coat")
    assert not chain.present("centre", "zebras at the edge")


def test_prompt_forms_carry_the_form_and_the_depth():
    for depth, lo, hi in (("simple", 1, 2), ("medium", 3, 5), ("deep", 6, 10)):
        text = prompts.bare(depth)
        assert f"{lo} to {hi} thoughts" in text and "never the whole list" in text
        assert "see, name, place, count" in text and "caption:" in text and "`see: two riders on grey horses`" in text
        assert "step: what you see" not in text
    shown = prompts.shown("medium", [dict(ROW, hint="A man making pizzas in a kitchen.")])
    assert "Picture: A man making pizzas in a kitchen." in shown and "see: One man" in shown
    spec = prompts.spec("deep", ROW["spec"])
    assert "person count: 1" in spec and "relation: man in front of oven" in spec and "mood from" not in spec
    assert "6 to 10 thoughts" in spec
    short = prompts.short("simple")
    assert "a quick look" in short and short.endswith("ending with a caption.") and "steps" not in short
    spec_short = prompts.spec_short("medium", ROW["spec"])
    assert "scene setting: kitchen" in spec_short and "a closer look" in spec_short
