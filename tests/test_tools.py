from qwen_symbolic_thinking.tools import medium, probe_read, screen_conflicts

ROW = ("see: One cup on a table, white against dark wood.\n"
       "light: Light falls from the left and leaves a soft shadow.\n"
       "mood: A quiet, everyday stillness.\n"
       "caption: A white cup on a dark table. A cup is a vessel.\n"
       "prompt: still life photograph, white cup, dark wooden table\n"
       "question: thing name -> cup [record]\n"
       "question: who set it there -> unknown [unknown]\n")


def test_expected_kind_by_catalogue_line():
    assert medium.expected("Medium: oil on canvas.") == "painting"
    assert medium.expected("gelatin silver print") == "photograph"
    assert medium.expected("Medium: etching on cream laid paper") == "print"
    assert medium.expected("graphite on paper") == "drawing"
    assert medium.expected("Covered box; porcelain painted in underglaze blue") == "painting"   # the pattern's known over-reach
    assert medium.expected("bronze") == "object"
    assert medium.expected("a view of a harbour") is None


def test_classification_to_kind():
    assert medium.kind_of_class("Drawings") == "drawing"
    assert medium.kind_of_class("decorative art") == "object"
    assert medium.kind_of_class("Books") is None


def test_first_term_and_names():
    assert medium.first_term(ROW) == "still life photograph"
    assert medium.names("still life photograph", "photograph")
    assert medium.names("photograph", "object")
    assert medium.names("oil painting", "painting")
    assert not medium.names("drawing", "painting")
    assert not medium.names("", "painting")


def test_conflicts_of_a_row():
    c = medium.conflicts(ROW)
    assert c == {"unknown": True, "light": True, "mood": True, "stock": True, "defining": False}
    assert medium.conflicts("see: A cup is a vessel for drink.\nprompt: photograph, cup\n")["defining"]


def test_probe_read_counts():
    rows = [{"sample_id": "a", "text": ROW}, {"sample_id": "b", "text": "see: x\nprompt: drawing, a man\n"}, {"sample_id": "zzz", "text": ROW}]
    s = probe_read.read(rows, {"a": "object", "b": "painting"})
    assert s["rows"] == 2
    assert s["by_kind"]["object"] == [1, 1] and s["by_kind"]["painting"] == [0, 1]
    assert s["conflicts"]["unknown"] == 1


def test_screen_read_terms_and_words():
    s = screen_conflicts.read([{"text": ROW}, {"text": "see: x\nprompt: oil painting, a man\n"}])
    assert s["rows"] == 2 and s["terms"] == {"photograph": 1, "painting": 1}
    assert s["conflicts"]["stock"] == 1 and s["words"] > 0
