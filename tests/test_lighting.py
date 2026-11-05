import io
import json
import zipfile

from qwen_symbolic_thinking import lighting as L


def test_choice_text_matches_the_card_form():
    text = L.choice_text("What's the lighting of this shot?", ("Soft light", "Hard light"))
    assert text == ("What's the lighting of this shot?\nOptions:\nA. Soft light\nB. Hard light\n"
                    "Please select the most likely answer from the options above.\n")


def test_picked_reads_letters_and_names():
    opts = L.LIGHTS
    assert L.picked("A", opts) == ["Soft light"]
    assert L.picked("B. Hard light", opts) == ["Hard light"]
    assert L.picked("A, C", opts) == ["Soft light", "Side light"]
    assert L.picked("Soft light, Low contrast", opts) == ["Soft light", "Low contrast"]
    assert L.picked("A soft lighting setup", opts) == []           # neither a letter nor an option's whole name
    assert L.picked("HMI", L.TYPES) == ["HMI"]
    assert L.picked("Z.", opts) == []


SCHEMA_HEADS = {"shot.lighting.cast": {"activation": "softmax", "labels": ["soft", "neutral", "hard", "na"]},
                "shot.lighting.contrast": {"activation": "softmax", "labels": ["low", "neutral", "high", "na"]},
                "shot.lighting.direction": {"activation": "sigmoid", "suggested_thresh": 0.67,
                                            "labels": ["ambient", "back", "front", "left", "right", "top", "under"]},
                "shot.lighting.edge": {"activation": "sigmoid", "suggested_thresh": 0.8, "labels": ["likelihood"]},
                "shot.lighting.silhouette": {"activation": "sigmoid", "suggested_thresh": 0.82, "labels": ["likelihood"]}}


def _flat():
    flat = {}
    for head, meta in SCHEMA_HEADS.items():
        for label in meta["labels"]:
            flat[f"{head}.{label}"] = 0.1
    flat.update({"shot.lighting.cast.soft": 0.7, "shot.lighting.contrast.low": 0.6, "shot.lighting.direction.front": 0.9,
                 "shot.lighting.direction.left": 0.67, "shot.lighting.edge.likelihood": 0.79})
    return flat


def test_lighting_heads_pick_by_activation():
    heads = L.lighting_heads(_flat(), SCHEMA_HEADS)
    assert heads["cast"]["pick"] == ["soft"] and heads["contrast"]["pick"] == ["low"]
    assert heads["direction"]["pick"] == ["front", "left"]            # sigmoid: every label at or over 0.67
    assert heads["edge"]["pick"] == [] and heads["silhouette"]["pick"] == []
    assert heads["direction"]["scores"]["back"] == 0.1


def test_pick_schema_matches_the_encoder_width():
    v2 = {"version": "2.0.0", "probabilities_labels": ["x"] * 315}
    v1 = {"version": "1.0.0", "probabilities_labels": ["x"] * 101}
    assert L.pick_schema([512, 101], [v2]) is None
    assert L.pick_schema([512, 101], [v2, v1]) is v1
    assert L.pick_schema([315], [v2, v1]) is v2


def test_trial_plumbing(tmp_path, monkeypatch):
    from PIL import Image
    zpath = tmp_path / "images.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name in ("a_1_front.jpg", "b_3_back.jpg"):
            buf = io.BytesIO()
            Image.new("RGB", (30, 40), (200, 200, 200)).save(buf, "JPEG")
            zf.writestr("images/" + name, buf.getvalue())
    asked = []

    def fake_ask(bundle, image, ask, max_new_tokens=64):
        asked.append(ask)
        return "B" if "Options:" in ask else "Soft light from the front."
    monkeypatch.setattr(L, "shotvl_load", lambda: {"processor_from": "fake"})
    monkeypatch.setattr(L, "shotvl_ask", fake_ask)
    monkeypatch.setattr(L, "cinemaclip_load", lambda: {})
    monkeypatch.setattr(L, "cinemaclip_scores", lambda b, im: L.lighting_heads(_flat(), SCHEMA_HEADS))
    monkeypatch.chdir(tmp_path)
    out = L.trial(str(zpath), ["a_1_front.jpg", "b_3_back.jpg"], views={"a_1_front.jpg": "front"}, ship=False)
    rows = [json.loads(line) for line in open(f"{out['out_dir']}/rows.jsonl", encoding="utf-8")]
    assert [r["file"] for r in rows] == ["a_1_front.jpg", "b_3_back.jpg"] and rows[0]["view"] == "front"
    assert rows[0]["shotvl"]["type_choice"] == {"text": "B", "picked": ["Sunny"]}
    assert rows[0]["shotvl"]["light_choice"]["picked"] == ["Hard light"]
    assert rows[0]["shotvl"]["light_open"] == {"text": "Soft light from the front."}
    assert rows[1]["cinemaclip"]["direction"]["pick"] == ["front", "left"]
    assert len(asked) == 8 and out["photos"] == 2 and out["processor_from"] == "fake"
