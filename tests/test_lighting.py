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


def _heads(direction=(), contrast="low", edge=False, silhouette=False):
    return {"direction": {"pick": list(direction)}, "contrast": {"pick": [contrast]}, "edge": {"pick": ["likelihood"] if edge else []},
            "silhouette": {"pick": ["likelihood"] if silhouette else []}, "cast": {"pick": ["soft"]}}


def test_shot_light_record_values():
    assert L.shot_light(_heads(["front"]), 0.5) == []                                   # frontal, mid-key: nothing to record
    assert L.shot_light(_heads(["front", "left"]), 0.5) == ["from the left"]
    assert L.shot_light(_heads(["top", "right"]), 0.5) == ["from above", "from the right"]   # in the head's label order
    assert L.shot_light(_heads(["back"], edge=True), 0.5) == ["backlit"]                # backlit already names the edge light
    assert L.shot_light(_heads([], edge=True), 0.5) == ["rim light"]
    assert L.shot_light(_heads([], "high"), 0.2) == ["low key"]
    assert L.shot_light(_heads([], "low"), 0.8) == ["high key"]
    assert L.shot_light(_heads([], "high"), 0.8) == [] and L.shot_light(_heads([], "low"), None) == []
    assert L.shot_light(_heads(["back"], silhouette=True), 0.1) == ["silhouette", "backlit"]


def test_heads_run_plumbing(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    zpath = tmp_path / "faces.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, shade in (("00000.webp", 240), ("00001.webp", 20), ("00002.webp", 128)):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (shade, shade, shade)).save(buf, "WEBP", lossless=True)
            zf.writestr(name, buf.getvalue())
    labels = [f"{h}.{lab}" for h, meta in SCHEMA_HEADS.items() for lab in meta["labels"]]

    class Inp:
        name, shape = "Image", ["batch", 3, 256, 256]

    class Session:
        def get_inputs(self):
            return [Inp()]

        def run(self, _, feed):
            x = feed["Image"]
            out = np.full((x.shape[0], len(labels)), 0.1, dtype=np.float32)
            bright = x.mean(axis=(1, 2, 3)) > 0.5
            for i, b in enumerate(bright):
                out[i, labels.index("shot.lighting.contrast.low" if b else "shot.lighting.contrast.high")] = 0.9
                out[i, labels.index("shot.lighting.direction.left")] = 0.8
            return [np.zeros((x.shape[0], 512), dtype=np.float32), out]
    monkeypatch.setattr(L, "cinemaclip_load", lambda: {"session": Session(), "labels": labels, "heads": SCHEMA_HEADS,
                                                       "schema_version": "test"})
    monkeypatch.chdir(tmp_path)
    out = L.heads_run(str(zpath), batch=2, ship=False, tag="t")
    rows = [json.loads(line) for line in open(f"{out['out_dir']}/rows.jsonl", encoding="utf-8")]
    assert [r["file"] for r in rows] == ["00000.webp", "00001.webp", "00002.webp"]
    assert rows[0]["shot_light"] == ["from the left", "high key"] and rows[1]["shot_light"] == ["from the left", "low key"]
    assert rows[2]["shot_light"] == ["from the left"]                       # mid gray: neither key
    assert out["photos"] == 3 and out["shot_light_counts"]["from the left"] == 3


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
