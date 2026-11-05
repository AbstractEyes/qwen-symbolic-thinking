"""captions.run's plumbing on a three-photo zip, with the VLM and the body fit faked (no GPU, no weights, nothing shipped): the
pose-only pass, and captions whose ask carries the pose line."""
import io
import json
import os
import sys
import zipfile
from types import SimpleNamespace

from PIL import Image

from qwen_symbolic_thinking import captions, pose

REC = {"hips": {"turn": 30.0, "nod": 0.0, "tilt": 0.0}, "chest": {"turn": 30.0, "nod": 0.0, "tilt": 0.0}, "twist": 0.0,
       "head": {"turn": 30.0, "nod": 0.0, "tilt": 0.0}, "head_on_chest": {"turn": 0.0, "nod": 0.0, "tilt": 0.0},
       "lean": {"forward": 0.0, "left": 0.0}, "frames": {}}
NAMES = ["A-id_1_front.jpg", "B-id_6_flat.jpg", "C-id_2_side.jpg"]


def photo_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        for name in NAMES:
            buf = io.BytesIO()
            Image.new("RGB", (32, 48), (200, 100, 50)).save(buf, "JPEG")
            zf.writestr(name, buf.getvalue())
    return str(path)


def fake_fit(monkeypatch):
    try:
        import torch  # noqa: F401
    except ImportError:                       # a CPU test machine without torch: run() only clears the CUDA cache with it
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setattr(captions, "preflight", lambda: {"fake": True})
    monkeypatch.setattr(pose, "load", lambda weights: {"weights": weights, "commit": "abc1234"})
    monkeypatch.setattr(pose, "predict", lambda fit, picture: dict(REC))
    monkeypatch.setattr(pose, "unload", lambda fit: None)


def rows_of(out_dir, summary):
    path = os.path.join(out_dir, summary["path_in_repo"].split("/")[-1], "rows.jsonl")
    with open(path, encoding="utf-8") as fh:
        return {r["file"]: r for r in map(json.loads, fh)}


def test_pose_only_pass(tmp_path, monkeypatch):
    fake_fit(monkeypatch)
    out = str(tmp_path / "out")
    summary = captions.run(photo_zip(tmp_path / "p.zip"), None, model_id=None, tagger=None, out_dir=out, do_ship=False,
                           batch=2, chunk=2, pose_fit="fake/weights", pose_skip=lambda f: "_6_flat" in f)
    rows = rows_of(out, summary)
    assert set(rows) == set(NAMES)
    assert rows["A-id_1_front.jpg"]["pose_status"] == "fitted" and rows["A-id_1_front.jpg"]["pose"]["hips"]["turn"] == 30.0
    assert rows["A-id_1_front.jpg"]["pose_line"] == "Pose (3D body fit): the body is turned 30 degrees to the subject's left."
    assert rows["B-id_6_flat.jpg"]["pose_status"] == "skipped" and rows["B-id_6_flat.jpg"]["pose"] is None
    assert "caption" not in rows["C-id_2_side.jpg"]
    assert summary["pose"]["counts"] == {"fitted": 2, "skipped": 1} and summary["pose"]["commit"] == "abc1234"
    assert set(summary["s_per_photo_by_op"]) == {"pose", "caption", "tags"}


def test_captions_with_the_pose_line_in_the_ask(tmp_path, monkeypatch):
    fake_fit(monkeypatch)
    monkeypatch.setattr(captions, "load_model", lambda model_id, think: ("processor", "model"))
    monkeypatch.setattr(captions.family, "marker", lambda model: "mark")
    monkeypatch.setattr(captions.family, "content", lambda text, picture, mark: text)
    monkeypatch.setattr(captions, "fast_path_note", lambda: "fake")
    monkeypatch.setattr(captions, "generate",
                        lambda processor, model, convs, images, max_new, think, sys: ([c[0]["content"] for c in convs], [7] * len(convs)))
    descriptions = tmp_path / "captions.json"
    descriptions.write_text(json.dumps({"A-id_1_front.jpg": "The lady wears a tank top."}), encoding="utf-8")
    out = str(tmp_path / "out")
    summary = captions.run(photo_zip(tmp_path / "p.zip"), str(descriptions), model_id="fake/vlm", tagger=None, out_dir=out,
                           do_ship=False, batch=3, chunk=3, pose_fit="fake/weights", pose_ask=True,
                           pose_skip=lambda f: "_6_flat" in f)
    rows = rows_of(out, summary)
    a, b = rows["A-id_1_front.jpg"], rows["B-id_6_flat.jpg"]
    assert "Helper caption: The lady wears a tank top." in a["caption"]
    assert "Helper pose: Pose (3D body fit): the body is turned 30 degrees to the subject's left." in a["caption"]
    assert a["pose_asked"] and a["helped"] and a["new_tokens"] == 7
    assert "Helper" not in b["caption"] and not b["pose_asked"]          # no description, the fit skipped: no helper lines
    assert summary["ask"].endswith(captions.POSE_LINE)
