"""pose: the angle geometry on a synthetic body (no weights, no GPU): known turns, twists, nods, tilts and leans read back."""
import json
import math

import numpy as np
import pytest

from qwen_symbolic_thinking import pose
from qwen_symbolic_thinking.captions import FASHION_ASK, fill

NAMES = {n: i for i, n in enumerate(("left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_ear", "right_ear",
                                     "left_eye", "right_eye"))}
# A person facing the camera upright, in the camera's frame (x right, y down, z away): their left is +x, up -y, forward -z.
BODY = np.array([[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0],              # hips
                 [0.20, -0.50, 0.0], [-0.20, -0.50, 0.0],          # shoulders
                 [0.075, -0.70, 0.0], [-0.075, -0.70, 0.0],        # ears
                 [0.03, -0.70, -0.09], [-0.03, -0.70, -0.09]])     # eyes: in front of the ears, at their height
UPPER = [NAMES[n] for n in ("left_shoulder", "right_shoulder", "left_ear", "right_ear", "left_eye", "right_eye")]
HEAD = [NAMES[n] for n in ("left_ear", "right_ear", "left_eye", "right_eye")]


def about(axis: str, deg: float) -> np.ndarray:
    """A rotation about the person's own axis (in left / up / forward terms), written in the camera's frame."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    r = {"up": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),          # + turns the forward toward the left
         "left": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),        # + tips the forward down
         "forward": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]  # + raises the left side
    return pose.FACING @ r @ pose.FACING.T


def turned(points: np.ndarray, rows: list[int], rot: np.ndarray, center: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[rows] = (points[rows] - center) @ rot.T + center
    return out


def head_center(k: np.ndarray) -> np.ndarray:
    return (k[NAMES["left_ear"]] + k[NAMES["right_ear"]]) / 2


def test_facing_the_camera_reads_zero():
    rec = pose.angles(BODY, NAMES)
    for part in ("hips", "chest", "head", "head_on_chest"):
        assert rec[part] == {"turn": 0.0, "nod": 0.0, "tilt": 0.0}
    assert rec["twist"] == 0.0 and rec["lean"] == {"forward": 0.0, "left": 0.0, "picture": 0.0}
    assert pose.words(rec) == "Pose (3D body fit): the body faces the camera."


@pytest.mark.parametrize("deg", [40.0, -65.0, 90.0, 150.0])
def test_a_whole_body_turn(deg):
    rec = pose.angles(BODY @ about("up", deg).T, NAMES)
    for part in ("hips", "chest", "head"):
        assert rec[part]["turn"] == pytest.approx(deg, abs=0.11)
    assert rec["twist"] == pytest.approx(0.0, abs=0.11)
    assert rec["head_on_chest"]["turn"] == pytest.approx(0.0, abs=0.11)


def test_the_back_to_the_camera():
    rec = pose.angles(BODY @ about("up", 180.0).T, NAMES)
    assert abs(rec["hips"]["turn"]) == pytest.approx(180.0, abs=0.11)
    assert "back to the camera" in pose.words(rec)


def test_shoulders_twisted_against_the_hips():
    rec = pose.angles(turned(BODY, UPPER, about("up", 25.0), np.zeros(3)), NAMES)
    assert rec["hips"]["turn"] == pytest.approx(0.0, abs=0.11)
    assert rec["chest"]["turn"] == pytest.approx(25.0, abs=0.11)
    assert rec["twist"] == pytest.approx(25.0, abs=0.11)
    assert rec["head_on_chest"]["turn"] == pytest.approx(0.0, abs=0.11)
    assert "the shoulders are turned 25 degrees toward the right of the picture" in pose.words(rec)


def test_head_turn_nod_and_tilt_against_the_chest():
    k = turned(BODY, HEAD, about("up", -30.0), head_center(BODY))
    assert pose.angles(k, NAMES)["head_on_chest"]["turn"] == pytest.approx(-30.0, abs=0.11)
    k = turned(BODY, HEAD, about("left", -20.0), head_center(BODY))           # the forward tipped up 20
    h = pose.angles(k, NAMES)["head_on_chest"]
    assert h["nod"] == pytest.approx(20.0, abs=0.11) and h["turn"] == pytest.approx(0.0, abs=0.11)
    k = turned(BODY, HEAD, about("forward", -15.0), head_center(BODY))        # the left side lowered 15
    rec = pose.angles(k, NAMES)
    assert rec["head_on_chest"]["tilt"] == pytest.approx(15.0, abs=0.11)
    assert "the head tilts 15 degrees toward the right of the picture" in pose.words(rec)   # its left shoulder, facing the camera


def test_lean_forward_and_to_the_side():
    rec = pose.angles(turned(BODY, UPPER, about("left", 20.0), np.zeros(3)), NAMES)  # the torso tipped forward 20
    assert rec["lean"]["forward"] == pytest.approx(20.0, abs=0.11)
    assert rec["lean"]["left"] == pytest.approx(0.0, abs=0.11)
    rec = pose.angles(turned(BODY, UPPER, about("forward", -12.0), np.zeros(3)), NAMES)  # the top moved to the subject's left
    assert rec["lean"]["left"] == pytest.approx(12.0, abs=0.11)
    assert "the torso leans 10 degrees toward the right of the picture" in pose.words(rec)


def test_words_thresholds_and_facing():
    assert pose.facing(10) == "the body faces the camera"
    assert pose.facing(-45) == "the body is turned 45 degrees toward the left of the picture"
    assert pose.facing(95) == "the body is in profile, facing the right of the picture"
    assert pose.facing(-130) == "the body is turned mostly away from the camera, facing the left of the picture"
    assert pose.facing(175, "the head") == "the head faces away from the camera"
    assert pose.facing(30, "the shoulders") == "the shoulders are turned 30 degrees toward the right of the picture"
    assert pose.words(None) == ""


def test_back_to_the_camera_the_sides_swap():
    """With the back to the camera the subject's left is the picture's left: a further turn to the subject's left (+) moves the
    shoulders toward the left of the picture."""
    k = turned(BODY @ about("up", 180.0).T, UPPER, about("up", 30.0), np.zeros(3))   # here "up" stays the camera's vertical
    rec = pose.angles(k, NAMES)
    assert rec["twist"] == pytest.approx(30.0, abs=0.11)
    assert "the shoulders are turned mostly away from the camera, facing the left of the picture" in pose.words(rec)


def test_degenerate_points_give_none():
    flat = BODY.copy()
    flat[NAMES["left_hip"]] = flat[NAMES["right_hip"]]
    assert pose.angles(flat, NAMES) is None


def test_append_pose_joins_by_file(tmp_path):
    from qwen_symbolic_thinking.captions import append_pose
    k = np.zeros((70, 3))
    for name, i in pose.KEYPOINTS.items():
        k[i] = BODY[NAMES[name]]
    k = k @ about("up", 90.0).T                                     # in profile, facing the right of the picture
    caps = tmp_path / "c.jsonl"
    caps.write_text("\n".join(json.dumps(r) for r in (
        {"file": "a_2_side.jpg", "caption": "A man in jeans."}, {"file": "b_6_flat.jpg", "caption": "A shirt laid flat."})) + "\n",
                    encoding="utf-8")
    poses = tmp_path / "p.jsonl"
    poses.write_text("\n".join(json.dumps(r) for r in (
        {"file": "a_2_side.jpg", "pose": {"keypoints_3d": k.tolist(), "focal_length": 1000.0, "hips": "old"}},
        {"file": "b_6_flat.jpg", "pose": None}, {"file": "c_1_front.jpg", "pose": None})) + "\n", encoding="utf-8")
    out = tmp_path / "o.jsonl"
    counts = append_pose(str(caps), str(poses), str(out))
    rows = {r["file"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
    a, b = rows["a_2_side.jpg"], rows["b_6_flat.jpg"]
    assert a["pose_line"] == "The body is in profile, facing the right of the picture."
    assert a["caption_pose"] == "A man in jeans. The body is in profile, facing the right of the picture."
    assert a["pose"]["hips"]["turn"] == pytest.approx(90.0, abs=0.11) and "keypoints_3d" not in a["pose"]
    assert b["caption_pose"] == "A shirt laid flat." and b["pose"] is None
    assert counts == {"with a pose": 1, "without a pose": 1, "pose rows without a caption": 1}


def test_fill_with_a_pose_line():
    ask = FASHION_ASK + "\nHelper pose: {pose}"
    both = fill(ask, "The lady wears a tank top.", "Pose (3D body fit): the body faces the camera.")
    assert both.endswith("Helper caption: The lady wears a tank top.\nHelper pose: Pose (3D body fit): the body faces the camera.")
    neither = fill(ask, "", "")
    assert "Helper" not in neither and "{" not in neither
    assert fill(FASHION_ASK, "x", "a pose line nobody asked for").endswith("Helper caption: x")


class _Estimator:
    """A stand-in for SAM 3D Body's estimator: each box gets the facing body, its 2D points spread over the box (a box with
    x0 < 0 gets no body); the call sizes are kept."""

    def __init__(self):
        self.calls = []

    def process_one_image(self, img, bboxes=None, inference_type="body"):
        self.calls.append(len(bboxes))
        k3d = np.zeros((70, 3))
        for name, i in pose.KEYPOINTS.items():
            k3d[i] = BODY[NAMES[name]]
        grid = np.linspace(0.0, 1.0, 70)
        out = []
        for x0, y0, x1, y1 in bboxes:
            k2d = np.stack([x0 + (x1 - x0) * grid, y0 + (y1 - y0) * grid[::-1]], axis=1)
            out.append({"pred_keypoints_3d": np.zeros((70, 3)) if x0 < 0 else k3d, "pred_keypoints_2d": k2d,
                        "focal_length": np.array([1234.56])})
        return out


def test_predict_many_keeps_the_boxes_order_and_their_pixels():
    from PIL import Image
    est = _Estimator()
    fit = {"estimator": est, "keypoints": pose.KEYPOINTS}
    boxes = [[10, 20, 110, 220], [-5, 0, 50, 50], [200, 20, 300, 220]] + [[i, 0, i + 40, 80] for i in range(0, 400, 20)]
    recs = pose.predict_many(fit, Image.new("RGB", (640, 480)), boxes, chunk=16)
    assert est.calls == [16, 7] and len(recs) == len(boxes)
    assert recs[1] is None                                          # no body fitted: the place is kept
    a, c = recs[0], recs[2]
    assert a["box"] == [10.0, 20.0, 110.0, 220.0] and c["box"] == [200.0, 20.0, 300.0, 220.0]
    assert a["keypoints_2d"][0] == [10.0, 220.0] and a["keypoints_2d"][-1] == [110.0, 20.0]
    assert a["focal_length"] == 1234.6 and a["hips"] == {"turn": 0.0, "nod": 0.0, "tilt": 0.0}
    assert pose.predict_many(fit, Image.new("RGB", (8, 8)), []) == []


def test_apart_reads_one_person_twice_and_two_people():
    k = np.zeros((70, 2))
    k[list(pose.BODY_2D)] = np.stack([np.linspace(100, 160, 17), np.linspace(50, 350, 17)], axis=1)
    one = {"keypoints_2d": k.tolist()}
    nudged = {"keypoints_2d": (k + np.array([3.0, -4.0])).tolist()}       # 5 px over a ~306 px body
    beside = {"keypoints_2d": (k + np.array([150.0, 0.0])).tolist()}      # a second person a body width away
    assert pose.apart(one, one) == 0.0
    assert pose.apart(one, nudged) == pytest.approx(5.0 / math.hypot(60, 300), abs=1e-4)
    assert pose.apart(one, beside) == pytest.approx(150.0 / math.hypot(60, 300), abs=1e-4)
    assert pose.apart(one, None) is None and pose.apart(one, {"keypoints_2d": None}) is None
