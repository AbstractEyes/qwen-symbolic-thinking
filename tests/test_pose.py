"""pose: the angle geometry on a synthetic body (no weights, no GPU): known turns, twists, nods, tilts and leans read back."""
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
    assert rec["twist"] == 0.0 and rec["lean"] == {"forward": 0.0, "left": 0.0}
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
    assert "the shoulders are turned 25 degrees further to the subject's left than the hips" in pose.words(rec)


def test_head_turn_nod_and_tilt_against_the_chest():
    k = turned(BODY, HEAD, about("up", -30.0), head_center(BODY))
    assert pose.angles(k, NAMES)["head_on_chest"]["turn"] == pytest.approx(-30.0, abs=0.11)
    k = turned(BODY, HEAD, about("left", -20.0), head_center(BODY))           # the forward tipped up 20
    h = pose.angles(k, NAMES)["head_on_chest"]
    assert h["nod"] == pytest.approx(20.0, abs=0.11) and h["turn"] == pytest.approx(0.0, abs=0.11)
    k = turned(BODY, HEAD, about("forward", -15.0), head_center(BODY))        # the left side lowered 15
    rec = pose.angles(k, NAMES)
    assert rec["head_on_chest"]["tilt"] == pytest.approx(15.0, abs=0.11)
    assert "the head is tilted 15 degrees toward the subject's left shoulder" in pose.words(rec)


def test_lean_forward_and_to_the_side():
    rec = pose.angles(turned(BODY, UPPER, about("left", 20.0), np.zeros(3)), NAMES)  # the torso tipped forward 20
    assert rec["lean"]["forward"] == pytest.approx(20.0, abs=0.11)
    assert rec["lean"]["left"] == pytest.approx(0.0, abs=0.11)
    rec = pose.angles(turned(BODY, UPPER, about("forward", -12.0), np.zeros(3)), NAMES)  # the top moved to the subject's left
    assert rec["lean"]["left"] == pytest.approx(12.0, abs=0.11)
    assert "the torso leans 10 degrees to the subject's left" in pose.words(rec)


def test_words_thresholds_and_facing():
    assert pose.facing(10) == "the body faces the camera"
    assert pose.facing(-45) == "the body is turned 45 degrees to the subject's right"
    assert pose.facing(95) == "the body is in profile, turned to the subject's left"
    assert pose.facing(-130) == "the body is turned mostly away from the camera, to the subject's right"
    assert pose.words(None) == ""


def test_degenerate_points_give_none():
    flat = BODY.copy()
    flat[NAMES["left_hip"]] = flat[NAMES["right_hip"]]
    assert pose.angles(flat, NAMES) is None


def test_fill_with_a_pose_line():
    ask = FASHION_ASK + "\nHelper pose: {pose}"
    both = fill(ask, "The lady wears a tank top.", "Pose (3D body fit): the body faces the camera.")
    assert both.endswith("Helper caption: The lady wears a tank top.\nHelper pose: Pose (3D body fit): the body faces the camera.")
    neither = fill(ask, "", "")
    assert "Helper" not in neither and "{" not in neither
    assert fill(FASHION_ASK, "x", "a pose line nobody asked for").endswith("Helper caption: x")
