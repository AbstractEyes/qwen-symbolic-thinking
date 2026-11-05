"""Body angles from a 3D body fit: SAM 3D Body (Meta; github facebookresearch/sam-3d-body, the SAM License; its body rig MHR is
Apache-2.0) fits a body to one photo, and the fit's named 3D keypoints give three frames, the hips, the chest (the shoulder line)
and the head, in the camera's frame. From them: each one's turn / nod / tilt against the camera, the twist of the shoulders against
the hips, the head against the chest, the torso's lean, and a plain pose line for a caption ask.

    from qwen_symbolic_thinking import pose
    fit = pose.load()                       # the weights are gated: request access to facebook/sam-3d-body-vith on the Hub
    rec = pose.predict(fit, picture)        # a PIL image -> the record (the whole picture as the person's box), or None
    pose.words(rec)                         # "Pose (3D body fit): the body is turned 30 degrees to the subject's left; ..."

A frame's columns are the subject's left, up and forward, written in the camera's frame (x right, y down, z away from the camera).
The frames come from keypoints (hips, shoulders, ears, eyes), not from the rig's joint rotations, so they do not depend on the rig's
internal axes. Angles are intrinsic Y-X-Z: the turn about the up axis (+ toward the subject's left), the nod about the left axis
(+ up), the tilt about the forward axis (+ toward the subject's left shoulder); a person facing the camera upright reads 0, 0, 0."""
from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = "https://github.com/facebookresearch/sam-3d-body"
COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"          # 2026-02-19
# ViT-H, 631M parameters: its backbone code is in the pinned repository. facebook/sam-3d-body-dinov3 (DINOv3-H+, 840M) scores the
# same on the published benchmarks (3DPW 54.8 / 54.8 MPJPE, EMDB 61.7 / 62.9, COCO 86.5 / 86.8 PCK) but builds its backbone from
# github facebookresearch/dinov3 through torch.hub at every load, unpinned.
WEIGHTS = "facebook/sam-3d-body-vith"
NEEDS = ("roma", "pytorch-lightning", "yacs", "omegaconf", "timm", "einops", "braceexpand")   # its inference path's imports
# A person facing the camera, upright: their left is the camera's +x, up is -y, forward (toward the camera) is -z.
FACING = np.diag([1.0, -1.0, -1.0])


def install(root: str | None = None) -> str:
    """The SAM 3D Body code at COMMIT (the repository is not a pip package): fetched once into root and put on sys.path."""
    root = root or os.path.join(os.path.expanduser("~"), ".cache", "qwen_symbolic_thinking", f"sam-3d-body-{COMMIT[:7]}")
    if not os.path.exists(os.path.join(root, "sam_3d_body", "__init__.py")):
        shutil.rmtree(root, ignore_errors=True)               # a fetch cut off earlier
        os.makedirs(root)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", REPO],
                    ["git", "fetch", "-q", "--depth", "1", "origin", COMMIT], ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=root, check=True)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load(weights: str = WEIGHTS, root: str | None = None) -> dict:
    """The fitter on the GPU, without a person detector or a field-of-view model: each picture is one person's box and the
    camera has the default field of view (a studio photo of one person; pass a box to predict() otherwise)."""
    from .hub import token
    token()                                                   # HF_TOKEN into the environment (Colab secrets) for the gated weights
    install(root)
    os.environ.setdefault("MOMENTUM_ENABLED", "0")            # the rig from the checkpoint's own TorchScript file, no extra package
    with contextlib.redirect_stdout(io.StringIO()):
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
        from sam_3d_body.metadata.mhr70 import pose_info
        model, cfg = load_sam_3d_body_hf(weights)
        estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    names = {v["name"]: int(v["id"]) for v in pose_info["keypoint_info"].values()}
    return {"estimator": estimator, "keypoints": names, "weights": weights, "commit": COMMIT}


def unload(fit: dict) -> None:
    import gc
    import torch
    fit.pop("estimator", None)
    gc.collect()
    torch.cuda.empty_cache()


def predict(fit: dict, picture, box=None) -> dict | None:
    """One photo (a PIL image) -> its record: the angles, the three frames, the 70 keypoints (camera frame, meters, about the
    body's root) and the focal length the fit assumed; None when no body was fitted. box: [x0, y0, x1, y1] of the person
    (default: the whole picture)."""
    img = np.asarray(picture.convert("RGB"))
    boxes = None if box is None else np.asarray(box, dtype=np.float32).reshape(1, 4)
    with contextlib.redirect_stdout(io.StringIO()):
        out = fit["estimator"].process_one_image(img, bboxes=boxes, inference_type="body")
    if not out:
        return None
    k3d = np.asarray(out[0]["pred_keypoints_3d"], dtype=np.float64)
    rec = angles(k3d, fit["keypoints"])
    if rec is None:
        return None
    rec["focal_length"] = round(float(np.asarray(out[0]["focal_length"]).reshape(-1)[0]), 1)
    rec["keypoints_3d"] = np.round(k3d, 3).tolist()
    return rec


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def frame(left: np.ndarray, rough: np.ndarray, rough_is_up: bool = True) -> np.ndarray | None:
    """A right-handed frame (columns: left, up, forward) from a left direction and a rough up (or, rough_is_up=False, a rough
    forward) direction; None when they are degenerate."""
    l = _unit(np.asarray(left, dtype=np.float64))
    if l is None:
        return None
    if rough_is_up:
        f = _unit(np.cross(l, rough))
        if f is None:
            return None
        u = np.cross(f, l)
    else:
        f = _unit(rough - np.dot(rough, l) * l)
        if f is None:
            return None
        u = np.cross(f, l)
    return np.stack([l, u, f], axis=1)


def turn_nod_tilt(r: np.ndarray) -> tuple[float, float, float]:
    """The intrinsic Y-X-Z angles (degrees) of a rotation written in a (left, up, forward) frame: the turn toward the subject's
    left, the nod up, the tilt toward the subject's left shoulder."""
    turn = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    nod_down = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
    tilt_right = math.degrees(math.atan2(r[1, 0], r[1, 1]))
    return turn, -nod_down, -tilt_right


def _tnt(r: np.ndarray) -> dict:
    t, n, s = turn_nod_tilt(r)
    return {"turn": round(t, 1), "nod": round(n, 1), "tilt": round(s, 1)}


def angles(k3d: np.ndarray, names: dict) -> dict | None:
    """The record for one fit from its keypoints (rows of k3d, camera frame): the hips, the chest and the head against the camera,
    the twist (the chest's turn against the hips; + the shoulders turned further to the subject's left), the head against the
    chest, the torso's lean against the camera's vertical, and the three frames (3x3, columns left / up / forward)."""
    p = {n: k3d[names[n]] for n in ("left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_ear", "right_ear",
                                     "left_eye", "right_eye")}
    mid_hip, mid_shoulder = (p["left_hip"] + p["right_hip"]) / 2, (p["left_shoulder"] + p["right_shoulder"]) / 2
    torso_up = mid_shoulder - mid_hip
    hips = frame(p["left_hip"] - p["right_hip"], torso_up)
    chest = frame(p["left_shoulder"] - p["right_shoulder"], torso_up)
    head = frame(p["left_ear"] - p["right_ear"], (p["left_eye"] + p["right_eye"]) / 2 - (p["left_ear"] + p["right_ear"]) / 2,
                 rough_is_up=False)
    if hips is None or chest is None or head is None:
        return None
    camera_up = np.array([0.0, -1.0, 0.0])
    up = _unit(torso_up)
    fwd_h = _unit(hips[:, 2] - np.dot(hips[:, 2], camera_up) * camera_up)      # the hips' forward, made level
    lean = None
    if up is not None and fwd_h is not None:
        left_h = np.cross(camera_up, fwd_h)
        lean = {"forward": round(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(up, fwd_h)))))), 1),
                "left": round(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(up, left_h)))))), 1),
                "picture": round(math.degrees(math.asin(max(-1.0, min(1.0, float(up[0]))))), 1)}   # + its top toward the right
    return {"hips": _tnt(FACING.T @ hips), "chest": _tnt(FACING.T @ chest), "twist": _tnt(hips.T @ chest)["turn"],
            "head": _tnt(FACING.T @ head), "head_on_chest": _tnt(chest.T @ head), "lean": lean,
            "frames": {"hips": np.round(hips, 4).tolist(), "chest": np.round(chest, 4).tolist(), "head": np.round(head, 4).tolist()}}


def _deg(x: float) -> int:
    return int(5 * round(abs(x) / 5))


def _picture(x: float) -> str:
    """The side of the picture a + / - points to: a turn toward the subject's left faces the right of the picture (sin turn > 0),
    whichever way the body faces; a + lean or tilt measured on the camera's x axis points right."""
    return "the right of the picture" if x > 0 else "the left of the picture"


def facing(turn: float, part: str = "the body") -> str:
    """Where a part faces, in the picture's own terms (a caption writer reads "the subject's left" as the picture's left)."""
    a = abs(turn)
    is_ = "are" if part.endswith("s") else "is"
    if a <= 20:
        return f"{part} {'face' if is_ == 'are' else 'faces'} the camera"
    if a <= 70:
        return f"{part} {is_} turned {_deg(turn)} degrees toward {_picture(turn)}"
    if a <= 110:
        return f"{part} {is_} in profile, facing {_picture(turn)}"
    if a <= 160:
        return f"{part} {is_} turned mostly away from the camera, facing {_picture(turn)}"
    if part == "the body":
        return "the body has its back to the camera"
    return f"{part} {'face' if is_ == 'are' else 'faces'} away from the camera"


def _up_x(rec: dict, frame: str) -> float | None:
    """The camera-x part of a frame's up axis (+: its top leans toward the right of the picture), from the stored frames."""
    try:
        return float(rec["frames"][frame][0][1])
    except (KeyError, IndexError, TypeError):
        return None


def words(rec: dict | None) -> str:
    """The pose line for a caption ask (empty without a record), every direction in the picture's terms: the body's facing, then
    only what stands out: the shoulders when they turn 15 degrees or more from the hips, the head when it turns 15 from the
    shoulders, nods 15 (the eye-ear line carries a few degrees of its own) or tilts 10, the torso's lean of 10."""
    if not rec:
        return ""
    parts = [facing(rec["hips"]["turn"])]
    if abs(rec["twist"]) >= 15:
        parts.append(facing(rec["chest"]["turn"], "the shoulders"))
    h, head = rec["head"], []
    if abs(rec["head_on_chest"]["turn"]) >= 15:
        head.append(facing(h["turn"], "the head"))
    if abs(h["nod"]) >= 15:
        head.append(f"the head is {'raised' if h['nod'] > 0 else 'lowered'} {_deg(h['nod'])} degrees")
    tilt = _up_x(rec, "head")
    if tilt is not None and abs(math.degrees(math.asin(max(-1.0, min(1.0, tilt))))) >= 10:
        head.append(f"the head tilts {_deg(math.degrees(math.asin(max(-1.0, min(1.0, tilt)))))} degrees toward {_picture(tilt)}")
    parts += head
    lean = rec.get("lean") or {}
    if abs(lean.get("forward", 0)) >= 10:
        parts.append(f"the torso leans {'forward' if lean['forward'] > 0 else 'back'} {_deg(lean['forward'])} degrees")
    if abs(lean.get("picture", 0)) >= 10:
        parts.append(f"the torso leans {_deg(lean['picture'])} degrees toward {_picture(lean['picture'])}")
    return "Pose (3D body fit): " + "; ".join(parts) + "."
