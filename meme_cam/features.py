"""Turn one webcam frame into a fixed-length feature vector.

Pipeline:  frame (BGR)  ->  MediaPipe Face Landmarker + Hand Landmarker
                        ->  numbers that describe expression + gesture.

Feature groups (total = FEATURE_DIM):
  face_present (1)          1 if a face was found
  blendshapes (52)          expression scores 0..1 (smile, jawOpen, eyeBlink...)
  head_pose (3)             pitch / yaw / roll, divided by pi
  for each hand slot (Left, Right):
    present (1)
    shape (63)              21 landmarks relative to the wrist, scale-free
    fingers (5)             how straight each finger is (thumb..pinky)
    pos_to_face (2)         hand centre relative to the face, in face widths
"""
from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

# The 52 blendshape names MediaPipe returns (fixed order we use for features).
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight",
]
N_BLENDSHAPES = len(BLENDSHAPE_NAMES)  # 52
_BS_INDEX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}

HAND_SLOTS = ("Left", "Right")
N_HAND_LANDMARKS = 21
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# (base joint, tip) for each finger. Thumb is compared to the index MCP instead.
_FINGER_JOINTS = {"thumb": (2, 4), "index": (5, 8), "middle": (9, 12),
                  "ring": (13, 16), "pinky": (17, 20)}

# Hand skeleton, used for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _feature_names() -> list[str]:
    names = ["face_present"]
    names += [f"bs_{n}" for n in BLENDSHAPE_NAMES]
    names += ["head_pitch", "head_yaw", "head_roll"]
    for slot in HAND_SLOTS:
        p = f"hand{slot}"
        names.append(f"{p}_present")
        names += [f"{p}_lm{i:02d}_{a}" for i in range(N_HAND_LANDMARKS) for a in "xyz"]
        names += [f"{p}_ext_{f}" for f in FINGER_NAMES]
        names += [f"{p}_pos_x", f"{p}_pos_y"]
    return names


FEATURE_NAMES = _feature_names()
FEATURE_DIM = len(FEATURE_NAMES)
HAND_BLOCK = 1 + N_HAND_LANDMARKS * 3 + len(FINGER_NAMES) + 2  # 71
EPS = 1e-6


# ----------------------------------------------------------------------------
# Pure numpy feature building (easy to unit-test, no MediaPipe needed)
# ----------------------------------------------------------------------------
def head_pose_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Pitch, yaw, roll (radians) from MediaPipe's 4x4 facial transformation matrix."""
    r = np.asarray(matrix, dtype=float)[:3, :3]
    # remove scale so the rotation part is orthonormal
    r = r / (np.linalg.norm(r, axis=0, keepdims=True) + EPS)
    pitch = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    roll = math.atan2(r[1, 0], r[0, 0])
    return pitch, yaw, roll


def hand_block(hand_pts: np.ndarray | None,
               face_center: np.ndarray | None,
               face_width: float | None) -> np.ndarray:
    """Features for one hand slot. hand_pts: (21, 3) in aspect-corrected units."""
    out = np.zeros(HAND_BLOCK, dtype=np.float32)
    if hand_pts is None:
        return out
    pts = np.asarray(hand_pts, dtype=np.float32)
    rel = pts - pts[0]                                   # wrist at origin
    scale = float(np.linalg.norm(rel[9, :2])) + EPS      # wrist -> middle MCP
    shape = rel / scale

    ext = []
    for f in FINGER_NAMES:
        base, tip = _FINGER_JOINTS[f]
        if f == "thumb":
            # thumb out = tip far from the index knuckle
            ext.append(np.linalg.norm(shape[tip, :2] - shape[5, :2]))
        else:
            ext.append(np.linalg.norm(shape[tip, :2]) /
                       (np.linalg.norm(shape[base, :2]) + EPS))

    pos = np.zeros(2, dtype=np.float32)
    if face_center is not None and face_width:
        centre = pts[:, :2].mean(axis=0)
        pos = (centre - face_center) / (face_width + EPS)

    out[0] = 1.0
    out[1:1 + 63] = shape.reshape(-1)
    out[64:69] = ext
    out[69:71] = pos
    return out


def build_features(face_pts: np.ndarray | None,
                   blendshapes: dict[str, float] | None,
                   face_matrix: np.ndarray | None,
                   hands: dict[str, np.ndarray]) -> np.ndarray:
    """Combine everything into one vector of length FEATURE_DIM.

    face_pts: (N, 3) face landmarks in aspect-corrected units, or None.
    hands:    {"Left": (21,3) array, "Right": (21,3) array} (missing = absent).
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    face_center, face_width = None, None
    if face_pts is not None:
        vec[0] = 1.0
        xy = np.asarray(face_pts)[:, :2]
        face_center = xy.mean(axis=0)
        face_width = float(xy[:, 0].max() - xy[:, 0].min())
        if blendshapes:
            for name, score in blendshapes.items():
                i = _BS_INDEX.get(name)
                if i is not None:
                    vec[1 + i] = score
        if face_matrix is not None:
            vec[1 + N_BLENDSHAPES:1 + N_BLENDSHAPES + 3] = (
                np.array(head_pose_from_matrix(face_matrix)) / math.pi)

    start = 1 + N_BLENDSHAPES + 3
    for k, slot in enumerate(HAND_SLOTS):
        s = start + k * HAND_BLOCK
        vec[s:s + HAND_BLOCK] = hand_block(hands.get(slot), face_center, face_width)
    return vec


# ----------------------------------------------------------------------------
# MediaPipe wrapper
# ----------------------------------------------------------------------------
def ensure_models() -> None:
    """Download the two MediaPipe model files the first time."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ((config.FACE_MODEL_PATH, config.FACE_MODEL_URL),
                      (config.HAND_MODEL_PATH, config.HAND_MODEL_URL)):
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Downloading {path.name} ...")
        tmp = Path(str(path) + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
        print(f"  saved to {path}")


@dataclass
class FrameInfo:
    """Everything we found in one frame (for features + drawing)."""
    features: np.ndarray
    face_box: tuple[int, int, int, int] | None = None       # pixels x1,y1,x2,y2
    hands_px: dict[str, np.ndarray] = field(default_factory=dict)  # slot -> (21,2) px
    blendshapes: dict[str, float] = field(default_factory=dict)

    @property
    def has_face(self) -> bool:
        return self.face_box is not None

    @property
    def has_anything(self) -> bool:
        return self.face_box is not None or bool(self.hands_px)


class FeatureExtractor:
    """Runs Face + Hand landmarkers in VIDEO mode on BGR frames."""

    def __init__(self, num_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        ensure_models()
        self._mp = mp
        mode = vision.RunningMode.VIDEO
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=config.FACE_MODEL_PATH.read_bytes()),
                running_mode=mode,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            ))
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=config.HAND_MODEL_PATH.read_bytes()),
                running_mode=mode,
                num_hands=num_hands,
            ))
        self._ts = 0

    # model_asset_buffer (bytes) is used instead of a path so that folders with
    # non-ASCII characters on Windows do not break MediaPipe.

    def _next_ts(self) -> int:
        self._ts = max(self._ts + 1, int(time.monotonic() * 1000))
        return self._ts

    def process(self, frame_bgr: np.ndarray) -> FrameInfo:
        h, w = frame_bgr.shape[:2]
        aspect = w / h
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = self._next_ts()
        face_res = self._face.detect_for_video(image, ts)
        hand_res = self._hands.detect_for_video(image, ts)

        info = FrameInfo(features=np.zeros(FEATURE_DIM, dtype=np.float32))

        # --- face
        face_pts, bs, matrix = None, None, None
        if face_res.face_landmarks:
            lms = face_res.face_landmarks[0]
            raw = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            face_pts = raw * np.array([aspect, 1.0, aspect], dtype=np.float32)
            x1, y1 = raw[:, 0].min() * w, raw[:, 1].min() * h
            x2, y2 = raw[:, 0].max() * w, raw[:, 1].max() * h
            info.face_box = (int(x1), int(y1), int(x2), int(y2))
            if face_res.face_blendshapes:
                bs = {c.category_name: float(c.score) for c in face_res.face_blendshapes[0]}
                info.blendshapes = bs
            if face_res.facial_transformation_matrixes:
                matrix = np.asarray(face_res.facial_transformation_matrixes[0])

        # --- hands (put each hand in its Left / Right slot)
        hands: dict[str, np.ndarray] = {}
        for i, lms in enumerate(hand_res.hand_landmarks or []):
            label = "Right"
            if hand_res.handedness and hand_res.handedness[i]:
                label = hand_res.handedness[i][0].category_name
            if label in hands:  # two hands with same label -> use the free slot
                label = "Left" if label == "Right" else "Right"
            if label in hands:
                continue
            raw = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            hands[label] = raw * np.array([aspect, 1.0, aspect], dtype=np.float32)
            info.hands_px[label] = raw[:, :2] * np.array([w, h], dtype=np.float32)

        info.features = build_features(face_pts, bs, matrix, hands)
        return info

    def close(self) -> None:
        self._face.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
