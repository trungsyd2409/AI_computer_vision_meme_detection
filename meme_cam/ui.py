"""Drawing helpers (OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from .features import HAND_CONNECTIONS, FrameInfo

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
YELLOW = (0, 220, 255)
RED = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Blendshapes worth showing in the debug text (similar to the TikTok video).
DEBUG_BLENDSHAPES = [
    ("smile", ("mouthSmileLeft", "mouthSmileRight")),
    ("jaw_open", ("jawOpen",)),
    ("blink_left", ("eyeBlinkLeft",)),
    ("blink_right", ("eyeBlinkRight",)),
    ("brow_up", ("browInnerUp",)),
    ("pucker", ("mouthPucker",)),
    ("frown", ("mouthFrownLeft", "mouthFrownRight")),
]


def put_text(img, text, org, color=GREEN, scale=0.6, thick=2):
    x, y = org
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, info: FrameInfo) -> None:
    if info.face_box:
        x1, y1, x2, y2 = info.face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
    for pts in info.hands_px.values():
        p = pts.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(p[a]), tuple(p[b]), WHITE, 2, cv2.LINE_AA)
        for q in p:
            cv2.circle(frame, tuple(q), 4, GREEN, -1, cv2.LINE_AA)


def draw_debug(frame: np.ndarray, info: FrameInfo,
               top: list[tuple[str, float]], shown: str | None,
               threshold: float, fps: float, extra: list[str] | None = None) -> None:
    lines: list[tuple[str, tuple]] = []
    lines.append((f"match: {shown or '-'}", YELLOW if shown else GREY))
    for label, p in top:
        lines.append((f"  {label}: {p:.2f}", GREEN if p >= threshold else GREY))
    lines.append((f"threshold: {threshold:.2f}", WHITE))
    for name, keys in DEBUG_BLENDSHAPES:
        v = np.mean([info.blendshapes.get(k, 0.0) for k in keys])
        lines.append((f"{name}: {v:.2f}", GREEN))
    lines.append((f"hands: {', '.join(info.hands_px) or 'none'}", GREEN))
    lines.append((f"fps: {fps:.0f}", WHITE))
    for text in extra or []:
        lines.append((text, WHITE))
    y = 28
    for text, color in lines:
        put_text(frame, text, (12, y), color)
        y += 26


def fit_into(img: np.ndarray, w: int, h: int, bg=(245, 245, 245)) -> np.ndarray:
    """Resize keeping aspect ratio and centre it on a w x h canvas."""
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def meme_panel(meme: np.ndarray | None, w: int, h: int,
               caption: str | None = None, empty_text: str = "no match") -> np.ndarray:
    if meme is None:
        panel = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
        (tw, _), _ = cv2.getTextSize(empty_text, FONT, 1.0, 2)
        cv2.putText(panel, empty_text, ((w - tw) // 2, h // 2), FONT, 1.0,
                    (180, 180, 180), 2, cv2.LINE_AA)
        return panel
    panel = fit_into(meme, w, h)
    if caption:
        put_text(panel, caption, (12, h - 16), YELLOW, 0.7)
    return panel


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 12) -> np.ndarray:
    h = left.shape[0]
    if right.shape[0] != h:
        right = fit_into(right, right.shape[1], h)
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.hstack([left, sep, right])
