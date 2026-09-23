"""Smooth the classifier output so the meme does not flicker.

Rules:
  1. Average the probability vectors of the last `window` frames.
  2. A meme is SHOWN only when the same label is the top one, is not the
     "_none" label, and its averaged probability >= threshold for
     `hold_frames` frames in a row.
  3. A shown meme is HIDDEN when its averaged probability drops below
     threshold - release_margin (hysteresis), or another label takes over.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import config


class MatchStabilizer:
    def __init__(self, labels: list[str],
                 threshold: float = config.DEFAULT_THRESHOLD,
                 window: int = config.SMOOTH_WINDOW,
                 hold_frames: int = config.HOLD_FRAMES,
                 release_margin: float = config.RELEASE_MARGIN,
                 none_label: str = config.NONE_LABEL):
        self.labels = list(labels)
        self.threshold = threshold
        self.hold_frames = hold_frames
        self.release_margin = release_margin
        self.none_label = none_label
        self._history: deque[np.ndarray] = deque(maxlen=window)
        self._candidate: str | None = None
        self._count = 0
        self.current: str | None = None       # label being shown (or None)
        self.smoothed = np.zeros(len(labels))

    def reset(self) -> None:
        self._history.clear()
        self._candidate, self._count, self.current = None, 0, None
        self.smoothed = np.zeros(len(self.labels))

    def update(self, probs: np.ndarray | None) -> str | None:
        """Feed one frame of probabilities (or None when nobody is visible)."""
        if probs is None:
            # nobody in front of the camera: forget quickly
            self.reset()
            return None

        self._history.append(np.asarray(probs, dtype=float))
        self.smoothed = np.mean(self._history, axis=0)
        top = int(np.argmax(self.smoothed))
        top_label, top_p = self.labels[top], float(self.smoothed[top])

        # keep showing the current meme while it is still good enough
        if self.current is not None:
            cur_p = float(self.smoothed[self.labels.index(self.current)])
            if top_label == self.current and cur_p >= self.threshold - self.release_margin:
                return self.current
            self.current = None

        # look for a new meme
        good = top_label != self.none_label and top_p >= self.threshold
        if good and top_label == self._candidate:
            self._count += 1
        elif good:
            self._candidate, self._count = top_label, 1
        else:
            self._candidate, self._count = None, 0

        if self._candidate is not None and self._count >= self.hold_frames:
            self.current = self._candidate
        return self.current

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        idx = np.argsort(self.smoothed)[::-1][:k]
        return [(self.labels[i], float(self.smoothed[i])) for i in idx]
