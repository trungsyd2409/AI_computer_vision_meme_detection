"""Decide which meme to show, fast but without flicker.

"Fast attack, slow release":
  1. Smooth probabilities with an exponential moving average (EMA).
     alpha close to 1 = reacts instantly, close to 0 = very smooth.
  2. SHOW a meme as soon as it is the top label (not "_none") with
     smoothed prob >= threshold for `hold_frames` frames (default 2).
     Switching to another meme uses the same rule, so it is also fast.
  3. HIDE the current meme only after its prob stays below
     threshold - release_margin for `release_frames` frames in a row,
     so one bad frame does not make it blink.
  4. If nobody is visible for `release_frames` frames -> hide.

Delay to show  ~= hold_frames / fps   (e.g. 2 frames at 25 fps = 80 ms)
"""
from __future__ import annotations

import numpy as np

from . import config


class MatchStabilizer:
    def __init__(self, labels: list[str],
                 threshold: float = config.DEFAULT_THRESHOLD,
                 alpha: float = config.SMOOTH_ALPHA,
                 hold_frames: int = config.HOLD_FRAMES,
                 release_frames: int = config.RELEASE_FRAMES,
                 release_margin: float = config.RELEASE_MARGIN,
                 none_label: str = config.NONE_LABEL):
        self.labels = list(labels)
        self._index = {lb: i for i, lb in enumerate(self.labels)}
        self.threshold = threshold
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.hold_frames = max(1, hold_frames)
        self.release_frames = max(1, release_frames)
        self.release_margin = release_margin
        self.none_label = none_label
        self.reset()

    def reset(self) -> None:
        self.smoothed = np.zeros(len(self.labels))
        self._has_history = False
        self._candidate: str | None = None
        self._count = 0
        self._below = 0
        self._missing = 0
        self.current: str | None = None       # label being shown (or None)

    def _prob(self, label: str) -> float:
        return float(self.smoothed[self._index[label]])

    def update(self, probs: np.ndarray | None) -> str | None:
        """Feed one frame of probabilities (None = nobody visible)."""
        if probs is None:
            self._missing += 1
            if self._missing >= self.release_frames:
                self.reset()
            return self.current
        self._missing = 0

        probs = np.asarray(probs, dtype=float)
        if self._has_history:
            self.smoothed = self.alpha * probs + (1 - self.alpha) * self.smoothed
        else:
            self.smoothed, self._has_history = probs.copy(), True

        top = int(np.argmax(self.smoothed))
        top_label, top_p = self.labels[top], float(self.smoothed[top])
        good = top_label != self.none_label and top_p >= self.threshold

        # 1) candidate for a new / different meme
        if good and top_label != self.current:
            if top_label == self._candidate:
                self._count += 1
            else:
                self._candidate, self._count = top_label, 1
            if self._count >= self.hold_frames:
                self.current, self._candidate, self._count, self._below = top_label, None, 0, 0
                return self.current
        elif not good or top_label == self.current:
            self._candidate, self._count = None, 0

        # 2) release the current meme when it is clearly gone
        if self.current is not None:
            if self._prob(self.current) < self.threshold - self.release_margin:
                self._below += 1
                if self._below >= self.release_frames:
                    self.current, self._below = None, 0
            else:
                self._below = 0
        return self.current

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        idx = np.argsort(self.smoothed)[::-1][:k]
        return [(self.labels[i], float(self.smoothed[i])) for i in idx]
