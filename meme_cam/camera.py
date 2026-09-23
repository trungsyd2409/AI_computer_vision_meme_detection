"""Webcam + window helpers."""
from __future__ import annotations

import os
import threading
import time

import cv2

from . import config


def open_camera(index: int = config.CAMERA_INDEX,
                width: int = config.CAMERA_WIDTH,
                height: int = config.CAMERA_HEIGHT) -> cv2.VideoCapture:
    # DirectShow opens much faster than the default backend on Windows.
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index}. Try --camera 1")
    # MJPG lets most USB/laptop webcams deliver HD at full fps (set it before the size).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if ok:
        h, w = frame.shape[:2]
        print(f"Camera {index}: {w}x{h} (aspect {w / h:.2f})")
    return cap


def read_mirrored(cap: cv2.VideoCapture):
    """Read one frame and flip it like a mirror (selfie view)."""
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1)


class CameraStream:
    """Reads the webcam in a background thread and keeps ONLY the newest frame.

    Without this, OpenCV keeps a queue of old frames: when processing is slower
    than the camera (e.g. 20 fps vs 30 fps) you always see a frame from the
    past, so the picture (and the meme) feels delayed. Here old frames are
    simply dropped.
    """

    def __init__(self, index: int = config.CAMERA_INDEX,
                 width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT):
        self.cap = open_camera(index, width, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ignored by some drivers, thread handles it
        self._cond = threading.Condition()
        self._frame = None
        self._id = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._id += 1
                self._cond.notify_all()

    def read(self, last_id: int = 0, timeout: float = 2.0):
        """Wait for a frame newer than `last_id`. Returns (id, mirrored frame) or (id, None)."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._id != last_id and self._frame is not None,
                                     timeout=timeout)
            if not ok:
                return last_id, None
            frame, fid = self._frame, self._id
        return fid, cv2.flip(frame, 1)

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def screen_size() -> tuple[int, int] | None:
    """Screen size in the same (logical) pixels an OpenCV window uses."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:  # noqa: BLE001
        return None


class Layout:
    """Fixed canvas: [camera (true aspect ratio)] [gap] [meme panel].

    Computed once from the first frame, so the window never changes size and
    the camera image is only scaled uniformly (never stretched).
    """

    GAP = 12

    def __init__(self, frame_w: int, frame_h: int,
                 height: int = config.DISPLAY_HEIGHT,
                 panel_ratio: float = config.MEME_PANEL_RATIO):
        aspect = frame_w / frame_h
        screen = screen_size()
        if screen:
            sw, sh = screen
            max_h = int(sh * config.SCREEN_FILL) - 60           # title bar / taskbar
            max_w = int(sw * config.SCREEN_FILL)
            height = min(height, max_h,
                         int((max_w - self.GAP) / (aspect + panel_ratio)))
        self.height = int(height)
        self.cam_w = int(round(self.height * aspect))
        self.panel_w = int(round(self.height * panel_ratio))
        self.width = self.cam_w + self.GAP + self.panel_w

    def camera(self, frame):
        """Scale the camera frame to the fixed size (uniform scale = no stretch)."""
        return cv2.resize(frame, (self.cam_w, self.height), interpolation=cv2.INTER_AREA)


def create_window(name: str) -> None:
    # AUTOSIZE = window has exactly the image size and cannot be resized by
    # dragging, so the picture is always shown 1:1 without distortion.
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)


def window_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
