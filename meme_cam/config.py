"""Project paths and default settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MEMES_DIR = ROOT / "memes"          # put meme images here: <label>.jpg / .png ...
DATA_DIR = ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
MODELS_DIR = ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Special label for "my pose does not look like any meme".
NONE_LABEL = "_none"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Window (fixed size, not resizable -> the camera image is never stretched)
DISPLAY_HEIGHT = 540       # height of the window content in pixels
MEME_PANEL_RATIO = 0.8     # meme panel width = DISPLAY_HEIGHT * ratio
SCREEN_FILL = 0.92         # never use more than this part of the screen

# Realtime matching defaults (can be changed with CLI flags / keys in app.py)
DEFAULT_THRESHOLD = 0.70   # minimum smoothed probability to show a meme
SMOOTH_WINDOW = 8          # average probabilities over the last N frames
HOLD_FRAMES = 5            # a label must win N frames in a row before it is shown
RELEASE_MARGIN = 0.10      # hysteresis: hide when prob < threshold - margin
