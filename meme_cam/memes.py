"""Load meme images from the memes/ folder. The file name (without extension)
is the label, e.g. memes/hamster_peace.jpg -> label "hamster_peace"."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config


def read_image(path: Path) -> np.ndarray | None:
    """Read an image as BGR. Works with non-ASCII Windows paths and GIF (first frame)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        from PIL import Image  # fallback, e.g. for .gif
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot read {path.name}: {exc}")
        return None


def list_meme_files(folder: Path = config.MEMES_DIR) -> dict[str, Path]:
    if not folder.exists():
        return {}
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS)
    return {p.stem: p for p in files}


def load_memes(folder: Path = config.MEMES_DIR) -> dict[str, np.ndarray]:
    memes = {}
    for label, path in list_meme_files(folder).items():
        img = read_image(path)
        if img is not None:
            memes[label] = img
    return memes
