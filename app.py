"""Step 3 - the meme reaction camera.

Webcam on the left, the meme that matches your face + gesture on the right.
If nothing is similar enough, the right panel stays empty.

Usage:
    python app.py
    python app.py --threshold 0.8 --camera 1

Keys:
    + / -   raise / lower the threshold
    [ / ]   fewer / more hold frames (fewer = meme appears faster)
    D       show / hide debug text
    S       save a screenshot to screenshots/
    Q/ESC   quit
"""
from __future__ import annotations

import argparse
import time

import cv2
import joblib
import numpy as np

from meme_cam import config
from meme_cam.camera import CameraStream, Layout, create_window, window_closed
from meme_cam.features import FEATURE_DIM, FeatureExtractor
from meme_cam.memes import load_memes
from meme_cam.stabilizer import MatchStabilizer
from meme_cam.ui import YELLOW, draw_debug, draw_landmarks, meme_panel, put_text, side_by_side

WINDOW = "meme reaction camera"


def load_classifier():
    if not config.CLASSIFIER_PATH.exists():
        raise SystemExit("No trained model found. Run: python collect.py  then  python train.py")
    bundle = joblib.load(config.CLASSIFIER_PATH)
    if bundle["feature_dim"] != FEATURE_DIM:
        raise SystemExit("Model was trained with another feature version. Run train.py again.")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    ap.add_argument("--alpha", type=float, default=config.SMOOTH_ALPHA,
                    help="smoothing: 1.0 = no smoothing (fastest), 0.3 = very smooth")
    ap.add_argument("--hold", type=int, default=config.HOLD_FRAMES,
                    help="frames a meme must win before it is shown (1 = instant)")
    ap.add_argument("--release", type=int, default=config.RELEASE_FRAMES,
                    help="frames below threshold before the meme is hidden")
    ap.add_argument("--process-width", type=int, default=config.PROCESS_WIDTH,
                    help="downscale frames to this width for MediaPipe (smaller = faster)")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--height", type=int, default=config.DISPLAY_HEIGHT,
                    help="window height in pixels (window size is fixed)")
    ap.add_argument("--cam-width", type=int, default=config.CAMERA_WIDTH)
    ap.add_argument("--cam-height", type=int, default=config.CAMERA_HEIGHT)
    args = ap.parse_args()

    bundle = load_classifier()
    model, labels = bundle["model"], bundle["labels"]
    print(f"Model: {bundle['model_name']}  labels: {labels}  trained: {bundle['trained_at']}")

    memes = load_memes()
    for lb in labels:
        if lb != config.NONE_LABEL and lb not in memes:
            print(f"[warn] no image for label '{lb}' in memes/")

    stab = MatchStabilizer(labels, threshold=args.threshold, alpha=args.alpha,
                           hold_frames=args.hold, release_frames=args.release)
    cam = CameraStream(args.camera, args.cam_width, args.cam_height)
    extractor = FeatureExtractor(process_width=args.process_width)
    create_window(WINDOW)
    layout = None
    panels: dict[str | None, np.ndarray] = {}   # meme panels resized once, reused
    debug = not args.no_debug
    fps, last = 0.0, time.time()
    fid = 0

    try:
        while True:
            fid, frame = cam.read(fid)          # always the newest camera frame
            if frame is None:
                print("Camera frame not available.")
                break
            info = extractor.process(frame)

            probs = None
            if info.has_anything:
                probs = model.predict_proba(info.features.reshape(1, -1))[0]
            shown = stab.update(probs)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if layout is None:
                layout = Layout(frame.shape[1], frame.shape[0], height=args.height)
                panels = {lb: meme_panel(img, layout.panel_w, layout.height)
                          for lb, img in memes.items()}
                panels[None] = meme_panel(None, layout.panel_w, layout.height)
            view = frame.copy()
            draw_landmarks(view, info)          # full resolution
            view = layout.camera(view)          # fixed size, same aspect ratio
            if debug:
                draw_debug(view, info, stab.top_k(3), shown, stab.threshold, fps,
                           extra=[f"hold: {stab.hold_frames}  mediapipe: {extractor.last_ms:.0f}ms"])

            if shown is not None and shown in panels:
                panel = panels[shown].copy()
                p = float(stab.smoothed[labels.index(shown)])
                put_text(panel, f"{shown} ({p:.0%})", (12, layout.height - 16), YELLOW, 0.7)
            else:
                panel = panels[None]
            canvas = side_by_side(view, panel)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("+"), ord("=")):
                stab.threshold = min(0.99, stab.threshold + 0.05)
            elif key in (ord("-"), ord("_")):
                stab.threshold = max(0.05, stab.threshold - 0.05)
            elif key == ord("["):
                stab.hold_frames = max(1, stab.hold_frames - 1)
            elif key == ord("]"):
                stab.hold_frames = min(15, stab.hold_frames + 1)
            elif key == ord("d"):
                debug = not debug
            elif key == ord("s"):
                out = config.ROOT / "screenshots"
                out.mkdir(exist_ok=True)
                path = out / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                ok, buf = cv2.imencode(".png", canvas)
                if ok:
                    np.asarray(buf).tofile(str(path))
                    print(f"saved {path}")
            if window_closed(WINDOW):
                break
    finally:
        extractor.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
