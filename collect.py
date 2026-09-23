"""Step 1 - record training samples for each meme.

For every meme in memes/ you copy the pose/expression in front of the webcam.
Each frame becomes one row (feature vector) in data/dataset.csv.

Usage:
    python collect.py                  # go through all memes + "_none"
    python collect.py --label hamster  # only one label
    python collect.py --seconds 6      # longer recordings

Keys in the window:
    SPACE  start recording (3 s countdown, then records --seconds)
    N / P  next / previous meme
    U      undo the last recording
    Q/ESC  quit (data is saved after every recording)

Tips: record 3-4 short sessions per meme, change distance / angle /
lighting a little between sessions. For "_none" record normal things:
neutral face, talking, looking around, random hand movements.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd

from meme_cam import config
from meme_cam.camera import CameraStream, Layout, create_window, window_closed
from meme_cam.features import FEATURE_DIM, FEATURE_NAMES, FeatureExtractor
from meme_cam.memes import load_memes
from meme_cam.ui import GREY, RED, WHITE, YELLOW, draw_landmarks, meme_panel, put_text, side_by_side

WINDOW = "collect - meme reaction cam"
NONE_HELP = ["NOT a meme:", "neutral face, talk,", "look around, move", "hands randomly"]


def load_dataset() -> pd.DataFrame:
    if config.DATASET_CSV.exists():
        df = pd.read_csv(config.DATASET_CSV)
        if list(df.columns[2:]) != FEATURE_NAMES:
            raise SystemExit(
                "data/dataset.csv was made with a different feature version. "
                "Rename or delete it and record again.")
        return df
    return pd.DataFrame(columns=["label", "session", *FEATURE_NAMES])


def save_dataset(df: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_CSV, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="record only this label (meme file name or _none)")
    ap.add_argument("--seconds", type=float, default=4.0, help="length of one recording")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--height", type=int, default=config.DISPLAY_HEIGHT,
                    help="window height in pixels (window size is fixed)")
    ap.add_argument("--cam-width", type=int, default=config.CAMERA_WIDTH)
    ap.add_argument("--cam-height", type=int, default=config.CAMERA_HEIGHT)
    args = ap.parse_args()

    memes = load_memes()
    labels = list(memes) + [config.NONE_LABEL]
    if args.label:
        labels = [args.label]
    if len(labels) == 1 and labels[0] == config.NONE_LABEL and not args.label:
        print(f"[warn] no images found in {config.MEMES_DIR}. Add meme images first.")

    df = load_dataset()
    cam = CameraStream(args.camera, args.cam_width, args.cam_height)
    extractor = FeatureExtractor()
    create_window(WINDOW)
    layout = None

    idx = 0
    state = "idle"          # idle -> countdown -> recording -> idle
    t0 = 0.0
    rows: list[np.ndarray] = []
    session = ""
    sessions_done: list[str] = []
    fid = 0

    try:
        while True:
            fid, frame = cam.read(fid)
            if frame is None:
                print("Camera frame not available.")
                break
            label = labels[idx]
            info = extractor.process(frame)
            if layout is None:
                layout = Layout(frame.shape[1], frame.shape[0], height=args.height)
            view = frame.copy()
            draw_landmarks(view, info)          # full resolution
            view = layout.camera(view)          # fixed size, same aspect ratio
            now = time.time()

            if state == "countdown":
                left = args.countdown - (now - t0)
                if left <= 0:
                    state, t0, rows = "recording", now, []
                    session = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
                else:
                    put_text(view, f"{left:.0f}", (view.shape[1] // 2 - 30, view.shape[0] // 2),
                             YELLOW, 4.0, 8)
            if state == "recording":
                if info.has_anything:
                    rows.append(info.features)
                left = args.seconds - (now - t0)
                cv2.circle(view, (view.shape[1] - 40, 40), 16, RED, -1)
                put_text(view, f"REC {left:.1f}s  frames={len(rows)}",
                         (view.shape[1] - 330, 48), RED, 0.7)
                if left <= 0:
                    state = "idle"
                    if rows:
                        new = pd.DataFrame(np.vstack(rows), columns=FEATURE_NAMES)
                        new.insert(0, "session", session)
                        new.insert(0, "label", label)
                        df = new if df.empty else pd.concat([df, new], ignore_index=True)
                        save_dataset(df)
                        sessions_done.append(session)
                        print(f"saved {len(rows)} frames for '{label}' ({session})")
                    else:
                        print("nothing detected - recording skipped")

            n_rows = int((df["label"] == label).sum()) if not df.empty else 0
            n_sess = df.loc[df["label"] == label, "session"].nunique() if not df.empty else 0
            put_text(view, f"[{idx + 1}/{len(labels)}] {label}", (12, 30), YELLOW, 0.8)
            put_text(view, f"samples: {n_rows}  sessions: {n_sess}", (12, 60), WHITE)
            put_text(view, "SPACE record | N/P next/prev | U undo | Q quit",
                     (12, view.shape[0] - 16), GREY, 0.55)
            if not info.has_face:
                put_text(view, "no face detected", (12, 90), RED)

            target = memes.get(label)
            panel = meme_panel(target, layout.panel_w, layout.height, caption="copy this!" if target is not None else None,
                               empty_text="(no image)")
            if label == config.NONE_LABEL:
                for i, line in enumerate(NONE_HELP):
                    put_text(panel, line, (20, 60 + i * 34), YELLOW, 0.8)
            cv2.imshow(WINDOW, side_by_side(view, panel))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if state == "idle":
                if key == ord(" "):
                    state, t0 = "countdown", now
                elif key == ord("n"):
                    idx = (idx + 1) % len(labels)
                elif key == ord("p"):
                    idx = (idx - 1) % len(labels)
                elif key == ord("u") and sessions_done:
                    last = sessions_done.pop()
                    df = df[df["session"] != last].reset_index(drop=True)
                    save_dataset(df)
                    print(f"removed session {last}")
            if window_closed(WINDOW):
                break
    finally:
        extractor.close()
        cam.release()
        cv2.destroyAllWindows()

    if not df.empty:
        print("\nSamples per label:")
        print(df.groupby("label")["session"].agg(frames="size", sessions="nunique"))
    print(f"\nDataset: {config.DATASET_CSV}  (feature dim = {FEATURE_DIM})")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
