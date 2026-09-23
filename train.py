"""Step 2 - train a classifier on data/dataset.csv.

It compares several scikit-learn models with cross-validation, prints a
report, then trains the best one on all data and saves it to
models/classifier.joblib.

Usage:
    python train.py                 # compare models, keep the best
    python train.py --model svm     # force one model

Why "grouped" cross-validation?  Frames from the same recording are almost
identical. If some frames of a recording are in the train fold and the others
in the test fold, the score looks great but is too optimistic. So we split by
recording session (StratifiedGroupKFold) whenever every label has >= 2 sessions.
"""
from __future__ import annotations

import argparse
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from meme_cam import config
from meme_cam.features import FEATURE_DIM, FEATURE_NAMES
from meme_cam.memes import list_meme_files


def make_models(seed: int = 42) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
        "svm": make_pipeline(StandardScaler(), SVC(C=5, gamma="scale", probability=True,
                                                   random_state=seed)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                n_jobs=-1, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64),
                                                             alpha=1e-3, max_iter=800,
                                                             random_state=seed)),
    }


def choose_cv(y: np.ndarray, groups: np.ndarray, seed: int = 42):
    sessions_per_label = pd.Series(groups).groupby(y).nunique()
    min_sessions = int(sessions_per_label.min())
    if min_sessions >= 2:
        k = min(5, min_sessions)
        print(f"CV: StratifiedGroupKFold(k={k}) - split by recording session")
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed), groups
    min_count = int(pd.Series(y).value_counts().min())
    k = max(2, min(5, min_count))
    print(f"[warn] some labels have only 1 session -> StratifiedKFold(k={k}). "
          "Scores will be too optimistic. Record >= 2 sessions per label.")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(make_models()), help="skip comparison, use this model")
    ap.add_argument("--data", default=str(config.DATASET_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if list(df.columns[2:]) != FEATURE_NAMES:
        raise SystemExit("Dataset columns do not match the current feature version.")
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    groups = df["session"].astype(str).to_numpy()

    # ---- sanity checks
    summary = df.groupby("label")["session"].agg(frames="size", sessions="nunique")
    print("Dataset summary:\n", summary, "\n", sep="")
    labels = sorted(set(y))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 labels (e.g. 2 memes, or 1 meme + _none).")
    meme_files = list_meme_files()
    for lb in labels:
        if lb != config.NONE_LABEL and lb not in meme_files:
            print(f"[warn] label '{lb}' has data but no image in memes/")
    for lb in meme_files:
        if lb not in labels:
            print(f"[warn] meme '{lb}' has no training data (it will never be shown)")
    if config.NONE_LABEL not in labels:
        print(f"[warn] no '{config.NONE_LABEL}' samples. The app can only rely on the "
              "threshold to reject non-meme poses. Record some with collect.py.")

    cv, cv_groups = choose_cv(y, groups)
    models = make_models()
    if args.model:
        models = {args.model: models[args.model]}

    # ---- compare models
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in models.items():
            t = time.time()
            s = cross_validate(model, X, y, cv=cv, groups=cv_groups,
                               scoring=["accuracy", "f1_macro"], n_jobs=1)
            results.append({"model": name,
                            "f1_macro": s["test_f1_macro"].mean(),
                            "f1_std": s["test_f1_macro"].std(),
                            "accuracy": s["test_accuracy"].mean(),
                            "seconds": time.time() - t})
    res = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    print("\nModel comparison (cross-validation):")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best = res.iloc[0]["model"]
    model = make_models()[best]
    print(f"\nBest model: {best}")

    # ---- detailed report for the best model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(model, X, y, cv=cv, groups=cv_groups)
    print("\nClassification report (out-of-fold predictions):")
    print(classification_report(y, pred, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                      index=[f"true:{l}" for l in labels], columns=labels)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm.to_string())

    # ---- final fit on all data and save
    model.fit(X, y)
    # Predict one frame at a time in the app: extra threads only add overhead
    # (RandomForest with n_jobs=-1 is ~4x slower for a single sample).
    for key in model.get_params():
        if key.endswith("n_jobs"):
            model.set_params(**{key: 1})
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "labels": [str(c) for c in model.classes_],
        "feature_names": FEATURE_NAMES,
        "feature_dim": FEATURE_DIM,
        "model_name": best,
        "cv_results": res.to_dict(orient="records"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(y)),
    }
    joblib.dump(bundle, config.CLASSIFIER_PATH)
    print(f"\nSaved -> {config.CLASSIFIER_PATH}")
    print("Next step: python app.py")


if __name__ == "__main__":
    main()
