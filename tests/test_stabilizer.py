import numpy as np

from meme_cam.stabilizer import MatchStabilizer

LABELS = ["_none", "hamster", "shocked"]


def p(none, hamster, shocked):
    return np.array([none, hamster, shocked])


def make(**kw):
    return MatchStabilizer(LABELS, threshold=0.7, window=1, hold_frames=3,
                           release_margin=0.1, **kw)


def test_needs_hold_frames_before_showing():
    s = make()
    assert s.update(p(0.1, 0.9, 0.0)) is None
    assert s.update(p(0.1, 0.9, 0.0)) is None
    assert s.update(p(0.1, 0.9, 0.0)) == "hamster"


def test_below_threshold_never_shows():
    s = make()
    for _ in range(10):
        assert s.update(p(0.4, 0.6, 0.0)) is None


def test_none_label_never_shows():
    s = make()
    for _ in range(10):
        assert s.update(p(0.95, 0.05, 0.0)) is None


def test_hysteresis_keeps_meme_then_releases():
    s = make()
    for _ in range(3):
        s.update(p(0.1, 0.9, 0.0))
    assert s.update(p(0.35, 0.65, 0.0)) == "hamster"   # 0.65 >= 0.7-0.1
    assert s.update(p(0.5, 0.5, 0.0)) is None          # dropped below 0.6


def test_switching_meme_needs_hold_again():
    s = make()
    for _ in range(3):
        s.update(p(0.1, 0.9, 0.0))
    assert s.update(p(0.0, 0.1, 0.9)) is None
    assert s.update(p(0.0, 0.1, 0.9)) is None
    assert s.update(p(0.0, 0.1, 0.9)) == "shocked"


def test_no_person_resets():
    s = make()
    for _ in range(3):
        s.update(p(0.1, 0.9, 0.0))
    assert s.update(None) is None
    assert s.update(p(0.1, 0.9, 0.0)) is None


def test_smoothing_window_filters_single_spike():
    s = MatchStabilizer(LABELS, threshold=0.7, window=5, hold_frames=1)
    for _ in range(4):
        s.update(p(0.9, 0.1, 0.0))
    assert s.update(p(0.0, 1.0, 0.0)) is None  # one spike is averaged away
