import numpy as np

from meme_cam.stabilizer import MatchStabilizer

LABELS = ["_none", "hamster", "shocked"]


def p(none, hamster, shocked):
    return np.array([none, hamster, shocked])


def make(**kw):
    opts = dict(threshold=0.7, alpha=1.0, hold_frames=2, release_frames=3, release_margin=0.1)
    opts.update(kw)
    return MatchStabilizer(LABELS, **opts)


def test_shows_after_hold_frames():
    s = make()
    assert s.update(p(0.1, 0.9, 0.0)) is None
    assert s.update(p(0.1, 0.9, 0.0)) == "hamster"


def test_hold_one_is_instant():
    s = make(hold_frames=1)
    assert s.update(p(0.1, 0.9, 0.0)) == "hamster"


def test_below_threshold_never_shows():
    s = make()
    for _ in range(10):
        assert s.update(p(0.4, 0.6, 0.0)) is None


def test_none_label_never_shows():
    s = make()
    for _ in range(10):
        assert s.update(p(0.95, 0.05, 0.0)) is None


def test_single_bad_frame_does_not_hide():
    s = make()
    s.update(p(0.1, 0.9, 0.0)); s.update(p(0.1, 0.9, 0.0))
    assert s.update(p(0.9, 0.1, 0.0)) == "hamster"     # 1 bad frame
    assert s.update(p(0.1, 0.9, 0.0)) == "hamster"


def test_hides_after_release_frames():
    s = make()
    s.update(p(0.1, 0.9, 0.0)); s.update(p(0.1, 0.9, 0.0))
    assert s.update(p(0.9, 0.1, 0.0)) == "hamster"
    assert s.update(p(0.9, 0.1, 0.0)) == "hamster"
    assert s.update(p(0.9, 0.1, 0.0)) is None


def test_hysteresis_margin_keeps_meme():
    s = make()
    s.update(p(0.1, 0.9, 0.0)); s.update(p(0.1, 0.9, 0.0))
    for _ in range(5):
        assert s.update(p(0.35, 0.65, 0.0)) == "hamster"   # 0.65 >= 0.7 - 0.1


def test_switch_to_other_meme_fast():
    s = make()
    s.update(p(0.1, 0.9, 0.0)); s.update(p(0.1, 0.9, 0.0))
    assert s.update(p(0.0, 0.1, 0.9)) == "hamster"
    assert s.update(p(0.0, 0.1, 0.9)) == "shocked"


def test_nobody_visible_hides_after_release_frames():
    s = make()
    s.update(p(0.1, 0.9, 0.0)); s.update(p(0.1, 0.9, 0.0))
    assert s.update(None) == "hamster"      # short detection drop-out is ignored
    assert s.update(None) == "hamster"
    assert s.update(None) is None


def test_ema_latency_default_settings():
    """With default alpha=0.6, hold=2: a clear pose shows on the 3rd frame."""
    s = make(alpha=0.6)
    s.update(p(1.0, 0.0, 0.0))                     # neutral before
    out = [s.update(p(0.0, 1.0, 0.0)) for _ in range(4)]
    assert out.index("hamster") <= 2


def test_ema_filters_single_spike():
    s = make(alpha=0.6, hold_frames=2)
    for _ in range(5):
        s.update(p(1.0, 0.0, 0.0))
    assert s.update(p(0.0, 1.0, 0.0)) is None
    assert s.update(p(1.0, 0.0, 0.0)) is None
