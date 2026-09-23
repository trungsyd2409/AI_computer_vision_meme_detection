import math

import numpy as np

from meme_cam.features import (FEATURE_DIM, FEATURE_NAMES, HAND_BLOCK, build_features,
                               hand_block, head_pose_from_matrix)


def _fake_hand(offset=(0.0, 0.0), scale=1.0, open_hand=True):
    """21 points: wrist at origin, fingers pointing up (y negative)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    bases = {1: -0.3, 5: -0.15, 9: 0.0, 13: 0.15, 17: 0.3}
    for base, x in bases.items():
        for j in range(4):
            length = (j + 1) * 0.1 if (open_hand or base == 1) else 0.05
            pts[base + j] = [x, -0.9 - length if base != 1 else -0.3 - j * 0.1, 0]
    pts[9] = [0.0, -1.0, 0]
    pts[:, :2] = pts[:, :2] * scale + np.array(offset)
    return pts


def test_dims():
    assert FEATURE_DIM == len(FEATURE_NAMES) == 1 + 52 + 3 + 2 * HAND_BLOCK


def test_empty_frame_is_all_zero():
    v = build_features(None, None, None, {})
    assert v.shape == (FEATURE_DIM,) and not v.any()


def test_hand_shape_is_scale_and_position_invariant():
    a = hand_block(_fake_hand(), None, None)
    b = hand_block(_fake_hand(offset=(3, 5), scale=2.5), None, None)
    np.testing.assert_allclose(a[:69], b[:69], atol=1e-5)


def test_open_vs_closed_fingers_differ():
    open_ext = hand_block(_fake_hand(open_hand=True), None, None)[64:69]
    closed_ext = hand_block(_fake_hand(open_hand=False), None, None)[64:69]
    assert (open_ext[1:] > closed_ext[1:]).all()


def test_hand_position_relative_to_face():
    face = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32)  # centre (0.5,0.5), width 1
    hand = _fake_hand(offset=(2, 0.5))
    v = build_features(face, {"jawOpen": 0.8}, np.eye(4), {"Right": hand})
    assert v[FEATURE_NAMES.index("face_present")] == 1
    assert math.isclose(v[FEATURE_NAMES.index("bs_jawOpen")], 0.8, rel_tol=1e-6)
    assert v[FEATURE_NAMES.index("handRight_present")] == 1
    assert v[FEATURE_NAMES.index("handLeft_present")] == 0
    assert v[FEATURE_NAMES.index("handRight_pos_x")] > 1.0  # hand is right of the face


def test_head_pose_identity_and_yaw():
    assert np.allclose(head_pose_from_matrix(np.eye(4)), 0, atol=1e-6)
    a = math.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    _, yaw, _ = head_pose_from_matrix(m)
    assert math.isclose(yaw, a, abs_tol=1e-6)
