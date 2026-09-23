from meme_cam import camera
from meme_cam.camera import Layout


def test_layout_keeps_camera_aspect(monkeypatch):
    monkeypatch.setattr(camera, "screen_size", lambda: None)
    for w, h in [(1280, 720), (640, 480), (1920, 1080)]:
        lay = Layout(w, h, height=540)
        assert lay.height == 540
        assert abs(lay.cam_w / lay.height - w / h) < 0.01
        assert lay.width == lay.cam_w + Layout.GAP + lay.panel_w


def test_layout_fits_small_screen(monkeypatch):
    monkeypatch.setattr(camera, "screen_size", lambda: (1366, 768))
    lay = Layout(1280, 720, height=540)
    assert lay.width <= 1366 and lay.height <= 768
    assert abs(lay.cam_w / lay.height - 16 / 9) < 0.01
