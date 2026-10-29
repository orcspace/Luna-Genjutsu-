import numpy as np

from genjutsu import faces


def test_sface_row_orders_points_left_to_right():
    pts = np.zeros((478, 2), np.float32)
    pts[:, 0] = np.linspace(100, 200, 478)
    pts[:, 1] = 150
    pts[468] = (170, 120)  # one iris on the right of the image
    pts[473] = (130, 121)  # the other on the left
    pts[faces.NOSE] = (150, 140)
    pts[faces.MOUTH_L] = (165, 170)
    pts[faces.MOUTH_R] = (135, 171)
    row = faces.sface_row(pts)
    assert row.shape == (1, 15)
    x, y, w, h, e1x, e1y, e2x, e2y, nx, ny, m1x, m1y, m2x, m2y, score = row[0]
    assert e1x < e2x and m1x < m2x
    assert (e1x, e2x, m1x, m2x) == (130, 170, 135, 165)
    assert w > 0 and h > 0 and score == 1.0


def test_one_euro_removes_jitter_but_follows_motion():
    rng = np.random.default_rng(0)
    still = [[[0.5 + rng.normal(0, 0.004), 0.5, 0.9]] for _ in range(48)]  # MediaPipe-sized jitter
    out = faces.smooth_track(still, min_cutoff=1.2, beta=4.0)
    assert np.std([p[0][0] for p in out[10:]]) < 0.5 * np.std([p[0][0] for p in still[10:]])
    assert out[5][0][2] == 0.9  # visibility passes through
    moving = [[[i / 48, 0.5, 1.0]] for i in range(48)]
    out = faces.smooth_track(moving, min_cutoff=1.2, beta=4.0)
    assert abs(out[-1][0][0] - moving[-1][0][0]) < 0.05


def test_gaps_reset_the_filter():
    out = faces.smooth_track([[[0.1, 0.1]], None, [[0.9, 0.9]]], min_cutoff=1.0, beta=0.0)
    assert out[1] is None and out[2] == [[0.9, 0.9]]


def test_score_scale():
    assert faces.score_value(0.1) == 0
    assert 40 <= faces.score_value(faces.SAME_PERSON) <= 50
    assert faces.score_value(0.9) == 100


def test_head_rois():
    pose = [[0.5, 0.2, 0.9], [0.48, 0.18, 0.9], [0.52, 0.18, 0.9]] + [[0.5, 0.5, 0.0]] * 30
    x, y, side = faces.head_roi_from_pose(pose, 1000, 1000)
    assert x < 500 < x + side and y < 200 < y + side
    mask = np.zeros((200, 100), bool)
    mask[40:190, 30:70] = True
    x, y, side = faces.head_roi_from_mask(mask)
    assert x < 50 < x + side and y <= 40
