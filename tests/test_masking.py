import numpy as np
import pytest

from genjutsu import masking, media
from conftest import box_masks


def test_compose_hides_the_person_and_keeps_background(make_video, tmp_path):
    src = make_video(seconds=2, audio=False)
    meta = media.normalize_video(src, tmp_path / "original.mp4")
    meta["frames"] = round(meta["duration"] * media.FPS)
    w, h, n = meta["width"], meta["height"], meta["frames"]
    mask_dir = box_masks(tmp_path, n, w, h)
    qa = masking.compose(tmp_path / "original.mp4", mask_dir, tmp_path, meta, status=lambda m: None)
    assert qa["ok"], qa
    assert (tmp_path / "preview.jpg").exists()

    original = list(media.read_frames(tmp_path / "original.mp4", w, h))
    guide = list(media.read_frames(tmp_path / "guidance-silent.mp4", w, h))
    assert len(guide) == len(original) == n
    i = n // 2
    x = int(i / (n - 1) * (w - w // 4))
    inner = (slice(h // 4 + 8, h - h // 8 - 8), slice(x + 8, x + w // 4 - 8))
    # Inside the box: silhouette, clearly different from the source picture.
    assert np.abs(guide[i][inner].astype(int) - original[i][inner]).mean() > 30
    # Far corner outside the box: the original background (only compression noise).
    corner = (slice(0, h // 8), slice(w - w // 10 if x < w // 2 else 0, w if x < w // 2 else w // 10))
    assert np.abs(guide[i][corner].astype(int) - original[i][corner]).mean() < 8


def test_temporal_vote_fills_single_dropped_frame(tmp_path):
    import cv2
    w, h, n = 160, 120, 20
    mask_dir = box_masks(tmp_path, n, w, h)
    cv2.imwrite(str(mask_dir / "mask_00010.png"), np.zeros((h, w), np.uint8))
    masks = list(masking.refined_masks(mask_dir, n, w, h))
    assert len(masks) == n
    # Neighbours vote the person back into the frame the segmenter missed.
    assert masks[10].mean() > 0.5 * masks[9].mean()


def test_mask_count_mismatch_is_resampled(tmp_path):
    mask_dir = box_masks(tmp_path, 10, 160, 120)
    assert len(list(masking.refined_masks(mask_dir, 24, 160, 120))) == 24


def test_quality_flags_problems():
    assert masking.quality([0.1] * 100)["ok"]
    bad = masking.quality([0.1] * 80 + [0.0] * 20)
    assert not bad["ok"] and bad["empty_frames"] == 20
    with pytest.raises(masking.MaskError):
        masking.quality([0.0] * 10)


def _blob(y0, y1, x0, x1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_follow_prefers_overlap_and_flags_jumps():
    top, bottom = _blob(5, 45, 30, 70), _blob(55, 95, 30, 70)
    assert masking._follow([bottom, top], top) == (1, True)
    # Only someone else is visible: best guess, but not sure (caller holds the old mask).
    assert masking._follow([bottom], top) == (0, False)
    # First frame: the biggest person.
    assert masking._follow([_blob(0, 10, 0, 10), bottom], None) == (1, True)


def test_keep_body_drops_bleed_without_landmarks():
    body, bleed = _blob(10, 60, 30, 70), _blob(80, 90, 40, 50)
    pose = [[0.5, 0.3, 0.9], [0.5, 0.5, 0.9], [0.45, 0.85, 0.1]]  # the low-visibility point lies in the bleed
    kept = masking._keep_body(body | bleed, pose)
    assert kept[30, 50] and not kept[85, 45]


def test_draw_pose_marks_limbs_inside_frame():
    frame = np.zeros((100, 100, 3), np.uint8)
    pose = [[0.5, 0.5, 0.0]] * 33
    pose[12], pose[14] = [0.3, 0.3, 0.9], [0.3, 0.7, 0.9]  # right upper arm
    out = masking.draw_pose(frame.copy(), pose)
    assert tuple(out[50, 30]) == (255, 60, 60)


def test_pseudo_depth_only_colours_the_silhouette():
    m = np.zeros((50, 50), bool)
    m[10:40, 10:40] = True
    d = masking.pseudo_depth(m)
    assert d.shape == (50, 50, 3) and d[25, 25].sum() > d[12, 12].sum()
