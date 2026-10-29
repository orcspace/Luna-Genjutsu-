"""Faces: reference photo checks and crops, face-mesh tracking for the guide, identity scoring.

Detection and landmarks come from MediaPipe Face Landmarker (478 points). Identity uses
OpenCV's SFace, aligned from the same landmarks, so no separate face detector is needed.
"""
import hashlib
import json
import math
import os
import platform
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests

from . import media
from .config import MODELS

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
SFACE = {
    "file": "face_recognition_sface_2021dec.onnx",
    "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    "urls": [
        "https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ],
}
SAME_PERSON = 0.363  # SFace cosine threshold for "same person" (OpenCV)

NOSE, MOUTH_L, MOUTH_R = 1, 61, 291
IRIS = (468, 473)
EYE_CORNERS = ((33, 133), (362, 263))


class FaceError(RuntimeError):
    pass


def available():
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def _download(urls, dst, sha256=None):
    MODELS.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            if sha256 and hashlib.sha256(r.content).hexdigest() != sha256:
                errors.append(f"{url}: контрольная сумма не совпала")
                continue
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dst)
            return dst
        except requests.RequestException as exc:
            errors.append(f"{url}: {type(exc).__name__}")
    raise FaceError("Не удалось скачать модель: " + "; ".join(errors))


def _landmarker_model():
    path = MODELS / "face_landmarker.task"
    return path if path.exists() else _download([LANDMARKER_URL], path)


def sface_model():
    path = MODELS / SFACE["file"]
    return path if path.exists() else _download(SFACE["urls"], path, SFACE["sha256"])


def _landmarker(video=False, num_faces=1, blendshapes=False):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=_model_bytes(_landmarker_model())),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=num_faces,
        output_face_blendshapes=blendshapes,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    try:
        return vision.FaceLandmarker.create_from_options(options)
    except OSError as exc:
        if "EGL" in str(exc) or "GLES" in str(exc):
            raise FaceError("MediaPipe нужна системная библиотека OpenGL: sudo apt install libegl1 libgles2")
        raise


def _model_bytes(path):
    from .masking import model_bytes

    return model_bytes(path)


def _pose_landmarker(num_poses=3, segmentation=False):
    from mediapipe.tasks.python import BaseOptions, vision

    from .masking import _pose_model

    options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=_model_bytes(_pose_model("heavy"))),
                                           num_poses=num_poses, output_segmentation_masks=segmentation)
    return vision.PoseLandmarker.create_from_options(options)


def _mp_image(rgb):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def read_rgb(path):
    bgr = media.read_image(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FaceError("Не удалось открыть фото.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _faces_from(result, w, h):
    faces = []
    for k, lm in enumerate(result.face_landmarks or []):
        pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        blend = {}
        if result.face_blendshapes and k < len(result.face_blendshapes):
            blend = {c.category_name: c.score for c in result.face_blendshapes[k]}
        faces.append({"pts": pts, "box": (float(x0), float(y0), float(x1 - x0), float(y1 - y0)), "blend": blend})
    return sorted(faces, key=_area, reverse=True)


def _area(face):
    return face["box"][2] * face["box"][3]


def _detect_in(lm, rgb, roi, size=512):
    """Run an image-mode landmarker on a square crop, upscaled, and map the points back.
    The landmarker's own detector only finds faces that fill a good part of the frame."""
    x, y, side = roi
    h, w = rgb.shape[:2]
    x0, y0 = int(max(0, x)), int(max(0, y))
    x1, y1 = int(min(w, x + side)), int(min(h, y + side))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return []
    scale = size / max(crop.shape[:2])
    big = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    faces = _faces_from(lm.detect(_mp_image(big)), big.shape[1], big.shape[0])
    for f in faces:
        f["pts"] = f["pts"] / scale + np.array([x0, y0], np.float32)
        (fx0, fy0), (fx1, fy1) = f["pts"].min(axis=0), f["pts"].max(axis=0)
        f["box"] = (float(fx0), float(fy0), float(fx1 - fx0), float(fy1 - fy0))
    return faces


def head_roi_from_pose(pose, width, height):
    """Square around the head from pose landmarks 0–10 (nose, eyes, ears, mouth), normalised [x, y, vis]."""
    pts = np.array([[x * width, y * height] for x, y, v in pose[:11] if v > 0.3], np.float32)
    if len(pts) < 2:
        return None
    cx, cy = pts.mean(axis=0)
    side = max(float(np.ptp(pts, axis=0).max()) * 2.6, 48.0)
    return (cx - side / 2, cy - side * 0.55, side)


def head_roi_from_mask(mask):
    """Top of the silhouette: where the head is when no pose is known."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    top = ys.min()
    side = max(0.24 * (ys.max() - top + 1), 48.0)
    band = ys < top + side
    return (float(np.median(xs[band])) - side * 0.6, top - side * 0.1, side * 1.2)


def _pose_head_rois(rgb, pose_lm=None):
    h, w = rgb.shape[:2]
    if pose_lm is None:
        with _pose_landmarker(3) as lm:
            return _pose_head_rois(rgb, lm)
    result = pose_lm.detect(_mp_image(rgb))
    rois = [head_roi_from_pose([[p.x, p.y, p.visibility or 0] for p in pose], w, h) for pose in result.pose_landmarks or []]
    return [r for r in rois if r]


def detect(rgb, num_faces=3, blendshapes=False, lm=None, pose_lm=None):
    """Faces, biggest first. Small faces (full-body shots) are found through the head from the pose.
    Pass open landmarkers when calling in a loop: loading the models costs more than running them."""
    if lm is None:
        with _landmarker(num_faces=num_faces, blendshapes=blendshapes) as own:
            return detect(rgb, num_faces, blendshapes, own, pose_lm)
    h, w = rgb.shape[:2]
    faces = _faces_from(lm.detect(_mp_image(rgb)), w, h)
    if faces:
        return faces
    for roi in _pose_head_rois(rgb, pose_lm):
        faces += _detect_in(lm, rgb, roi)[:1]
    return sorted(faces, key=_area, reverse=True)


# ---------- reference photos ----------

def _yaw(pts):
    """Rough head turn: nose offset from the eye midpoint, in eye distances (0 = frontal)."""
    a, b = _eye_centres(pts)
    dist = max(1.0, float(np.hypot(*(b - a))))
    return float((pts[NOSE][0] - (a[0] + b[0]) / 2) / dist)


def analyze(path, role="face"):
    """What a reference photo is worth, with plain-language warnings. role: face | body | extra."""
    rgb = read_rgb(path)
    h, w = rgb.shape[:2]
    faces = detect(rgb, num_faces=3, blendshapes=True)
    report = {"faces": len(faces), "width": w, "height": h, "warnings": []}
    if not faces:
        report["warnings"].append("Лицо не найдено — для фото лица это обязательно; для фото в полный рост лицо тоже должно быть видно.")
        report["ok"] = False
        return report
    face = faces[0]
    x, y, fw, fh = face["box"]
    crop = cv2.cvtColor(rgb[max(0, int(y)):int(y + fh), max(0, int(x)):int(x + fw)], cv2.COLOR_RGB2GRAY)
    if crop.shape[0] > 256:  # compare sharpness at one scale; never upscale (that fakes blur)
        crop = cv2.resize(crop, (max(1, int(256 * crop.shape[1] / crop.shape[0])), 256), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    bright = float(crop.mean())
    yaw = _yaw(face["pts"])
    blink = min(face["blend"].get("eyeBlinkLeft", 0), face["blend"].get("eyeBlinkRight", 0))
    report.update(box=face["box"], face_ratio=round(fh / h, 3), face_px=int(fh), sharpness=round(sharp, 1), brightness=round(bright, 1), yaw=round(yaw, 2))
    if len(faces) > 1:
        report["warnings"].append("На фото несколько лиц — беру самое крупное. Лучше фото, где только вы.")
    if role == "face" and fh < 200:
        report["warnings"].append(f"Лицо мелкое ({int(fh)} px). Для точного сходства нужно от 300 px — снимите ближе или загрузите оригинал без сжатия.")
    elif fh < 48:
        report["warnings"].append(f"Лица почти не видно ({int(fh)} px) — добавьте отдельное фото лица крупно.")
    if sharp < 40:
        report["warnings"].append("Лицо размыто — сходство пострадает. Возьмите резкое фото.")
    if bright < 55:
        report["warnings"].append("Лицо слишком тёмное — снимите при ровном свете.")
    elif bright > 215:
        report["warnings"].append("Лицо пересвечено — снимите без вспышки и прямого солнца.")
    if abs(yaw) > 0.45:
        report["warnings"].append("Голова сильно повёрнута — для основного фото лучше анфас (профиль можно добавить как доп. ракурс).")
    if blink > 0.55:
        report["warnings"].append("Глаза закрыты — возьмите фото с открытыми глазами.")
    report["ok"] = not report["warnings"]
    return report


def face_closeup(path, dst, report=None):
    """Tight head-and-shoulders crop. None if the face is too small to be worth it."""
    report = report or analyze(path)
    if not report.get("faces") or report.get("face_px", 0) < 200:
        return None
    rgb = read_rgb(path)
    h, w = rgb.shape[:2]
    x, y, fw, fh = report["box"]
    cx, cy = x + fw / 2, y + fh * 0.45
    ch = min(h, fh * 2.0)
    cw = min(w, ch * 0.8)
    x0 = int(min(max(0, cx - cw / 2), w - cw))
    y0 = int(min(max(0, cy - ch / 2), h - ch))
    out = rgb[y0:y0 + int(ch), x0:x0 + int(cw)]
    media.write_image(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst


def clean_background(path, dst, background=(236, 236, 236)):
    """Person on a plain light background, cropped to fill the frame. None if segmentation is unsure."""
    rgb = read_rgb(path)
    h, w = rgb.shape[:2]
    with _pose_landmarker(2, segmentation=True) as lm:
        result = lm.detect(_mp_image(rgb))
    masks = [np.squeeze(m.numpy_view()).astype(np.float32) for m in (result.segmentation_masks or [])]
    if not masks:
        return None
    soft = max(masks, key=lambda m: (m > 0.5).sum())
    hard = soft > 0.5
    share = hard.mean()
    if not 0.04 < share < 0.85:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), 8)
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((labels == main).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    alpha = np.where(keep, soft, 0)
    blur = max(1, int(min(h, w) * 0.004)) * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)[..., None]
    out = (rgb * alpha + np.array(background, np.float32) * (1 - alpha)).astype(np.uint8)
    ys, xs = np.where(keep)
    m = int(0.06 * max(np.ptp(ys), np.ptp(xs)))
    out = out[max(0, ys.min() - m):min(h, ys.max() + m), max(0, xs.min() - m):min(w, xs.max() + m)]
    media.write_image(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst


# ---------- face mesh for the guide video ----------

@lru_cache(maxsize=1)
def mesh_lines():
    """Connection pairs per facial feature, from MediaPipe's own definitions."""
    from mediapipe.tasks.python import vision

    c = vision.FaceLandmarksConnections
    groups = {
        "oval": c.FACE_LANDMARKS_FACE_OVAL,
        "brows": c.FACE_LANDMARKS_LEFT_EYEBROW + c.FACE_LANDMARKS_RIGHT_EYEBROW,
        "eyes": c.FACE_LANDMARKS_LEFT_EYE + c.FACE_LANDMARKS_RIGHT_EYE,
        "irises": c.FACE_LANDMARKS_LEFT_IRIS + c.FACE_LANDMARKS_RIGHT_IRIS,
        "lips": c.FACE_LANDMARKS_LIPS,
    }
    return {k: [(p.start, p.end) for p in v] for k, v in groups.items()}


MESH_COLOURS = {"oval": (235, 235, 235), "brows": (255, 255, 255), "eyes": (255, 255, 255), "irises": (120, 220, 255), "lips": (255, 110, 170)}


class OneEuro:
    """One Euro filter: steady when still, responsive when moving (Casiez et al.)."""

    def __init__(self, freq=media.FPS, min_cutoff=1.0, beta=1.0, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    def _alpha(self, cutoff):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau * self.freq)

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx = x, np.zeros_like(x)
            return x
        dx = (x - self.x) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff)
        self.x = a * x + (1 - a) * self.x
        return self.x

    def reset(self):
        self.x = self.dx = None


def smooth_track(frames, min_cutoff, beta):
    """Filter a per-frame list of point arrays (or None); a gap restarts the filter.
    Columns after x, y (e.g. visibility) pass through unchanged."""
    out, filt = [], OneEuro(min_cutoff=min_cutoff, beta=beta)
    for pts in frames:
        if pts is None:
            filt.reset()
            out.append(None)
            continue
        arr = np.asarray(pts, np.float32)
        xy = filt(arr[:, :2])
        out.append(np.concatenate([xy, arr[:, 2:]], axis=1).astype(np.float64).round(4).tolist())
    return out


def track(video, mask_dir, width, height, frames, status):
    """Face landmarks of the hidden person on every frame → mask_dir/face.json.

    The face whose nose lies inside that frame's mask is the one we follow, so other people
    in the shot are ignored.
    """
    status("MediaPipe · сетка лица для мимики и липсинка")
    from .masking import _load_mask

    files = sorted(Path(mask_dir).glob("mask_*.png"))
    pose_file = Path(mask_dir) / "pose.json"
    poses = json.loads(pose_file.read_text()) if pose_file.exists() else []
    used = sorted({i for pairs in mesh_lines().values() for pair in pairs for i in pair})
    track_pts = []

    def inside(face, mask):
        nx, ny = face["pts"][NOSE]
        return 0 <= int(ny) < height and 0 <= int(nx) < width and mask[int(ny), int(nx)]

    with _landmarker(video=True, num_faces=4) as whole, _landmarker(num_faces=1) as close:
        for i, rgb in enumerate(media.read_frames(video, width, height)):
            mask = _load_mask(mask_dir, files, i, frames, width, height)
            pick = None
            if mask.any():
                faces = _faces_from(whole.detect_for_video(_mp_image(rgb), int(i * 1000 / media.FPS)), width, height)
                pick = next((f for f in faces if inside(f, mask)), None)
                if pick is None:
                    # Small or distant face: look again on an enlarged crop around the head.
                    roi = head_roi_from_pose(poses[i], width, height) if i < len(poses) and poses[i] else head_roi_from_mask(mask)
                    if roi:
                        pick = next((f for f in _detect_in(close, rgb, roi) if inside(f, cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool))), None)
            track_pts.append(None if pick is None else (pick["pts"][used] / [width, height]).tolist())
    smoothed = smooth_track(track_pts, min_cutoff=2.0, beta=8.0)
    (Path(mask_dir) / "face.json").write_text(json.dumps({"indices": used, "frames": smoothed}))
    found = sum(p is not None for p in smoothed)
    return found / max(1, len(smoothed))


def draw_mesh(frame, face, indices):
    if not face:
        return frame
    h, w = frame.shape[:2]
    lookup = {idx: (int(x * w), int(y * h)) for idx, (x, y) in zip(indices, face)}
    xs = [p[0] for p in lookup.values()]
    # Lines scale with the face, so a small face keeps an open mouth readable.
    thick = max(1, round((max(xs) - min(xs)) * 0.012))
    for group, pairs in mesh_lines().items():
        colour = MESH_COLOURS[group]
        t = thick
        for a, b in pairs:
            if a in lookup and b in lookup:
                cv2.line(frame, lookup[a], lookup[b], colour, t, cv2.LINE_AA)
    return frame


# ---------- identity ----------

def _eye_centres(pts):
    if len(pts) > max(IRIS):
        a, b = pts[IRIS[0]], pts[IRIS[1]]
    else:
        a = (pts[EYE_CORNERS[0][0]] + pts[EYE_CORNERS[0][1]]) / 2
        b = (pts[EYE_CORNERS[1][0]] + pts[EYE_CORNERS[1][1]]) / 2
    return (a, b) if a[0] <= b[0] else (b, a)


def sface_row(pts):
    """The 15-value face row (box + 5 points) SFace's alignCrop expects, built from mesh landmarks.
    Points go left-to-right in the image, matching SFace's alignment template."""
    pts = np.asarray(pts, np.float32)
    e1, e2 = _eye_centres(pts)
    m1, m2 = sorted([pts[MOUTH_L], pts[MOUTH_R]], key=lambda p: p[0])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return np.array([[x0, y0, x1 - x0, y1 - y0, *e1, *e2, *pts[NOSE], *m1, *m2, 1.0]], np.float32)


def _ascii_path(path):
    """OpenCV's model loader can't open non-ASCII paths on Windows: keep a copy in the ASCII Public folder."""
    if str(path).isascii() or platform.system() != "Windows":
        return path
    target = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "LunaGenjutsu" / path.name
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return target


class Identity:
    def __init__(self):
        self.rec = cv2.FaceRecognizerSF.create(str(_ascii_path(sface_model())), "")

    def embed(self, rgb, pts):
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        return self.rec.feature(self.rec.alignCrop(bgr, sface_row(pts))).copy()

    def cosine(self, a, b):
        return float(self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def score_value(similarity):
    """SFace cosine → 0–100 for people: ~45 at the same-person threshold, 100 at a very strong match."""
    return int(round(100 * min(1.0, max(0.0, (similarity - 0.15) / (0.62 - 0.15)))))


def score_video(video, reference_photos, samples=16, identity=None):
    """How much the person in the video looks like the reference photos."""
    identity = identity or Identity()
    info = media.probe(video)
    w, h = info["width"], info["height"]
    per_frame = []
    with _landmarker(num_faces=4) as lm, _pose_landmarker(3) as pose_lm:
        refs = []
        for p in reference_photos:
            rgb = read_rgb(p)
            found = detect(rgb, lm=lm, pose_lm=pose_lm)
            if found:
                refs.append(identity.embed(rgb, found[0]["pts"]))
        if not refs:
            raise FaceError("На фото не найдено лицо для сравнения.")
        # Only the sampled frames leave ffmpeg; the models stay loaded for the whole pass.
        rate = samples / max(info["duration"], 0.5)
        for rgb in media.read_frames(video, w, h, fps=f"{rate:.4f}"):
            found = detect(rgb, lm=lm, pose_lm=pose_lm)
            per_frame.append(max((max(identity.cosine(identity.embed(rgb, f["pts"]), r) for r in refs) for f in found), default=None))
    found = [s for s in per_frame if s is not None]
    if not found:
        return {"score": 0, "similarity": None, "coverage": 0.0, "matched": 0.0, "frames": len(per_frame)}
    similarity = float(np.median(found))
    return {
        "score": score_value(similarity),
        "similarity": round(similarity, 3),
        "worst": round(float(np.percentile(found, 20)), 3),
        "coverage": round(len(found) / len(per_frame), 2),
        "matched": round(sum(s >= SAME_PERSON for s in found) / len(per_frame), 2),
        "frames": len(per_frame),
    }


def score_still(image, reference_photos, identity=None):
    """Likeness of the best-matching face in one generated image to the reference photos."""
    identity = identity or Identity()
    with _landmarker(num_faces=4) as lm, _pose_landmarker(3) as pose_lm:
        refs = []
        for p in reference_photos:
            rgb = read_rgb(p)
            found = detect(rgb, lm=lm, pose_lm=pose_lm)
            if found:
                refs.append(identity.embed(rgb, found[0]["pts"]))
        if not refs:
            raise FaceError("На фото не найдено лицо для сравнения.")
        rgb = read_rgb(image)
        found = detect(rgb, lm=lm, pose_lm=pose_lm)
    best = max((max(identity.cosine(identity.embed(rgb, f["pts"]), r) for r in refs) for f in found), default=None)
    if best is None:
        return {"score": 0, "similarity": None, "face": False}
    return {"score": score_value(best), "similarity": round(best, 3), "face": True}
