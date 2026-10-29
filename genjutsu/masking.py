"""Hide the person in the source clip so the generator only sees their motion.

Each frame keeps the original background; the person becomes a coloured depth silhouette.
Masks are stored as one PNG per frame (mask_00000.png …) whichever segmenter made them.
"""
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import media, replicate
from .config import MODELS

SAM3_VERSION = "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269"  # lucataco/sam3-video
DEPTH_MODEL = "lucataco/depth-anything-video"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

EMPTY_FRAME = 0.002  # mask area below this share of the frame counts as "nobody found"
HOLD_FRAMES = 12  # half a second at 24 fps


class MaskError(RuntimeError):
    pass


# ---------- segmenters ----------

def segment_cloud(video, folder, who, status):
    """SAM 3 video masks on Replicate: text prompt → one PNG per frame."""
    folder = Path(folder)
    status("SAM 3 · ищу «%s» на каждом кадре" % who)
    url = replicate.upload(video)
    output = replicate.run(SAM3_VERSION, {"video": url, "prompt": who, "mask_only": True, "return_zip": True}, folder / "sam3.json")
    zipped = replicate.download(output if isinstance(output, str) else output[0], folder / "sam3-masks.zip")
    out = folder / "masks"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zipped) as z:
        names = sorted(n for n in z.namelist() if Path(n).name.startswith("mask_") and n.endswith(".png"))
        if not names:
            raise MaskError("SAM 3 не вернул ни одной маски.")
        for i, name in enumerate(names):
            (out / f"mask_{i:05d}.png").write_bytes(z.read(name))
    return out


def depth_cloud(video, folder, status):
    """Coloured video depth on Replicate (input field resolved from the live schema)."""
    status("Video Depth Anything · строю карту глубины")
    version = replicate.latest_version(DEPTH_MODEL)
    props = version["openapi_schema"]["components"]["schemas"]["Input"]["properties"]
    field = next((k for k, v in props.items() if "video" in k and v.get("format") == "uri"), None)
    if not field:
        raise MaskError("Не нашёл поле video у модели глубины на Replicate.")
    output = replicate.run(version["id"], {field: replicate.upload(video)}, Path(folder) / "depth.json")
    if isinstance(output, dict):
        output = next(v for k, v in output.items() if "depth" in k or "video" in k)
    if isinstance(output, list):
        output = output[0]
    return replicate.download(output, Path(folder) / "depth-raw.mp4")


def _pose_model():
    path = MODELS / "pose_landmarker_heavy.task"
    if not path.exists():
        import requests

        MODELS.mkdir(parents=True, exist_ok=True)
        r = requests.get(POSE_MODEL_URL, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def segment_local(video, folder, width, height, status, max_people=4):
    """MediaPipe pose segmentation on this machine: free and nothing leaves the PC.

    Follows one person: the biggest on the first frame, then whoever stays closest to them.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        raise MaskError("Для локального режима установите mediapipe: pip install mediapipe")
    status("MediaPipe · выделяю человека локально")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_pose_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max_people,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    out = Path(folder) / "masks"
    out.mkdir(exist_ok=True)
    previous = None
    poses = []
    try:
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except OSError as exc:
        if "EGL" in str(exc) or "GLES" in str(exc):
            raise MaskError("MediaPipe нужна системная библиотека OpenGL: sudo apt install libegl1 libgles2")
        raise
    lost, last_pose = 0, None
    with landmarker:
        for i, frame in enumerate(media.read_frames(video, width, height)):
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)), int(i * 1000 / media.FPS))
            candidates = [np.squeeze(m.numpy_view()) > 0.5 for m in (result.segmentation_masks or [])]
            pick, sure = _follow(candidates, previous) if candidates else (None, False)
            if not sure and previous is not None and lost < HOLD_FRAMES:
                # The detector drops people for a frame or two; keep covering them meanwhile
                # instead of jumping to someone else or exposing the face.
                mask, pose, lost = previous, last_pose, lost + 1
            else:
                lost = 0
                mask = candidates[pick] if pick is not None else np.zeros((height, width), bool)
                pose = None
                if pick is not None and pick < len(result.pose_landmarks or []):
                    pose = [[round(p.x, 4), round(p.y, 4), round(p.visibility or 0, 2)] for p in result.pose_landmarks[pick]]
                    mask = _keep_body(mask, pose)
            poses.append(pose)
            if mask.any():
                previous, last_pose = mask, pose
            cv2.imwrite(str(out / f"mask_{i:05d}.png"), mask.astype(np.uint8) * 255)
    from .faces import smooth_track

    # Steadier joints → a steadier guide → less jitter in the generated body.
    (out / "pose.json").write_text(json.dumps(smooth_track(poses, min_cutoff=1.2, beta=4.0)))
    return out


def _keep_body(mask, pose):
    """Drop mask pieces that hold none of this person's visible landmarks (bleed onto neighbours)."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    h, w = mask.shape
    keep = {labels[min(h - 1, max(0, int(y * h))), min(w - 1, max(0, int(x * w)))] for x, y, v in pose if v > 0.5} - {0}
    if not keep:
        return mask
    return np.isin(labels, list(keep))


def _follow(candidates, previous):
    """(index, sure) of the candidate that continues the tracked person.

    Overlap first. Without overlap (person lost, or a cut) the nearest person of similar size
    is the best guess, flagged as not sure so the caller can hold the previous mask a little.
    """
    if previous is None:
        return max(range(len(candidates)), key=lambda k: candidates[k].sum()), True
    overlaps = [(m & previous).sum() / max(1, (m | previous).sum()) for m in candidates]
    best = max(range(len(candidates)), key=overlaps.__getitem__)
    if overlaps[best] >= 0.1:
        return best, True
    py, px = np.argwhere(previous).mean(axis=0)
    diag = np.hypot(*previous.shape)
    area = previous.sum()

    def cost(k):
        m = candidates[k]
        if not m.any():
            return float("inf")
        cy, cx = np.argwhere(m).mean(axis=0)
        return np.hypot(cy - py, cx - px) / diag + abs(np.log(max(1, m.sum()) / max(1, area))) * 0.25

    return min(range(len(candidates)), key=cost), False


# MediaPipe pose connections, grouped so left and right limbs get different colours (OpenPose style).
SKELETON = {
    (255, 60, 60): [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],      # right arm
    (60, 120, 255): [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],     # left arm
    (255, 170, 0): [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)],               # right leg
    (0, 220, 220): [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)],               # left leg
    (80, 255, 80): [(11, 12), (11, 23), (12, 24), (23, 24)],                         # torso
    (255, 255, 255): [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],  # face
}


def draw_pose(frame, pose, face_lines=True):
    """Limb lines inside the silhouette: the depth fill alone hides arms crossing the body.
    face_lines=False when a detailed face mesh is drawn instead."""
    if not pose:
        return frame
    h, w = frame.shape[:2]
    thick = max(2, int(min(w, h) * 0.011))
    pts = [(int(x * w), int(y * h), v) for x, y, v in pose]
    for colour, pairs in SKELETON.items():
        if colour == (255, 255, 255) and not face_lines:
            continue
        for a, b in pairs:
            if pts[a][2] > 0.5 and pts[b][2] > 0.5:
                cv2.line(frame, pts[a][:2], pts[b][:2], colour, thick if colour != (255, 255, 255) else max(1, thick // 2), cv2.LINE_AA)
    for k, (x, y, v) in enumerate(pts):
        if v > 0.5 and (face_lines or k > 10):
            cv2.circle(frame, (x, y), max(2, thick // 2 + 1), (255, 255, 255), -1, cv2.LINE_AA)
    return frame


# ---------- refinement and composite ----------

def _load_mask(mask_dir, files, i, n, width, height):
    if not files:
        return np.zeros((height, width), bool)
    j = min(len(files) - 1, round(i * len(files) / n)) if len(files) != n else i
    m = cv2.imread(str(files[j]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((height, width), bool)
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _clean(mask, width, height):
    """Drop slivers on the frame edge, close holes, and grow the silhouette a little.

    Growing hides hair and clothing edges, and gives the new body room to have a different outline.
    """
    m = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        on_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        if area < width * height * 0.0015 or (on_edge and area < width * height * 0.03 and min(w, h) <= max(6, int(min(width, height) * 0.05))):
            m[labels == k] = 0
    r = max(3, int(min(width, height) * 0.012))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m.astype(bool)


def refined_masks(mask_dir, n, width, height):
    """Cleaned masks with a 3-frame temporal majority vote against flicker."""
    files = sorted(Path(mask_dir).glob("mask_*.png"))
    raw = (_load_mask(mask_dir, files, i, n, width, height) for i in range(n))
    prev, cur = None, next(raw, None)
    if cur is None:
        return
    for nxt in raw:
        yield _clean(_vote(prev, cur, nxt), width, height)
        prev, cur = cur, nxt
    yield _clean(_vote(prev, cur, None), width, height)


def _vote(prev, cur, nxt):
    if prev is None or nxt is None:
        return cur
    return (prev.astype(np.uint8) + cur + nxt) >= 2


def pseudo_depth(mask):
    """Rounded relief from the distance to the silhouette edge, for runs without a depth model."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = np.sqrt(dist / dist.max())
    colored = cv2.applyColorMap((dist * 220 + 35).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def align_depth(depth_raw, dst, duration):
    """Stretch the depth clip onto the source timeline at 24 fps."""
    d = media.probe(depth_raw)["duration"] or duration
    media.run(["-i", depth_raw, "-an", "-vf", f"setpts=PTS*{duration / d:.6f},fps={media.FPS},tpad=stop_mode=clone:stop_duration=1",
               "-t", f"{duration:.3f}", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", dst], "выравнивание глубины")
    return dst


def compose(original, mask_dir, folder, meta, depth=None, status=print):
    """Write guidance-silent.mp4 (background + depth silhouette) and mask.mp4, return QA stats."""
    folder = Path(folder)
    w, h, n = meta["width"], meta["height"], meta["frames"]
    status("Собираю видео-гид: фон оригинала + силуэт вместо человека")
    depth_frames = media.read_frames(depth, w, h) if depth else None
    pose_file = Path(mask_dir) / "pose.json"
    poses = json.loads(pose_file.read_text()) if pose_file.exists() else []
    face_file = Path(mask_dir) / "face.json"
    mesh = json.loads(face_file.read_text()) if face_file.exists() else {"indices": [], "frames": []}
    samples_at = set(np.linspace(0, max(0, n - 1), 4).round().astype(int).tolist())
    samples, areas = [], []
    with media.FrameWriter(folder / "guidance-silent.mp4", w, h) as out, media.FrameWriter(folder / "mask.mp4", w, h, crf=20) as mask_out:
        for i, (rgb, mask) in enumerate(zip(media.read_frames(original, w, h), refined_masks(mask_dir, n, w, h))):
            d = next(depth_frames, None) if depth_frames is not None else None
            fill = d if d is not None else pseudo_depth(mask)
            frame = np.where(mask[..., None], fill, rgb)
            face = mesh["frames"][i] if i < len(mesh["frames"]) and mask.any() else None
            if i < len(poses) and mask.any():
                frame = draw_pose(np.ascontiguousarray(frame), poses[i], face_lines=face is None)
            if face:
                from .faces import draw_mesh

                frame = draw_mesh(np.ascontiguousarray(frame), face, mesh["indices"])
            out.write(frame)
            mask_out.write(np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2))
            areas.append(float(mask.mean()))
            if i in samples_at:
                samples.append((rgb, frame))
    if depth_frames is not None:
        depth_frames.close()
    _contact_sheet(samples, folder / "preview.jpg")
    return quality(areas)


def quality(areas):
    """Automatic check that replaces the manual full-clip review when it passes."""
    n = len(areas)
    if n == 0:
        raise MaskError("Видео пустое.")
    empty = sum(a < EMPTY_FRAME for a in areas)
    jumps = sum(1 for a, b in zip(areas, areas[1:]) if max(a, b) > EMPTY_FRAME and abs(a - b) / max(a, b) > 0.5)
    stats = {
        "frames": n,
        "empty_frames": empty,
        "empty_share": round(empty / n, 3),
        "mean_area": round(sum(areas) / n, 4),
        "flicker_frames": jumps,
        "issues": [],
    }
    if empty == n:
        raise MaskError("Человек не найден ни на одном кадре. Уточните, кого заменить (на английском, например «man in black jacket»), или смените режим маски.")
    if empty / n > 0.10:
        stats["issues"].append(f"Человек не найден на {empty} из {n} кадров — на этих кадрах он останется виден модели.")
    if jumps / n > 0.05:
        stats["issues"].append(f"Маска резко меняется на {jumps} кадрах — возможны мерцания.")
    if stats["mean_area"] > 0.6:
        stats["issues"].append("Маска закрывает больше 60% кадра — возможно, выделено лишнее.")
    stats["ok"] = not stats["issues"]
    return stats


def _contact_sheet(samples, dst, row_height=260):
    tiles = []
    for rgb, guide in samples:
        pair = np.concatenate([rgb, guide], axis=0)
        scale = (row_height * 2) / pair.shape[0]
        tiles.append(cv2.resize(pair, (max(1, int(pair.shape[1] * scale)), row_height * 2), interpolation=cv2.INTER_AREA))
    if tiles:
        sheet = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(dst), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def save_stats(folder, stats):
    (Path(folder) / "mask-qa.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
