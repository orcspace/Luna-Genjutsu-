import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from genjutsu import jobs, media


@pytest.fixture
def make_video(tmp_path):
    def make(seconds=5, audio=True, size="640x360", name="clip.mp4", fps=30):
        path = tmp_path / name
        args = [media.ffmpeg(), "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}"]
        if audio:
            args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(args, check=True)
        return path
    return make


@pytest.fixture
def photo(tmp_path):
    def make(name="p.jpg", size=(600, 800)):
        path = tmp_path / name
        Image.new("RGB", size, (200, 150, 120)).save(path)
        return path
    return make


@pytest.fixture
def runs(tmp_path, monkeypatch):
    target = tmp_path / "runs"
    target.mkdir()
    monkeypatch.setattr(jobs, "RUNS", target)
    monkeypatch.setattr(jobs, "POLL_EVERY", 0.2)
    return target


def box_masks(folder, n, width, height):
    """A 'person': a box sliding left to right."""
    out = folder / "masks"
    out.mkdir(exist_ok=True)
    for i in range(n):
        m = np.zeros((height, width), np.uint8)
        x = int(i / max(1, n - 1) * (width - width // 4))
        m[height // 4: height - height // 8, x: x + width // 4] = 255
        cv2.imwrite(str(out / f"mask_{i:05d}.png"), m)
    return out
