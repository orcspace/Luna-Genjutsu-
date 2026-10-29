import numpy as np
import pytest

from genjutsu import hardware, masking, media, providers
from conftest import box_masks


@pytest.fixture
def force_nvenc(monkeypatch):
    """Pretend an NVIDIA card is there. This container has none, so every NVENC encode fails for real."""
    monkeypatch.setattr(hardware, "encoder", lambda: "nvenc")
    monkeypatch.setattr(media, "_nvenc_failed", False)


def test_venc_choice(monkeypatch):
    monkeypatch.setattr(media, "_nvenc_failed", False)
    monkeypatch.setattr(hardware, "encoder", lambda: "nvenc")
    assert media.venc(16)[:2] == ["-c:v", "h264_nvenc"] and "19" in media.venc(16)
    monkeypatch.setattr(hardware, "encoder", lambda: "cpu")
    assert media.venc(16)[:4] == ["-c:v", "libx264", "-crf", "16"]


def test_cpu_fallback_rewrites_only_the_encoder_block(monkeypatch):
    monkeypatch.setattr(media, "_nvenc_failed", False)
    monkeypatch.setattr(hardware, "encoder", lambda: "nvenc")
    args = ["-i", "in.mp4", "-vf", "fps=24", *media.venc(16), "-c:a", "aac", "out.mp4"]
    out = media._cpu_fallback(args)
    assert out == ["-i", "in.mp4", "-vf", "fps=24", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "out.mp4"]


def test_failed_nvenc_falls_back_to_cpu(force_nvenc, make_video, tmp_path):
    meta = media.normalize_video(make_video(seconds=2), tmp_path / "n.mp4")
    assert meta["width"] > 0 and abs(meta["duration"] - 2) < 0.1
    assert media._nvenc_failed  # later encodes go straight to the CPU


def test_compose_redone_on_cpu_when_streaming_nvenc_dies(force_nvenc, make_video, tmp_path):
    meta = media.normalize_video(make_video(seconds=1, audio=False), tmp_path / "original.mp4")
    media._nvenc_failed = False  # normalising already tripped it; make the frame writer meet NVENC too
    meta["frames"] = round(meta["duration"] * media.FPS)
    qa = masking.compose(tmp_path / "original.mp4", box_masks(tmp_path, meta["frames"], meta["width"], meta["height"]), tmp_path, meta, status=lambda m: None)
    assert qa["ok"] and media.probe(tmp_path / "guidance-silent.mp4")["width"] == meta["width"]


def test_images_in_non_ascii_folders(tmp_path):
    folder = tmp_path / "Пользователь Нурамир"
    folder.mkdir()
    img = np.full((20, 30, 3), 77, np.uint8)
    media.write_image(folder / "маска.png", img)
    back = media.read_image(folder / "маска.png")
    assert back.shape == (20, 30, 3) and int(back[0, 0, 0]) == 77
    assert media.read_image(folder / "нет.png") is None


def test_file_uri_download_round_trip(tmp_path):
    src = tmp_path / "Видео" / "out file.mp4"
    src.parent.mkdir()
    src.write_bytes(b"data")
    providers.download(src.as_uri(), tmp_path / "copy.mp4")
    assert (tmp_path / "copy.mp4").read_bytes() == b"data"


def test_hardware_summary_without_gpu(monkeypatch):
    monkeypatch.setenv("LOCAL_QUALITY", "balanced")
    info = hardware.summary()
    assert info["cpu_threads"] >= 1 and info["pose_model"] == "full"
    assert info["encoder"] in ("nvenc", "cpu")
    monkeypatch.setenv("VIDEO_ENCODER", "cpu")
    assert hardware.encoder() == "cpu"
