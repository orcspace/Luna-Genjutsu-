import subprocess

import numpy as np
import pytest

from genjutsu import media


def audio_md5(path):
    out = subprocess.run([media.ffmpeg(), "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_normalize_makes_24fps_even_and_silent(make_video, tmp_path):
    src = make_video(seconds=3, size="1921x1081")
    meta = media.normalize_video(src, tmp_path / "n.mp4")
    assert meta["width"] <= 960 and meta["width"] % 2 == 0 and meta["height"] % 2 == 0
    assert abs(meta["fps"] - 24) < 0.01
    assert not meta["has_audio"]
    assert abs(meta["duration"] - 3) < 0.1


def test_normalize_respects_max_seconds(make_video, tmp_path):
    meta = media.normalize_video(make_video(seconds=6), tmp_path / "n.mp4", max_seconds=4)
    assert abs(meta["duration"] - 4) < 0.1


def dominant_hz(wav):
    pcm = subprocess.run([media.ffmpeg(), "-loglevel", "error", "-i", str(wav), "-ac", "1", "-ar", "44100", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    samples = np.frombuffer(pcm, np.int16).astype(float)[44100:88200]
    return np.argmax(np.abs(np.fft.rfft(samples))) * 44100 / len(samples)


@pytest.mark.parametrize("rubberband", [True, False])
def test_pitch_shift_raises_pitch_and_keeps_duration(make_video, tmp_path, monkeypatch, rubberband):
    if rubberband and not media.has_rubberband():
        pytest.skip("ffmpeg without rubberband")
    monkeypatch.setattr(media, "has_rubberband", lambda: rubberband)
    media.extract_audio(make_video(seconds=4), tmp_path / "a.wav")
    media.pitch_shift(tmp_path / "a.wav", tmp_path / "p.wav", 3)
    assert abs(media.duration(tmp_path / "p.wav") - 4) < 0.1
    assert abs(dominant_hz(tmp_path / "p.wav") - 440 * 2 ** (3 / 12)) < 6


@pytest.mark.parametrize("gen_seconds", [3.4, 4, 5])
def test_restore_audio_copies_original_track_and_matches_length(make_video, tmp_path, gen_seconds):
    source = make_video(seconds=4, name="src.mp4")
    generated = make_video(seconds=gen_seconds, audio=False, name="gen.mp4")
    out = tmp_path / "out.mp4"
    media.restore_audio(generated, source, out, 4.0)
    info = media.probe(out)
    assert info["has_audio"]
    assert abs(info["duration"] - 4.0) < 0.15
    # The audio packets are the original ones, byte for byte.
    assert audio_md5(out) == audio_md5(source)


def test_restore_without_source_audio(make_video, tmp_path):
    source = make_video(seconds=4, audio=False, name="src.mp4")
    generated = make_video(seconds=4, audio=False, name="gen.mp4")
    media.restore_audio(generated, source, tmp_path / "out.mp4", 4.0)
    assert not media.probe(tmp_path / "out.mp4")["has_audio"]


def test_concat_joins_different_sizes(make_video, tmp_path):
    a = make_video(seconds=2, audio=False, name="a.mp4", size="640x360")
    b = make_video(seconds=3, audio=False, name="b.mp4", size="1280x720")
    media.concat([a, b], tmp_path / "c.mp4")
    info = media.probe(tmp_path / "c.mp4")
    assert (info["width"], info["height"]) == (640, 360)
    assert abs(info["duration"] - 5) < 0.15


def test_probe_without_ffprobe_matches(make_video, monkeypatch):
    src = make_video(seconds=3, size="640x360")
    real = media.probe(src)
    monkeypatch.setattr(media, "ffprobe", lambda: None)
    parsed = media.probe(src)
    assert (parsed["width"], parsed["height"], parsed["has_audio"]) == (real["width"], real["height"], real["has_audio"])
    assert abs(parsed["duration"] - real["duration"]) < 0.05 and abs(parsed["fps"] - 30) < 0.01


def test_frames_roundtrip(make_video, tmp_path):
    src = make_video(seconds=1, audio=False)
    frames = list(media.read_frames(src, 320, 180))
    assert frames and frames[0].shape == (180, 320, 3)
    with media.FrameWriter(tmp_path / "w.mp4", 320, 180) as w:
        for f in frames:
            w.write(f)
    assert media.probe(tmp_path / "w.mp4")["width"] == 320


def test_crossfade_concat_has_exact_length(make_video, tmp_path):
    # BytePlus-style parts come back rounded up; they are trimmed to plan before blending.
    a = make_video(seconds=6, audio=False, name="a.mp4")
    b = make_video(seconds=6, audio=False, name="b.mp4")
    media.crossfade_concat([a, b], [5.5, 5.5], 1.0, tmp_path / "x.mp4")
    assert abs(media.probe(tmp_path / "x.mp4")["duration"] - 10.0) < 0.1
