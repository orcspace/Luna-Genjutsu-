"""Every video/audio operation goes through the ffmpeg binary; frames move as raw RGB over pipes."""
import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

FPS = 24


class MediaError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise MediaError("ffmpeg не найден. Установите ffmpeg (brew install ffmpeg / sudo apt install ffmpeg).")


@lru_cache(maxsize=1)
def ffprobe():
    return shutil.which("ffprobe")


@lru_cache(maxsize=1)
def has_rubberband():
    out = subprocess.run([ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return bool(re.search(r"\srubberband\s", out))


def run(args, what="ffmpeg"):
    proc = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"{what}: {proc.stderr.strip()[-800:]}")


def probe(path):
    """Duration, display size (after rotation), fps and whether there is an audio track."""
    path = str(path)
    if ffprobe():
        proc = subprocess.run(
            [ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise MediaError("Не удалось прочитать файл: " + proc.stderr.strip()[-300:])
        data = json.loads(proc.stdout)
        video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
        if video is None:
            raise MediaError("В файле нет видеодорожки.")
        width, height = int(video["width"]), int(video["height"])
        rotation = abs(int(float(video.get("tags", {}).get("rotate", 0) or 0)))
        for side in video.get("side_data_list", []) or []:
            if "rotation" in side:
                rotation = abs(int(float(side["rotation"])))
        if rotation % 180 == 90:
            width, height = height, width
        num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        duration = float(data["format"].get("duration") or video.get("duration") or 0)
        return {
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"]),
        }
    # Fallback for a bundled ffmpeg without ffprobe: parse `ffmpeg -i`.
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    v = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", err)
    if not v:
        raise MediaError("В файле нет видеодорожки.")
    f = re.search(r"([\d.]+) fps", err)
    width, height = int(v.group(1)), int(v.group(2))
    if re.search(r"rotate\s*:\s*-?(90|270)|rotation of -?(90|270)", err):
        width, height = height, width
    return {
        "duration": int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0,
        "width": width,
        "height": height,
        "fps": float(f.group(1)) if f else 0.0,
        "has_audio": "Audio:" in err,
    }


def duration(path):
    """Length of any media file, audio-only included."""
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        raise MediaError("Не удалось определить длительность файла.")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def normalize_video(src, dst, max_side=960, max_seconds=None):
    """Constant 24 fps, long side ≤ max_side, even dimensions, no audio."""
    scale = (
        f"scale='if(gte(iw,ih),min({max_side},iw),-2)':'if(gte(iw,ih),-2,min({max_side},ih))':flags=lanczos,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
    )
    args = ["-i", src, "-an", "-vf", f"fps={FPS},{scale}", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if max_seconds:
        args += ["-t", f"{max_seconds:.3f}"]
    run(args + [dst], "нормализация видео")
    return probe(dst)


def extract_audio(src, dst, seconds=None):
    args = ["-i", src, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le"]
    if seconds:
        args += ["-t", f"{seconds:.3f}"]
    run(args + [dst], "извлечение аудио")


def pitch_shift(src, dst, semitones=3):
    """Shift pitch while keeping duration. Rubber Band when ffmpeg has it, resample + atempo otherwise."""
    ratio = 2 ** (semitones / 12)
    if has_rubberband():
        af = f"rubberband=pitch={ratio:.6f}:pitchq=quality"
    else:
        af = f"asetrate=44100*{ratio:.6f},aresample=44100,atempo={1 / ratio:.6f}"
    run(["-i", src, "-af", af, "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", dst], "сдвиг тона")


def mux(video, audio, dst):
    """Silent video + guide audio, cut to the video length."""
    run(["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", dst], "сборка видео со звуком")


def cut(src, dst, start, duration):
    run(["-ss", f"{start:.3f}", "-i", src, "-t", f"{duration:.3f}", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dst], "нарезка")


def concat(parts, dst):
    """Re-encodes so segments with different sizes or codecs from the provider still join cleanly."""
    first = probe(parts[0])
    w, h = first["width"], first["height"]
    inputs, chains = [], []
    for i, p in enumerate(parts):
        inputs += ["-i", p]
        chains.append(f"[{i}:v]fps={FPS},scale={w}:{h}:flags=lanczos,setsar=1[v{i}]")
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
    run(inputs + ["-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dst], "склейка")


def restore_audio(generated, source, dst, duration):
    """Put the untouched original soundtrack on the generated picture.

    The picture is held on its last frame or trimmed to the source length, never stretched,
    so speech and beats stay where they were in the original clip.
    """
    gen = probe(generated)
    src_has_audio = probe(source)["has_audio"]
    gap = duration - gen["duration"]
    video_args = ["-c:v", "copy"]
    if gap > 0.05:
        video_args = ["-vf", f"tpad=stop_mode=clone:stop_duration={gap:.3f}", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    elif gap < -0.05:
        video_args = ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    base = ["-i", generated]
    if not src_has_audio:
        run(base + ["-map", "0:v:0", *video_args, "-t", f"{duration:.3f}", "-movflags", "+faststart", dst], "финальный файл")
        return
    base += ["-i", source, "-map", "0:v:0", "-map", "1:a:0", *video_args]
    tail = ["-t", f"{duration:.3f}", "-movflags", "+faststart", dst]
    try:
        run(base + ["-c:a", "copy"] + tail, "возврат оригинального звука")
    except MediaError:
        # Codecs mp4 cannot carry (e.g. Opus from WebM) are re-encoded at high bitrate.
        run(base + ["-c:a", "aac", "-b:a", "320k"] + tail, "возврат оригинального звука")


def thumbnail(video, dst, at=0.5):
    run(["-ss", f"{at:.2f}", "-i", video, "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-q:v", "3", dst], "превью")


def read_frames(path, width, height):
    """Yields RGB uint8 frames of exactly width×height."""
    proc = subprocess.Popen(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", f"scale={width}:{height}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


class FrameWriter:
    def __init__(self, path, width, height, crf=16):
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise MediaError("Запись видео: " + err[-500:])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
