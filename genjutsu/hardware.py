"""What this PC can do: NVIDIA encoder, CPU threads, RAM. Used to pick fast defaults.

Tuned for a typical gaming PC (e.g. RTX 3070 8 GB, i5-13400F, 16 GB): video encodes go to
NVENC so the CPU stays free for MediaPipe, which runs on the CPU (its Python GPU delegate is
not available on Windows).
"""
import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache

from .config import setting

POSE_VARIANTS = {"fast": "lite", "balanced": "full", "best": "heavy"}


def _run(args, timeout=20):
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=1)
def gpu():
    """Name and memory of the first NVIDIA GPU, or None."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    out = _run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out or out.returncode != 0 or not out.stdout.strip():
        return None
    name, _, mem = out.stdout.strip().splitlines()[0].partition(",")
    try:
        return {"name": name.strip(), "vram_gb": round(int(mem.strip()) / 1024, 1)}
    except ValueError:
        return {"name": name.strip(), "vram_gb": None}


@lru_cache(maxsize=1)
def nvenc_works():
    """Actually encode a few frames: having the encoder compiled in is not enough without a driver."""
    from .media import ffmpeg

    out = _run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=24:d=0.2",
                "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p",
                "-f", "null", "-"], timeout=30)
    return bool(out and out.returncode == 0)


def encoder():
    """'nvenc' or 'cpu' for intermediate and final encodes."""
    choice = setting("VIDEO_ENCODER")
    if choice == "cpu":
        return "cpu"
    if choice == "nvenc":
        return "nvenc"
    return "nvenc" if nvenc_works() else "cpu"


def pose_variant():
    return POSE_VARIANTS.get(setting("LOCAL_QUALITY"), "heavy")


def cpu_threads():
    return os.cpu_count() or 4


@lru_cache(maxsize=1)
def ram_gb():
    try:
        system = platform.system()
        if system == "Windows":
            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = Status()
            st.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 2 ** 30, 1)
        if system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return round(int(out.stdout) / 2 ** 30, 1)
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2 ** 20, 1)
    except Exception:
        return None


def summary():
    return {
        "gpu": gpu(),
        "encoder": encoder(),
        "nvenc": nvenc_works(),
        "cpu_threads": cpu_threads(),
        "ram_gb": ram_gb(),
        "pose_model": pose_variant(),
        "os": platform.system(),
    }
