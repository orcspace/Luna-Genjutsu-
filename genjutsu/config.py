"""Settings live in the project's .env file; real environment variables win."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
RUNS = ROOT / "runs"
BIN = ROOT / "bin"
MODELS = ROOT / "models"

DEFAULTS = {
    "PROVIDER": "dreamina",  # dreamina | enhancor | byteplus | mock
    "ENHANCOR_API_KEY": "",
    "ENHANCOR_BASE_URL": "https://apireq.enhancor.ai/api/seedance2.5/v1",
    "USEAPI_TOKEN": "",
    "DREAMINA_ACCOUNT": "",  # email or REGION:email; empty = first connected account
    "DREAMINA_MODEL": "seedance-2.5",
    "BYTEPLUS_API_KEY": "",
    "BYTEPLUS_BASE_URL": "https://ark.ap-southeast.bytepluses.com/api/v3",
    "BYTEPLUS_MODEL": "dreamina-seedance-2-0-260128",
    "REPLICATE_API_TOKEN": "",
    "MEDIA_HOST": "tunnel",  # tunnel | tmpfiles | litterbox
    "PUBLIC_BASE_URL": "",  # your own HTTPS tunnel to the public port, instead of cloudflared
    "VIDEO_ENCODER": "auto",  # auto (NVENC on an NVIDIA card) | nvenc | cpu
    "LOCAL_QUALITY": "best",  # MediaPipe pose model: best (heavy) | balanced (full) | fast (lite)
    "PORT": "8770",
    "PUBLIC_PORT": "8771",
}

SECRET_KEYS = {"ENHANCOR_API_KEY", "USEAPI_TOKEN", "BYTEPLUS_API_KEY", "REPLICATE_API_TOKEN"}


def _read_env_file():
    values = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep:
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name):
    if os.environ.get(name):
        return os.environ[name]
    return _read_env_file().get(name) or DEFAULTS.get(name, "")


def has(name):
    return bool(setting(name))


def save_settings(updates):
    """Write keys into .env, keeping comments and unrelated lines intact."""
    allowed = {k: v for k, v in updates.items() if k in DEFAULTS}
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    written = set()
    for i, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if key in allowed and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={allowed[key]}"
            written.add(key)
    for key, value in allowed.items():
        if key not in written:
            lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key in allowed:
        # The file is the source of truth for values set from the UI.
        os.environ.pop(key, None)


def masked(value):
    if not value:
        return ""
    return value[:4] + "…" + value[-4:] if len(value) > 12 else "••••"


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80] or "file"
