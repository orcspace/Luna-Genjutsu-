#!/usr/bin/env bash
# One-time setup: virtual environment, dependencies, .env
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "Нужен Python 3.10+"'
[ -d .venv ] || $PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
if .venv/bin/pip install -q -r requirements-local.txt; then
  echo "✓ Локальное маскирование (MediaPipe) установлено"
else
  echo "! MediaPipe не установился — локальная маска недоступна, будет нужен REPLICATE_API_TOKEN"
fi
[ -f .env ] || cp .env.example .env
command -v ffmpeg >/dev/null || echo "! Системный ffmpeg не найден — будет использован встроенный (imageio-ffmpeg). Лучше: brew install ffmpeg / sudo apt install ffmpeg"
echo "✓ Готово. Запуск: ./start.sh"
