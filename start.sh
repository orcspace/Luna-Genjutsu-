#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || ./setup.sh
exec .venv/bin/python run.py
