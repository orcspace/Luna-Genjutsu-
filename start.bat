@echo off
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
  .venv\Scripts\pip install -r requirements-local.txt
  if not exist .env copy .env.example .env
)
.venv\Scripts\python run.py
