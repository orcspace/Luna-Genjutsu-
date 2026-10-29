@echo off
rem UTF-8 everywhere: Cyrillic folder names and ffmpeg messages must not hit the console code page.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d %~dp0
where python >nul 2>nul || (echo Установите Python 3.10+ с python.org и отметьте "Add to PATH". & pause & exit /b 1)
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
  .venv\Scripts\pip install -r requirements-local.txt
  if not exist .env copy .env.example .env
)
.venv\Scripts\python run.py
pause
