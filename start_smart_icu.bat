@echo off
start "Smart ICU Backend" cmd /k "cd /d %~dp0backend && if not exist .venv python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt && uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 >nul
start "Smart ICU Frontend" cmd /k "cd /d %~dp0frontend && npm install && npm run dev"
