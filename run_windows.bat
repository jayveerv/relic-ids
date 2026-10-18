@echo off
REM RELIC-IDS on Windows. For live capture: install Npcap (https://npcap.com) and run this file as Administrator.
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
streamlit run app.py
