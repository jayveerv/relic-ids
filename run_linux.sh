#!/usr/bin/env bash
# RELIC-IDS on Linux/macOS.  ./run_linux.sh        (upload and demo)
#                            ./run_linux.sh --live (with live capture; asks for sudo)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
if [ "$1" = "--live" ]; then
  sudo -E .venv/bin/streamlit run app.py
else
  .venv/bin/streamlit run app.py
fi
