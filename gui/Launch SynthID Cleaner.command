#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  source venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements-gui.txt
else
  source venv/bin/activate
fi
python gui.py
