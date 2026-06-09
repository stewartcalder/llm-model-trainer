#!/usr/bin/env bash
# Build the frontend (if needed) and launch the single-user web app.
# Usage:
#   ./run.sh            # build frontend + serve everything on http://localhost:8000
#   ./run.sh dev        # run backend (:8000) and Vite dev server (:5173) with hot reload
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-prod}"

# --- one-time setup ---
if [ ! -d backend/.venv ]; then
  echo "Creating Python venv and installing backend deps…"
  python3 -m venv backend/.venv
  backend/.venv/bin/python -m pip install --quiet --upgrade pip
  backend/.venv/bin/python -m pip install -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend deps…"
  (cd frontend && npm install)
fi

if [ "$MODE" = "dev" ]; then
  echo "Starting backend (:8000) and Vite dev server (:5173)…"
  (cd backend && .venv/bin/python -m uvicorn app.main:app --reload --port 8000) &
  BACK=$!
  trap "kill $BACK 2>/dev/null" EXIT
  (cd frontend && npm run dev)
else
  echo "Building frontend…"
  (cd frontend && npm run build)
  echo "Serving app on http://localhost:8000"
  cd backend && exec .venv/bin/python -m uvicorn app.main:app --port 8000
fi
