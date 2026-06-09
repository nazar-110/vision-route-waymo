#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INSTALL_WAYMO=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --waymo|--with-waymo)
      INSTALL_WAYMO=1; shift ;;
    *)
      echo "Unknown argument: $1"; exit 2 ;;
  esac
done

find_python() {
  for candidate in "${PYTHON_BIN:-}" python python.exe py python3; do
    if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "Python was not found. Install Python 3.10+ first."
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
fi

if [ -x ".venv/bin/python" ]; then
  PYTHON_RUNNER=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_RUNNER=".venv/Scripts/python.exe"
else
  PYTHON_RUNNER="$PYTHON_BIN"
fi

"$PYTHON_RUNNER" -m pip install --upgrade pip
"$PYTHON_RUNNER" -m pip install -r requirements.txt
if [ "$INSTALL_WAYMO" -eq 1 ]; then
  "$PYTHON_RUNNER" -m pip install -r requirements-waymo.txt
  echo "If pip could not find a waymo_open_dataset wheel, install the official wheel matching your Python/TensorFlow version."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH. OpenCV can still write MP4, but install ffmpeg for robust video tooling:"
  echo "  macOS: brew install ffmpeg"
  echo "  Ubuntu: sudo apt-get install ffmpeg"
  echo "  Windows: winget install Gyan.FFmpeg"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Google Cloud CLI not found. Install it before downloading Waymo data:"
  echo "  https://cloud.google.com/sdk/docs/install"
fi

"$PYTHON_RUNNER" scripts/check_environment.py
