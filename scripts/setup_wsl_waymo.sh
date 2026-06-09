#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is intended for WSL/Linux. Native Windows can download Waymo data but should parse TFRecords from WSL/Linux."
  exit 1
fi

ENV_DIR="${VISIONROUTE_WAYMO_ENV:-$HOME/visionroute-py310}"
BOOTSTRAP_ENV="${VISIONROUTE_BOOTSTRAP_ENV:-$HOME/visionroute-venv}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required."
  exit 1
fi

if ! python3 -m pip --version >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Missing pip/venv. Install them first:"
  echo "  sudo apt-get update && sudo apt-get install -y python3-pip python3-venv python3.12-venv"
  echo "From Windows, this often works without a sudo password:"
  echo "  wsl.exe -u root -- bash -lc \"apt-get update && apt-get install -y python3-pip python3-venv python3.12-venv\""
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv "$BOOTSTRAP_ENV"
  # shellcheck disable=SC1091
  source "$BOOTSTRAP_ENV/bin/activate"
  python -m pip install --upgrade pip uv
  export PATH="$BOOTSTRAP_ENV/bin:$PATH"
fi

uv python install 3.10
rm -rf "$ENV_DIR"
uv venv --python 3.10 "$ENV_DIR"
"$ENV_DIR/bin/python" -m ensurepip --upgrade || true
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install \
  numpy==1.23.5 \
  protobuf==3.20.3 \
  tensorflow==2.12.0 \
  opencv-python-headless==4.8.1.78 \
  PyYAML tqdm matplotlib pytest
"$ENV_DIR/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$ENV_DIR/bin/python" -m pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.7

echo "WSL Waymo environment ready:"
echo "  source $ENV_DIR/bin/activate"
