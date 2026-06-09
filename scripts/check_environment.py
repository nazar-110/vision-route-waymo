"""Check local environment for VisionRoute."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_module(name: str, required: bool = True) -> bool:
    ok = importlib.util.find_spec(name) is not None
    status = "OK" if ok else ("MISSING" if required else "optional missing")
    print(f"{name:24s} {status}")
    return ok or not required


def find_command(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate = Path(local) / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / f"{name}.cmd"
            if candidate.exists():
                return str(candidate)
    return None


def check_command(name: str, required: bool = False) -> bool:
    found = find_command(name)
    ok = found is not None
    status = "OK" if ok else ("MISSING" if required else "optional missing")
    suffix = f" ({found})" if found and name in {"gcloud", "gsutil"} else ""
    print(f"{name:24s} {status}{suffix}")
    return ok or not required


def adc_credentials_path() -> Path:
    """Return the expected Application Default Credentials file path."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "gcloud" / "application_default_credentials.json"
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def main() -> int:
    print("VisionRoute environment check")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    required_ok = True
    for module in ["numpy", "cv2", "matplotlib", "yaml", "tqdm", "torch", "torchvision"]:
        required_ok = check_module(module, required=True) and required_ok
    check_module("tensorflow", required=False)
    check_module("waymo_open_dataset", required=False)
    check_module("pytest", required=False)
    check_command("ffmpeg", required=False)
    check_command("gcloud", required=False)
    check_command("gsutil", required=False)

    try:
        import torch

        print(f"{'torch cuda':24s} {'OK' if torch.cuda.is_available() else 'CPU only'}")
    except Exception:
        pass

    gcloud = find_command("gcloud")
    if gcloud:
        proc = subprocess.run([gcloud, "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], capture_output=True, text=True)
        active = proc.stdout.strip()
        print(f"{'gcloud auth':24s} {'OK: ' + active if active else 'not authenticated'}")
        adc_path = adc_credentials_path()
        print(f"{'gcloud ADC':24s} {'OK' if adc_path.exists() else 'not set; run gcloud auth application-default login'}")
    else:
        print("Google Cloud CLI install: https://cloud.google.com/sdk/docs/install")

    repo_root = Path(__file__).resolve().parents[1]
    for path in ["configs/default.yaml", "src", "outputs", "data/raw"]:
        exists = (repo_root / path).exists()
        print(f"{path:24s} {'OK' if exists else 'MISSING'}")
        required_ok = exists and required_ok

    waymo_data = repo_root / "data/raw/waymo_perception"
    print(f"{'data/raw/waymo_perception':24s} {'OK' if waymo_data.exists() else 'missing; run download script'}")

    if not required_ok:
        print("\nOne or more required dependencies are missing. Run scripts/install_deps.sh or setup_windows.ps1.")
        return 1
    print("\nEnvironment is ready for the VisionRoute codebase.")
    if importlib.util.find_spec("waymo_open_dataset") is None:
        print("Waymo parsing needs the official waymo_open_dataset wheel for your Python/TensorFlow version.")
    if not waymo_data.exists():
        print("Waymo data has not been downloaded yet. Run scripts/download_waymo_perception_subset.sh after accepting Waymo terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
