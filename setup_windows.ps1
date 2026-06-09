param(
  [switch]$Waymo
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($Waymo) {
  Write-Host "Real Waymo TFRecord parsing should be set up from WSL/Linux:"
  Write-Host "  wsl.exe -u root -- bash -lc `"apt-get update && apt-get install -y python3-pip python3-venv python3.12-venv`""
  Write-Host "  wsl bash scripts/setup_wsl_waymo.sh"
}

Write-Host ""
Write-Host "VisionRoute Windows setup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: python scripts/check_environment.py"
Write-Host "For real Waymo parsing extras, use WSL/Linux: bash scripts/setup_wsl_waymo.sh"
