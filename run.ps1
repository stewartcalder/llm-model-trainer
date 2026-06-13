# run.ps1 — Windows launcher for the LoRA Data Builder backend.
#
# Use this on native Windows (NOT WSL) so the screen-text-scraper can actually
# see the screen and drive the mouse — mss/pyautogui only work on the OS that
# owns the display. Under WSL the screen grab comes back black.
#
# Usage (PowerShell, from the repo root):
#   .\run.ps1
#
# Serves the app on http://localhost:8000. The prebuilt frontend in
# frontend\dist is committed, so Node is NOT required. Keep your WSL instance
# running if your PostgreSQL lives there — this backend reaches it over
# 127.0.0.1:5433 via WSL2 localhost forwarding (same backend\.env).

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venv   = Join-Path $PSScriptRoot "backend\.venv"
$python = Join-Path $venv "Scripts\python.exe"

# --- one-time venv + deps ---
if (-not (Test-Path $python)) {
    Write-Host "Creating Python venv and installing backend deps..." -ForegroundColor Cyan
    python -m venv $venv
    & $python -m pip install --quiet --upgrade pip
    & $python -m pip install -r (Join-Path $PSScriptRoot "backend\requirements.txt")
    # Screen-text-scraper tool deps (not in requirements.txt — optional there).
    & $python -m pip install mss pyautogui pytesseract Pillow
}

# --- frontend ---
if (-not (Test-Path (Join-Path $PSScriptRoot "frontend\dist\index.html"))) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "Building frontend..." -ForegroundColor Cyan
        Push-Location (Join-Path $PSScriptRoot "frontend")
        npm install
        npm run build
        Pop-Location
    } else {
        Write-Warning "frontend\dist is missing and npm isn't installed. Pull the committed dist, or install Node and re-run."
    }
}

# --- tesseract OCR engine check (the scraper needs the binary, pip can't supply it) ---
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    Write-Warning "tesseract not found on PATH. The scraper's OCR step needs it."
    Write-Warning "Install: winget install UB-Mannheim.TesseractOCR  (then reopen PowerShell, or add its folder to PATH)."
}

# --- serve ---
Write-Host "Serving app on http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Green
Set-Location -Path (Join-Path $PSScriptRoot "backend")
& $python -m uvicorn app.main:app --port 8000
