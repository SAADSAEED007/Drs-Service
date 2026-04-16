param(
    [string]$PythonExe = "python",
    [string]$VenvDir = ".venv",
    [string]$YoloModelPath = "",
    [switch]$DisableStabilization,
    [switch]$DisableReplayAudio
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Directory {
    param([string]$PathValue)
    if (-not (Test-Path -LiteralPath $PathValue)) {
        New-Item -ItemType Directory -Path $PathValue | Out-Null
    }
}

Write-Step "Creating required directories"
Ensure-Directory "tmp"
Ensure-Directory "public"
Ensure-Directory "public\videos"
Ensure-Directory "models"

Write-Step "Creating virtual environment in $VenvDir"
if (-not (Test-Path -LiteralPath $VenvDir)) {
    & $PythonExe -m venv $VenvDir
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment python executable was not found at $venvPython"
}

Write-Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Step "Installing Python dependencies"
& $venvPython -m pip install -r requirements.txt

Write-Step "Checking FFmpeg availability"
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpeg) {
    Write-Warning "FFmpeg is not installed or not on PATH. Replay videos may not play correctly in browsers."
} else {
    Write-Host "FFmpeg found at $($ffmpeg.Source)" -ForegroundColor Green
}

Write-Step "Preparing environment configuration"
$envExamplePath = Join-Path $repoRoot ".env.drs.example"
$envPath = Join-Path $repoRoot ".env.drs"
$envLines = @(
    "DRS_MODELS_DIRECTORY=models"
    "DRS_YOLO_MODEL=$YoloModelPath"
    "DRS_ENABLE_STABILIZATION=$([int](-not $DisableStabilization.IsPresent))"
    "DRS_ENABLE_REPLAY_AUDIO=$([int](-not $DisableReplayAudio.IsPresent))"
)
$envLines | Set-Content -Path $envExamplePath
$envLines | Set-Content -Path $envPath

Write-Step "Setup complete"
Write-Host "Activate the virtual environment with:" -ForegroundColor Yellow
Write-Host "  $VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the service with:" -ForegroundColor Yellow
Write-Host "  uvicorn main:app --host 0.0.0.0 --port 8001"
Write-Host ""
Write-Host "Environment file created at .env.drs with the current DRS options." -ForegroundColor Green
