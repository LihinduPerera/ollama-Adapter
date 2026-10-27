[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $BasePython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $BasePython) {
        throw "Python was not found on PATH. Install Python 3.10 or newer first."
    }

    & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 or newer is required. Found: $(& python --version 2>&1)"
    }

    Write-Host "Creating virtual environment at $VenvDir"
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}

Write-Host "Installing pinned dependencies"
Push-Location $RepoRoot
try {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

    & $Python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies." }
}
finally {
    Pop-Location
}

$EnvFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath (Join-Path $RepoRoot ".env.example") -Destination $EnvFile
    Write-Host "Created .env from .env.example. Edit ADAPTER_WORKSPACE before starting the adapter."
}

if (-not $SkipTests) {
    Write-Host "Running tests"
    & $Python -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start the adapter with:"
Write-Host "  .\scripts\start-adapter.ps1 -Workspace `"D:\path\to\your\project`""
