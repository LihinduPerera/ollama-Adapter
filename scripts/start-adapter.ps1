[CmdletBinding()]
param(
    [string]$Workspace = (Get-Location).Path,
    [int]$Port = 8765,
    [string]$Models = "",
    [string]$ApiKey = "",
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python virtual environment not found at $Python. Run scripts\setup.ps1 first."
}

if (-not (Test-Path -LiteralPath $Workspace -PathType Container)) {
    throw "Workspace does not exist or is not a directory: $Workspace"
}

$WorkspacePath = (Resolve-Path -LiteralPath $Workspace).Path

# These values apply to this process only. Values already present in .env are
# loaded by the adapter, but never override variables set here.
$env:ADAPTER_HOST = "127.0.0.1"
$env:ADAPTER_PORT = "$Port"
$env:ADAPTER_WORKSPACE = $WorkspacePath
if ($Models -ne "") {
    $env:ADAPTER_MODELS = $Models
}
$env:ADAPTER_LOG_LEVEL = $LogLevel
if ($ApiKey) {
    $env:ADAPTER_API_KEY = $ApiKey
}

# Keep working even if the editable install is missing.
$env:PYTHONPATH = Join-Path $RepoRoot "src"

Write-Host "Starting Ollama tool adapter"
Write-Host "  URL:       http://127.0.0.1:$Port"
Write-Host "  Workspace: $WorkspacePath"
Write-Host "  Models:    $Models"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

Push-Location $RepoRoot
try {
    & $Python -m adapter.main
}
finally {
    Pop-Location
}
