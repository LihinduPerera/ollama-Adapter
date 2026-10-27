[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $env:USERPROFILE ".config\opencode\opencode.json"),
    [string]$BackupPath = "",
    [switch]$List
)

$ErrorActionPreference = "Stop"

$ConfigDir = Split-Path -Parent $ConfigPath
$BackupPattern = (Split-Path -Leaf $ConfigPath) + ".bak-*"
if (-not (Test-Path -LiteralPath $ConfigDir -PathType Container)) {
    throw "OpenCode config directory not found: $ConfigDir"
}

if ($List) {
    $backups = Get-ChildItem -LiteralPath $ConfigDir -Filter $BackupPattern -File |
        Sort-Object LastWriteTime -Descending
    if (-not $backups) {
        Write-Host "No backups found in $ConfigDir"
        return
    }
    $backups | Select-Object LastWriteTime, Length, FullName | Format-Table -AutoSize
    return
}

if (-not $BackupPath) {
    $BackupPath = Get-ChildItem -LiteralPath $ConfigDir -Filter $BackupPattern -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $BackupPath) {
        throw "No backup found. Pass -BackupPath explicitly, or use -List to inspect available backups."
    }
}

if (-not (Test-Path -LiteralPath $BackupPath -PathType Leaf)) {
    throw "Backup file not found: $BackupPath"
}

$raw = Get-Content -LiteralPath $BackupPath -Raw
try {
    $null = $raw | ConvertFrom-Json
}
catch {
    throw "Backup is not valid JSON; refusing to restore it: $BackupPath"
}

Copy-Item -LiteralPath $BackupPath -Destination $ConfigPath -Force
Write-Host "Restored $ConfigPath from $BackupPath"
Write-Host "Restart OpenCode for the change to take effect."
