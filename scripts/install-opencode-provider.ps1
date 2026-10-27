[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $env:USERPROFILE ".config\opencode\opencode.json"),
    [string]$ProviderId = "ollama-adapter",
    [string]$BaseUrl = "http://127.0.0.1:8765/v1",
    [string]$Model = "qwen2.5-coder:7b",
    [int]$ContextLimit = 16384,
    [int]$OutputLimit = 4096,
    [int]$TimeoutMs = 300000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "OpenCode config not found: $ConfigPath"
}

$raw = Get-Content -LiteralPath $ConfigPath -Raw
try {
    $config = $raw | ConvertFrom-Json
}
catch {
    throw "OpenCode config is not plain JSON (it may contain comments). Back it up and edit it manually, or remove comments first."
}

if ($null -eq $config) {
    throw "OpenCode config is empty: $ConfigPath"
}

if (-not $config.PSObject.Properties["provider"]) {
    $config | Add-Member -NotePropertyName "provider" -NotePropertyValue ([pscustomobject]@{})
}
if ($null -eq $config.provider) {
    $config.provider = [pscustomobject]@{}
}

$providerEntry = [ordered]@{
    npm     = "@ai-sdk/openai-compatible"
    name    = "Ollama Tool Adapter"
    options = [ordered]@{
        baseURL = $BaseUrl
        timeout = $TimeoutMs
    }
    models  = [ordered]@{
        $Model = [ordered]@{
            name      = "Qwen2.5 Coder 7B (adapter)"
            tool_call = $true
            limit     = [ordered]@{
                context = $ContextLimit
                output  = $OutputLimit
            }
        }
    }
}

$existing = $config.provider.PSObject.Properties[$ProviderId]
if ($existing) {
    $config.provider.$ProviderId = [pscustomobject]$providerEntry
}
else {
    $config.provider | Add-Member -NotePropertyName $ProviderId -NotePropertyValue ([pscustomobject]$providerEntry)
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupPath = "$ConfigPath.bak-$timestamp"
$suffix = 1
while (Test-Path -LiteralPath $backupPath) {
    $backupPath = "$ConfigPath.bak-$timestamp-$suffix"
    $suffix++
}
Copy-Item -LiteralPath $ConfigPath -Destination $backupPath
Write-Host "Backed up config to: $backupPath"

$json = $config | ConvertTo-Json -Depth 20
try {
    $null = $json | ConvertFrom-Json
}
catch {
    throw "Refusing to write invalid JSON to $ConfigPath"
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ConfigPath, $json + [Environment]::NewLine, $utf8NoBom)

Write-Host "Added provider '$ProviderId' to $ConfigPath"
Write-Host "Model: $ProviderId/$Model"
Write-Host "Restart OpenCode, then verify with:"
Write-Host "  opencode models"
Write-Host ""
Write-Host "To revert:"
Write-Host "  .\scripts\revert-opencode-config.ps1 -BackupPath `"$backupPath`""
