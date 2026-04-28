param(
  [string]$Url = "http://localhost:8000/explain",
  [int]$TopK = 10,
  [switch]$Normalized,
  [string]$OutFile = "test_explain_payload.json"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Prefer venv python if it exists
$venvPython = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
  $python = $venvPython
} else {
  $python = "python"
}

$payloadPath = & $python "scripts/make_explain_payload.py" --out $OutFile --top-k $TopK @(
  if ($Normalized) { "--normalized" } else { }
)

if ($LASTEXITCODE -ne 0) {
  throw "Failed to generate payload (exit code $LASTEXITCODE)."
}

if (-not $payloadPath) {
  throw "Payload generator did not return an output path."
}

if (-not (Test-Path $payloadPath)) {
  throw "Payload file was not created: $payloadPath"
}

Write-Host "Posting payload file: $payloadPath" -ForegroundColor Cyan

# In PowerShell, make sure @file is passed as a string to curl.exe
curl.exe -X POST $Url -H "Content-Type: application/json" --data-binary "@$payloadPath"
