<# This little helper lets me kick off the pipeline without remembering arguments. #>
param(
    [string]$ProjectRoot = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\\.."
)

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$logDir = Join-Path $ProjectRoot "logs"
if (!(Test-Path $logDir)) {
    # TODO: add log rotation when these files start piling up.
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "pipeline_$timestamp.log"

# Keeping the messaging friendly because I often run this late at night.
Write-Host "Running pipeline at $timestamp..."
Push-Location $ProjectRoot

try {
    & python "src\student_loan_default_analysis.py" 2>&1 | Tee-Object -FilePath $logFile
} finally {
    Pop-Location
}

Write-Host "Pipeline run complete. Log saved to $logFile"

