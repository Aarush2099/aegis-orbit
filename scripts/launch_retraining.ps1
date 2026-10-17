param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string[]]$Methods = @("sac_primary"),
    [string]$Config = "configs\sac_unshielded.yaml"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputDir))
}
$resolvedConfig = if ([System.IO.Path]::IsPathRooted($Config)) {
    [System.IO.Path]::GetFullPath($Config)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Config))
}
$pythonExecutable = (Resolve-Path (Join-Path $repositoryRoot ".venv\Scripts\python.exe")).Path

if (Test-Path -LiteralPath $resolvedOutputDir) {
    $existingEntries = @(Get-ChildItem -LiteralPath $resolvedOutputDir -Force)
    if ($existingEntries.Count -gt 0) {
        Write-Error "Refusing to start: output directory is non-empty: $resolvedOutputDir"
        exit 1
    }
} else {
    New-Item -ItemType Directory -Path $resolvedOutputDir | Out-Null
}

$stdoutPath = Join-Path $resolvedOutputDir "run.log"
$stderrPath = Join-Path $resolvedOutputDir "run.log.err"
$reservationPath = Join-Path $resolvedOutputDir ".runner-output-reservation"
$launchPath = Join-Path $resolvedOutputDir "launch.json"
$reservationToken = [guid]::NewGuid().ToString("N")
[System.IO.File]::WriteAllText($reservationPath, $reservationToken)
[System.IO.File]::WriteAllText($stdoutPath, "")
[System.IO.File]::WriteAllText($stderrPath, "")

$pythonArguments = @(
    "-u",
    "-m",
    "benchmarks.run_unshielded_benchmark",
    "--config",
    $Config,
    "--output-dir",
    $OutputDir,
    "--methods"
) + $Methods
$quotedArguments = foreach ($argument in $pythonArguments) {
    '"' + $argument.Replace('"', '\"') + '"'
}
$argumentLine = $quotedArguments -join " "
$fullCommandLine = '"' + $pythonExecutable + '" ' + $argumentLine
$launchMetadata = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    command_line = $fullCommandLine
    config_path = $resolvedConfig
    python_executable = $pythonExecutable
    git_commit = "no git"
}
$launchMetadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchPath -Encoding UTF8

try {
    $env:AEGIS_ORBIT_OUTPUT_RESERVATION = $reservationToken
    $child = Start-Process `
        -FilePath $pythonExecutable `
        -ArgumentList $argumentLine `
        -WorkingDirectory $repositoryRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    [System.IO.File]::WriteAllText(
        (Join-Path $resolvedOutputDir "pid.txt"),
        [string]$child.Id
    )
} catch {
    Remove-Item Env:AEGIS_ORBIT_OUTPUT_RESERVATION -ErrorAction SilentlyContinue
    Write-Error "Failed to start retraining process: $($_.Exception.Message)"
    exit 1
}
Remove-Item Env:AEGIS_ORBIT_OUTPUT_RESERVATION -ErrorAction SilentlyContinue

$deadline = (Get-Date).AddSeconds(180)
$firstProgressLine = $null
while ((Get-Date) -lt $deadline -and -not $firstProgressLine) {
    Start-Sleep -Seconds 2
    if (Test-Path -LiteralPath $stdoutPath) {
        $firstProgressLine = Get-Content -LiteralPath $stdoutPath -TotalCount 1 -ErrorAction SilentlyContinue
    }
}

if ($firstProgressLine) {
    Write-Output "PASS PID=$($child.Id) first_progress_line=$firstProgressLine"
    exit 0
}

Write-Error "FAIL PID=$($child.Id): no progress line within 180 seconds"
if (Test-Path -LiteralPath $stderrPath) {
    Get-Content -LiteralPath $stderrPath -Tail 30
}
exit 1
