param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$Config = "configs\sac_unshielded.yaml",
    [string]$PolicyDir = "results\rerun_primary_20261014\policies"
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
$resolvedPolicyDir = if ([System.IO.Path]::IsPathRooted($PolicyDir)) {
    [System.IO.Path]::GetFullPath($PolicyDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $PolicyDir))
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
[System.IO.File]::WriteAllText((Join-Path $resolvedOutputDir "pid.txt"), "")

$pythonArguments = @(
    "-u",
    "-m",
    "benchmarks.run_shield_comparison",
    "--config",
    $Config,
    "--output-dir",
    $OutputDir,
    "--policy-dir",
    $PolicyDir,
    "--d-safe",
    "40",
    "67",
    "--truth-model",
    "cw",
    "cw_j2"
)
$quotedArguments = foreach ($argument in $pythonArguments) {
    '"' + $argument.Replace('"', '\"') + '"'
}
$argumentLine = $quotedArguments -join " "
$fullCommandLine = '"' + $pythonExecutable + '" ' + $argumentLine
$launchMetadata = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    command_line = $fullCommandLine
    config_path = $resolvedConfig
    policy_dir = $resolvedPolicyDir
    python_executable = $pythonExecutable
    git_commit = "no git"
    scope = "exact CW and full nonlinear CW + J2; CW-only shield drift; d_safe parameters 40 m and 67 m"
    blocked_environment = @{
        condition = "CW + J2 + drag"
        reason = "drag ballistic coefficients and atmosphere settings not sourced; PLACEHOLDER"
    }
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
    $parentPid = (Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)").ParentProcessId
} catch {
    Remove-Item Env:AEGIS_ORBIT_OUTPUT_RESERVATION -ErrorAction SilentlyContinue
    Write-Error "Failed to start comparison process: $($_.Exception.Message)"
    exit 1
}
Remove-Item Env:AEGIS_ORBIT_OUTPUT_RESERVATION -ErrorAction SilentlyContinue

$deadline = (Get-Date).AddSeconds(180)
$firstProgressLine = $null
$measurementLine = $null
while ((Get-Date) -lt $deadline -and -not $measurementLine) {
    Start-Sleep -Seconds 2
    if (Test-Path -LiteralPath $stdoutPath) {
        $lines = @(Get-Content -LiteralPath $stdoutPath -ErrorAction SilentlyContinue)
        if ($lines.Count -gt 0 -and -not $firstProgressLine) {
            $firstProgressLine = $lines[0]
        }
        $measurementLine = $lines | Where-Object { $_ -like "MEASURED first_10_physical_episodes=10 *" } | Select-Object -First 1
    }
    if ($child.HasExited -and -not $measurementLine) {
        break
    }
}

if ($firstProgressLine -and $measurementLine -and -not $child.HasExited) {
    $stderrLength = (Get-Item -LiteralPath $stderrPath).Length
    if ($stderrLength -ne 0) {
        Write-Error "FAIL PID=$($child.Id): stderr is not empty"
        Get-Content -LiteralPath $stderrPath -Tail 30
        exit 1
    }
    Write-Output "PASS PID=$($child.Id) PARENT_PID=$parentPid first_progress_line=$firstProgressLine"
    Write-Output $measurementLine
    exit 0
}

Write-Error "FAIL PID=$($child.Id): process did not reach the first-ten-episode measurement while running"
if (Test-Path -LiteralPath $stderrPath) {
    Get-Content -LiteralPath $stderrPath -Tail 30
}
exit 1
