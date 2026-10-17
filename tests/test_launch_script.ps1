$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcherPath = Join-Path $repositoryRoot "scripts\launch_retraining.ps1"
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("aegis-launch-refusal-" + [guid]::NewGuid().ToString("N"))
$outputDir = Join-Path $temporaryRoot "nonempty"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$sentinelPath = Join-Path $outputDir "sentinel.txt"
[System.IO.File]::WriteAllText($sentinelPath, "preserve")

try {
    $process = Start-Process `
        -FilePath (Get-Command powershell.exe).Source `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            ('"{0}"' -f $launcherPath),
            "-OutputDir",
            ('"{0}"' -f $outputDir)
        ) `
        -Wait `
        -PassThru `
        -WindowStyle Hidden
    if ($process.ExitCode -eq 0) {
        throw "Launcher unexpectedly accepted a non-empty output directory."
    }
    if ([System.IO.File]::ReadAllText($sentinelPath) -ne "preserve") {
        throw "Launcher modified the non-empty output directory."
    }
    Write-Output "PASS launcher refused non-empty output directory (exit=$($process.ExitCode))."
} finally {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
}
