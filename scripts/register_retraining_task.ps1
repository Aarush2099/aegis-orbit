<#
.SYNOPSIS
Registers a one-time scheduled-task fallback for detached retraining.
.DESCRIPTION
This script is not run automatically. It only registers a task when explicitly invoked.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [datetime]$StartTime,
    [string[]]$Methods = @("sac_primary"),
    [string]$Config = "configs\sac_unshielded.yaml",
    [string]$TaskName = "AegisOrbitPrimaryRetraining"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcherPath = Join-Path $PSScriptRoot "launch_retraining.ps1"
$powershellPath = (Get-Command powershell.exe).Source
$launcherArguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    ('"{0}"' -f $launcherPath),
    "-OutputDir",
    ('"{0}"' -f $OutputDir),
    "-Methods"
) + $Methods + @(
    "-Config",
    ('"{0}"' -f $Config)
)
$action = New-ScheduledTaskAction `
    -Execute $powershellPath `
    -Argument ($launcherArguments -join " ") `
    -WorkingDirectory $repositoryRoot
$trigger = New-ScheduledTaskTrigger -Once -At $StartTime
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "One-time detached AegisOrbit primary SAC retraining fallback." `
    -Force
