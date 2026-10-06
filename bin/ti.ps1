<#
    Launch the Tweet Index shell from anywhere.

    Typing plain `ti` finds this file in PowerShell: PowerShell looks for
    .ps1 on PATH itself, even though PATHEXT does not list it (measured --
    PATHEXT here is .COM;.EXE;.BAT;.CMD;... and `Get-Command ti` still
    resolves to this script). The .cmd next to it is what makes plain `ti`
    work in cmd.exe, which goes by PATHEXT alone. Both are shipped so the
    command behaves the same in either shell.

    The project folder comes from this file's own location (bin\..), so
    moving the project needs no edit.

    It calls the virtual environment's python.exe directly instead of
    dot-sourcing Activate.ps1: that is all activation does, it cannot be
    blocked by the execution policy, and it leaves the window's environment
    untouched afterwards.
#>

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'ti.py'

# Saved tweets contain emoji, Japanese and Turkish. Without this, Python
# falls back to the locale codepage the moment output is redirected and
# printing a tweet raises UnicodeEncodeError. ti.py also sets it on itself.
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path $python)) {
    Write-Host "Tweet Index: no virtual environment at" -ForegroundColor Yellow
    Write-Host "  $python"
    Write-Host ""
    Write-Host "Create one with:"
    Write-Host "  cd `"$root`""
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path $script)) {
    Write-Host "Tweet Index: ti.py is missing from $root" -ForegroundColor Yellow
    exit 1
}

& $python $script @args
exit $LASTEXITCODE
