@echo off
rem Launch the Tweet Index shell from anywhere.
rem
rem Works in PowerShell and in cmd.exe. Put this folder on PATH and the
rem command is just: ti
rem
rem The project folder is worked out from where this file sits (bin\..), so
rem moving or renaming the project needs no edit here.
rem
rem It calls the virtual environment's own python.exe rather than running
rem Activate.ps1 first. That is what activation does -- puts that python on
rem PATH -- and calling it directly avoids the PowerShell execution policy
rem blocking the activation script, and leaves no modified environment behind
rem in the window afterwards.

setlocal
set "TI_ROOT=%~dp0.."
set "TI_PY=%TI_ROOT%\.venv\Scripts\python.exe"

rem Saved tweets contain emoji, Japanese and Turkish. Without this, Python
rem falls back to the locale codepage (cp1252 here) the moment output is
rem redirected, and printing a tweet raises UnicodeEncodeError. ti.py sets
rem this on itself too; belt and braces, and it covers anything it starts.
set "PYTHONIOENCODING=utf-8"

if not exist "%TI_PY%" (
  echo Tweet Index: no virtual environment at
  echo   %TI_PY%
  echo.
  echo Create one with:
  echo   cd /d "%TI_ROOT%"
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

if not exist "%TI_ROOT%\ti.py" (
  echo Tweet Index: ti.py is missing from "%TI_ROOT%".
  exit /b 1
)

"%TI_PY%" "%TI_ROOT%\ti.py" %*
exit /b %ERRORLEVEL%
