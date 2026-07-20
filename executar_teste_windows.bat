@echo off
setlocal
cd /d %~dp0
PowerShell -ExecutionPolicy Bypass -File scripts\setup_vscode.ps1
if errorlevel 1 exit /b 1
PowerShell -ExecutionPolicy Bypass -File scripts\diagnostico_ambiente.ps1
pause
endlocal
