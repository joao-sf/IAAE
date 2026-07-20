$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $PWD ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Ambiente .venv não encontrado. Execute scripts\setup_vscode.ps1 primeiro."
}

& $python -m ruff check .
& $python -m ruff format --check .
& $python -m pytest --cov=src --cov-report=term-missing
