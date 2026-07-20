$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) não encontrado. Confirme a instalação do Python 3.14.6."
}

$version = & py -3.14 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.14 não foi localizado pelo Python Launcher. Execute: py -0p"
}

Write-Host "Python localizado: $version"
if (-not $version.StartsWith("3.14.")) {
    throw "A versão localizada não pertence à série 3.14: $version"
}

if (Test-Path ".venv") {
    $venvVersion = & .\.venv\Scripts\python.exe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
    if ($venvVersion -and -not $venvVersion.StartsWith("3.14.")) {
        Write-Host "Removendo .venv criado com Python $venvVersion..."
        Remove-Item ".venv" -Recurse -Force
    }
}

if (-not (Test-Path ".venv")) {
    & py -3.14 -m venv .venv
}

$python = Join-Path $PWD ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install -e ".[dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Arquivo .env criado a partir de .env.example."
}

& $python -m pytest
& $python -c "import pandas, pyarrow, duckdb; print('pandas:', pandas.__version__); print('pyarrow:', pyarrow.__version__); print('duckdb:', duckdb.__version__)"
Write-Host "`nAmbiente IAAE preparado com Python $version."
