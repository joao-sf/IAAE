$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "=== IAAE | Diagnóstico do ambiente ==="
Write-Host "Pasta: $PWD"
Write-Host "VS Code:"
code --version
Write-Host "`nPython Launcher:"
py -0p
Write-Host "`nPython 3.14:"
py -3.14 --version
Write-Host "`nGit:"
git --version

if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "`nAmbiente virtual:"
    & .\.venv\Scripts\python.exe --version
    & .\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
    Write-Host "`nDependências:"
    & .\.venv\Scripts\python.exe -c "import pandas, pyarrow, duckdb, yaml, requests; print('pandas', pandas.__version__); print('pyarrow', pyarrow.__version__); print('duckdb', duckdb.__version__); print('PyYAML', yaml.__version__); print('requests', requests.__version__)"
} else {
    Write-Host "`n.venv ainda não existe. Execute scripts\setup_vscode.ps1."
}
