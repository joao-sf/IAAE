@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo Ambiente .venv nao encontrado. Execute executar_teste_windows.bat primeiro.
  exit /b 1
)

.venv\Scripts\python.exe main.py run ^
  --start-date 2025-01-01 ^
  --end-date 2025-03-31 ^
  --max-materials 3 ^
  --catalog-file config\catmat_eletricos.csv

pause
endlocal
