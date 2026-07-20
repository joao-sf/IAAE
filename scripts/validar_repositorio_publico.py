from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAMES = {
    ".env",
    "modelo_powerbi_temp",
}
FORBIDDEN_SUFFIXES = {".zip", ".bak", ".tmp"}
FORBIDDEN_FRAGMENTS = {
    "_Pre_UF.pbix",
    "integrar_medidores_staging_powerbi_backup.py",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".ps1",
    ".bat",
    ".txt",
    ".csv",
}
PATTERNS = {
    "caminho local absoluto": re.compile(r"[A-Za-z]:[\\/](?:DEV|dev)[\\/]IAAE"),
    "referência ao case corporativo": re.compile(r"\bCPFL\b", re.IGNORECASE),
}
REQUIRED = {
    Path("README.md"),
    Path("LICENSE"),
    Path("pyproject.toml"),
    Path(".env.example"),
    Path("dashboard/IAAE_Dashboard_Final.pbix"),
    Path("docs/ARQUITETURA.md"),
    Path("docs/METODOLOGIA_PREVISAO.md"),
    Path("docs/POWER_BI_PORTABILIDADE.md"),
    Path("tests/test_estado_powerbi.py"),
}


def main() -> int:
    problems: list[str] = []

    for required in sorted(REQUIRED):
        if not (ROOT / required).exists():
            problems.append(f"Arquivo obrigatório ausente: {required}")

    skipped_parts = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"}

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in skipped_parts for part in relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            problems.append(f"Item local proibido: {relative}")
            continue
        if path.is_file():
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                problems.append(f"Extensão proibida: {relative}")
            if any(fragment in path.name for fragment in FORBIDDEN_FRAGMENTS):
                problems.append(f"Arquivo obsoleto: {relative}")
            if path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for label, pattern in PATTERNS.items():
                    if pattern.search(text):
                        problems.append(f"{label}: {relative}")

    pbix = ROOT / "dashboard/IAAE_Dashboard_Final.pbix"
    if pbix.exists() and pbix.stat().st_size == 0:
        problems.append("Dashboard final está vazio.")

    if problems:
        print("Repositório público reprovado:")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1

    print("Repositório público aprovado.")
    print(f"Raiz: {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
