from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_TABLES = [
    "FatoCompras.parquet",
    "FatoPrevisao.parquet",
    "DimCalendario.parquet",
    "DimMaterial.parquet",
    "DimFornecedor.parquet",
    "DimUASG.parquet",
    "DimUnidade.parquet",
    "DimCenario.parquet",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_parquet(path: Path) -> dict[str, Any]:
    dataframe = pd.read_parquet(path)

    return {
        "arquivo": path.name,
        "linhas": len(dataframe),
        "colunas": len(dataframe.columns),
        "hash_sha256": sha256_file(path),
    }


def save_psv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe.to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def restore_backup(
    backup_dir: Path,
    production_dir: Path,
    published_files: list[Path],
) -> None:
    for published in published_files:
        if published.exists():
            published.unlink()

    for backup_file in backup_dir.glob("*"):
        if backup_file.is_file():
            shutil.copy2(
                backup_file,
                production_dir / backup_file.name,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publica as tabelas Power BI aprovadas no staging, "
            "com backup, validação e rollback automático."
        )
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("data/staging_powerbi/integracao_medidores"),
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("data/powerbi"),
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("data/backups/powerbi"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/staging_powerbi/publicacao_medidores"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Valida staging e produção sem substituir arquivos."),
    )
    args = parser.parse_args()

    staging_dir = args.staging_root / "powerbi"
    manifest_path = args.staging_root / "manifesto_integracao_medidores.json"

    if not manifest_path.exists():
        raise SystemExit(f"Manifesto não encontrado: {manifest_path.resolve()}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("aprovado") is not True:
        raise SystemExit("Publicação bloqueada: o staging não está aprovado.")

    if not staging_dir.exists():
        raise SystemExit(f"Diretório do staging não encontrado: {staging_dir.resolve()}")

    missing_required = [
        file_name for file_name in REQUIRED_TABLES if not (staging_dir / file_name).exists()
    ]

    if missing_required:
        raise SystemExit(
            "Arquivos obrigatórios ausentes no staging:\n" + "\n".join(missing_required)
        )

    staging_files = sorted(staging_dir.glob("*.parquet"))

    if not staging_files:
        raise SystemExit("Nenhum Parquet foi encontrado no staging.")

    staging_validation = pd.DataFrame([validate_parquet(path) for path in staging_files])
    staging_validation["origem"] = "STAGING"

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    save_psv(
        staging_validation,
        args.report_dir / "validacao_pre_publicacao.psv",
    )

    print("Staging validado.")
    print(
        "Arquivos Parquet:",
        len(staging_files),
    )
    print(
        "Manifesto aprovado:",
        manifest.get("aprovado"),
    )

    if args.dry_run:
        print("Modo dry-run: nenhum arquivo foi alterado.")
        print(
            "Relatório:",
            (args.report_dir / "validacao_pre_publicacao.psv").resolve(),
        )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_root / f"powerbi_{timestamp}"
    temp_dir = args.production_dir.parent / f"{args.production_dir.name}_publish_tmp"

    args.production_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for current_file in args.production_dir.glob("*"):
        if current_file.is_file():
            shutil.copy2(
                current_file,
                backup_dir / current_file.name,
            )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for staging_file in staging_files:
        shutil.copy2(
            staging_file,
            temp_dir / staging_file.name,
        )

    temp_validation = pd.DataFrame(
        [validate_parquet(path) for path in sorted(temp_dir.glob("*.parquet"))]
    )

    staging_hashes = {
        row["arquivo"]: row["hash_sha256"] for row in staging_validation.to_dict(orient="records")
    }
    temp_hashes = {
        row["arquivo"]: row["hash_sha256"] for row in temp_validation.to_dict(orient="records")
    }

    if staging_hashes != temp_hashes:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )
        raise SystemExit("Falha na validação da cópia temporária. A produção não foi alterada.")

    published_files: list[Path] = []

    try:
        for staging_file in staging_files:
            destination = args.production_dir / staging_file.name
            source = temp_dir / staging_file.name

            shutil.copy2(
                source,
                destination,
            )
            published_files.append(destination)

        production_validation = pd.DataFrame(
            [
                validate_parquet(path)
                for path in sorted(args.production_dir.glob("*.parquet"))
                if path.name in staging_hashes
            ]
        )
        production_validation["origem"] = "PRODUCAO"

        production_hashes = {
            row["arquivo"]: row["hash_sha256"]
            for row in production_validation.to_dict(orient="records")
        }

        if production_hashes != staging_hashes:
            raise RuntimeError(
                "Os hashes da produção não correspondem aos arquivos aprovados do staging."
            )

    except Exception as exc:
        restore_backup(
            backup_dir,
            args.production_dir,
            published_files,
        )
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )
        raise SystemExit(
            "Falha durante a publicação. "
            "Backup restaurado automaticamente.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    final_report = staging_validation.merge(
        production_validation[
            [
                "arquivo",
                "linhas",
                "colunas",
                "hash_sha256",
            ]
        ],
        on="arquivo",
        how="outer",
        suffixes=(
            "_staging",
            "_producao",
        ),
        validate="one_to_one",
    )

    final_report["hash_igual"] = (
        final_report["hash_sha256_staging"] == final_report["hash_sha256_producao"]
    )
    final_report["linhas_iguais"] = (
        final_report["linhas_staging"] == final_report["linhas_producao"]
    )
    final_report["colunas_iguais"] = (
        final_report["colunas_staging"] == final_report["colunas_producao"]
    )

    save_psv(
        final_report,
        args.report_dir / "validacao_pos_publicacao.psv",
    )

    publication_manifest = {
        "publicado_em": timestamp,
        "staging_root": str(args.staging_root.resolve()),
        "production_dir": str(args.production_dir.resolve()),
        "backup_dir": str(backup_dir.resolve()),
        "arquivos_publicados": [path.name for path in staging_files],
        "hashes_validos": bool(final_report["hash_igual"].all()),
        "linhas_validas": bool(final_report["linhas_iguais"].all()),
        "colunas_validas": bool(final_report["colunas_iguais"].all()),
        "publicacao_aprovada": bool(
            final_report[
                [
                    "hash_igual",
                    "linhas_iguais",
                    "colunas_iguais",
                ]
            ]
            .all()
            .all()
        ),
    }

    (args.report_dir / "manifesto_publicacao.json").write_text(
        json.dumps(
            publication_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Publicação concluída.")
    print(
        "Arquivos publicados:",
        len(staging_files),
    )
    print(
        "Backup:",
        backup_dir.resolve(),
    )
    print(
        "Hashes válidos:",
        publication_manifest["hashes_validos"],
    )
    print(
        "Publicação aprovada:",
        publication_manifest["publicacao_aprovada"],
    )
    print(
        "Relatório:",
        (args.report_dir / "validacao_pos_publicacao.psv").resolve(),
    )

    if not publication_manifest["publicacao_aprovada"]:
        raise SystemExit("Publicação finalizada com validações pendentes.")


if __name__ == "__main__":
    main()
