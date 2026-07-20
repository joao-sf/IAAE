from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

FIELDS = [
    "siglaUnidadeFornecimento",
    "nomeUnidadeFornecimento",
    "capacidadeUnidadeFornecimento",
    "siglaUnidadeMedida",
    "nomeUnidadeMedida",
]


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """Extrai registros dos formatos usuais da API Compras.gov.br."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in (
        "resultado",
        "resultados",
        "dados",
        "items",
        "content",
        "_embedded",
    ):
        value = payload.get(key)

        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

        if isinstance(value, dict):
            for nested_value in value.values():
                if isinstance(nested_value, list):
                    return [item for item in nested_value if isinstance(item, dict)]

    return []


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"none", "nan", "null"}:
        return None

    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audita a cobertura de unidade de fornecimento e medida "
            "nos JSONs Bronze de preços praticados."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/bronze/precos_praticados"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gold/eda_v2/auditoria_unidades_bronze.csv"),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/gold/eda_v2/amostra_unidades_bronze.csv"),
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Diretório não encontrado: {args.input_dir.resolve()}")

    json_files = sorted(args.input_dir.rglob("*.json"))

    if not json_files:
        raise SystemExit(f"Nenhum JSON encontrado em: {args.input_dir.resolve()}")

    counters = {
        field: {
            "preenchidos": 0,
            "valores": set(),
        }
        for field in FIELDS
    }

    total_records = 0
    invalid_files = 0
    sample_rows: list[dict[str, Any]] = []

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except OSError, UnicodeDecodeError, json.JSONDecodeError:
            invalid_files += 1
            continue

        records = extract_records(payload)

        for record in records:
            total_records += 1

            row = {
                "arquivo": str(json_path),
                "codigo_catmat": record.get("codigoItemCatalogo"),
                "descricao": record.get("descricaoItem"),
                "data_resultado": record.get("dataResultado"),
            }

            has_unit_data = False

            for field in FIELDS:
                value = normalize_value(record.get(field))
                row[field] = value

                if value is not None:
                    counters[field]["preenchidos"] += 1
                    counters[field]["valores"].add(value)
                    has_unit_data = True

            if has_unit_data and len(sample_rows) < 500:
                sample_rows.append(row)

    audit_rows: list[dict[str, Any]] = []

    for field in FIELDS:
        filled = int(counters[field]["preenchidos"])
        missing = total_records - filled
        distinct_values = sorted(counters[field]["valores"])

        audit_rows.append(
            {
                "campo_api": field,
                "registros_totais": total_records,
                "preenchidos": filled,
                "ausentes": missing,
                "percentual_preenchido": (
                    round(filled / total_records * 100, 2) if total_records else 0.0
                ),
                "valores_distintos": len(distinct_values),
                "exemplos": " ; ".join(distinct_values[:15]),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        "preenchidos",
        ascending=False,
    )

    samples = pd.DataFrame(sample_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    audit.to_csv(
        args.output,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )

    samples.to_csv(
        args.sample_output,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )

    print("Auditoria Bronze concluída.")
    print("Arquivos JSON:", len(json_files))
    print("Arquivos inválidos:", invalid_files)
    print("Registros encontrados:", total_records)
    print()
    print("Cobertura dos campos:")
    for row in audit.to_dict(orient="records"):
        print(f"- {row['campo_api']}: {row['preenchidos']} ({row['percentual_preenchido']}%)")
    print()
    print(f"Relatório: {args.output.resolve()}")
    print(f"Amostra: {args.sample_output.resolve()}")


if __name__ == "__main__":
    main()
