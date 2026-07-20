from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("resultado", "resultados", "dados", "items", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]

    embedded = payload.get("_embedded")
    if isinstance(embedded, dict):
        for value in embedded.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    return []


def clean_identifier(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def clean_text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def format_capacity(value: Any) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    numeric = float(numeric)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def build_comparable_unit(row: pd.Series) -> str | None:
    unit = row.get("sigla_unidade_fornecimento")
    if pd.isna(unit) or not str(unit).strip():
        unit = row.get("nome_unidade_fornecimento")
    if pd.isna(unit) or not str(unit).strip():
        unit = row.get("sigla_unidade_medida")
    if pd.isna(unit) or not str(unit).strip():
        return None

    unit = str(unit).strip().upper()
    capacity = format_capacity(row.get("capacidade_unidade_fornecimento"))
    if capacity and capacity != "1":
        return f"{unit}|CAP={capacity}"
    return unit


def load_bronze(input_dir: Path) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    invalid_files = 0

    for path in sorted(input_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except OSError, UnicodeDecodeError, json.JSONDecodeError:
            invalid_files += 1
            continue

        for record in extract_records(payload):
            rows.append(
                {
                    "id_item_compra_raw": clean_text_value(record.get("idItemCompra")),
                    "id_compra_item_raw": clean_text_value(record.get("idCompraItem")),
                    "id_compra_raw": clean_text_value(record.get("idCompra")),
                    "codigo_catmat_raw": record.get("codigoItemCatalogo"),
                    "sigla_unidade_fornecimento": clean_text_value(
                        record.get("siglaUnidadeFornecimento")
                    ),
                    "nome_unidade_fornecimento": clean_text_value(
                        record.get("nomeUnidadeFornecimento")
                    ),
                    "capacidade_unidade_fornecimento": record.get("capacidadeUnidadeFornecimento"),
                    "sigla_unidade_medida": clean_text_value(record.get("siglaUnidadeMedida")),
                    "nome_unidade_medida": clean_text_value(record.get("nomeUnidadeMedida")),
                    "arquivo_bronze": str(path),
                }
            )

    bronze = pd.DataFrame(rows)
    if bronze.empty:
        raise SystemExit(f"Nenhum registro foi encontrado em {input_dir.resolve()}")

    for column in ("id_item_compra_raw", "id_compra_item_raw", "id_compra_raw"):
        bronze[column] = clean_identifier(bronze[column])

    bronze["codigo_catmat_raw"] = pd.to_numeric(
        bronze["codigo_catmat_raw"], errors="coerce"
    ).astype("Int64")
    bronze["capacidade_unidade_fornecimento"] = pd.to_numeric(
        bronze["capacidade_unidade_fornecimento"], errors="coerce"
    )
    return bronze, invalid_files


def candidate_matches(fact: pd.DataFrame, bronze: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("id_item_compra", "id_item_compra_raw"),
        ("id_compra_item", "id_compra_item_raw"),
        ("purchase_id", "id_item_compra_raw"),
        ("purchase_id", "id_compra_item_raw"),
    ]
    rows: list[dict[str, Any]] = []

    for fact_column, bronze_column in candidates:
        if fact_column not in fact.columns:
            continue
        fact_values = clean_identifier(fact[fact_column])
        bronze_values = clean_identifier(bronze[bronze_column])
        fact_non_null = fact_values.dropna()
        bronze_non_null = bronze_values.dropna()
        if fact_non_null.empty or bronze_non_null.empty:
            continue

        bronze_set = set(bronze_non_null.tolist())
        matched = int(fact_non_null.isin(bronze_set).sum())
        rows.append(
            {
                "coluna_gold": fact_column,
                "coluna_bronze": bronze_column,
                "gold_preenchidos": len(fact_non_null),
                "bronze_preenchidos": len(bronze_non_null),
                "gold_unicos": int(fact_non_null.nunique()),
                "bronze_unicos": int(bronze_non_null.nunique()),
                "correspondencias": matched,
                "percentual_correspondencia": round(matched / len(fact_non_null) * 100, 2),
            }
        )

    if not rows:
        raise SystemExit("Nenhum par de identificadores compatível foi encontrado.")

    return pd.DataFrame(rows).sort_values(
        ["correspondencias", "percentual_correspondencia"],
        ascending=False,
    )


def deduplicate_bronze(bronze: pd.DataFrame, key: str) -> tuple[pd.DataFrame, int]:
    fields = [
        "sigla_unidade_fornecimento",
        "nome_unidade_fornecimento",
        "capacidade_unidade_fornecimento",
        "sigla_unidade_medida",
        "nome_unidade_medida",
    ]
    valid = bronze.loc[bronze[key].notna()].copy()
    conflict_count = 0

    for field in fields:
        distinct = valid.groupby(key, dropna=False)[field].nunique(dropna=True)
        conflict_count += int((distinct > 1).sum())

    valid = valid.sort_values(
        [key, "sigla_unidade_fornecimento", "capacidade_unidade_fornecimento"],
        na_position="last",
    ).drop_duplicates(subset=[key], keep="first")
    return valid, conflict_count


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="|", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fact",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=Path("data/bronze/precos_praticados"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gold/fact_compras_enriquecida.parquet"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/gold/eda_v2"),
    )
    args = parser.parse_args()

    if not args.fact.exists():
        raise SystemExit(f"Fato não encontrada: {args.fact.resolve()}")
    if not args.bronze_dir.exists():
        raise SystemExit(f"Bronze não encontrado: {args.bronze_dir.resolve()}")

    fact = pd.read_parquet(args.fact).copy()
    bronze, invalid_files = load_bronze(args.bronze_dir)

    matches = candidate_matches(fact, bronze)
    best = matches.iloc[0]
    fact_key = str(best["coluna_gold"])
    bronze_key = str(best["coluna_bronze"])

    fact[fact_key] = clean_identifier(fact[fact_key])
    bronze_lookup, conflicts = deduplicate_bronze(bronze, bronze_key)

    unit_fields = [
        "sigla_unidade_fornecimento",
        "nome_unidade_fornecimento",
        "capacidade_unidade_fornecimento",
        "sigla_unidade_medida",
        "nome_unidade_medida",
    ]

    existing_to_drop = [column for column in unit_fields if column in fact.columns]
    if existing_to_drop:
        fact = fact.drop(columns=existing_to_drop)

    lookup = bronze_lookup[[bronze_key, *unit_fields]].copy()
    enriched = fact.merge(
        lookup,
        left_on=fact_key,
        right_on=bronze_key,
        how="left",
        validate="one_to_one",
    )

    if bronze_key != fact_key:
        enriched = enriched.drop(columns=[bronze_key])

    enriched["unidade_comparavel"] = enriched.apply(build_comparable_unit, axis=1)
    enriched["unidade_fornecimento_informada"] = (
        enriched["sigla_unidade_fornecimento"].notna()
        | enriched["nome_unidade_fornecimento"].notna()
    )

    total = len(enriched)
    supplied = int(enriched["unidade_fornecimento_informada"].sum())
    comparable = int(enriched["unidade_comparavel"].notna().sum())
    capacities = int(enriched["capacidade_unidade_fornecimento"].notna().sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)

    audit = pd.DataFrame(
        [
            {"metrica": "registros_gold", "valor": total},
            {"metrica": "arquivos_bronze_invalidos", "valor": invalid_files},
            {"metrica": "chave_gold_escolhida", "valor": fact_key},
            {"metrica": "chave_bronze_escolhida", "valor": bronze_key},
            {"metrica": "correspondencias", "valor": int(best["correspondencias"])},
            {
                "metrica": "percentual_correspondencia",
                "valor": float(best["percentual_correspondencia"]),
            },
            {"metrica": "unidade_fornecimento_preenchida", "valor": supplied},
            {
                "metrica": "percentual_unidade_fornecimento",
                "valor": round(supplied / total * 100, 2) if total else 0,
            },
            {"metrica": "capacidade_preenchida", "valor": capacities},
            {"metrica": "unidade_comparavel_preenchida", "valor": comparable},
            {"metrica": "conflitos_de_unidade_por_chave", "valor": conflicts},
        ]
    )

    unmatched_columns = [
        column
        for column in [
            fact_key,
            "codigo_catmat",
            "descricao_material",
            "descricao_item",
            "data_publicacao",
            "quantidade",
            "valor_unitario",
        ]
        if column in enriched.columns
    ]
    unmatched = enriched.loc[enriched["unidade_comparavel"].isna(), unmatched_columns].copy()

    sample_columns = [
        column
        for column in [
            fact_key,
            "codigo_catmat",
            "descricao_material",
            "descricao_item",
            "sigla_unidade_fornecimento",
            "nome_unidade_fornecimento",
            "capacidade_unidade_fornecimento",
            "sigla_unidade_medida",
            "unidade_comparavel",
        ]
        if column in enriched.columns
    ]
    sample = enriched.loc[enriched["unidade_comparavel"].notna(), sample_columns].head(1000)

    save_csv(matches, args.report_dir / "diagnostico_chaves_enriquecimento.csv")
    save_csv(audit, args.report_dir / "auditoria_enriquecimento_unidades.csv")
    save_csv(unmatched, args.report_dir / "registros_sem_unidade_enriquecida.csv")
    save_csv(sample, args.report_dir / "amostra_unidades_enriquecidas.csv")

    print("Enriquecimento concluído.")
    print("Chave utilizada:", f"{fact_key} ↔ {bronze_key}")
    print(
        "Correspondências:",
        f"{int(best['correspondencias'])}/{total}",
        f"({float(best['percentual_correspondencia']):.2f}%)",
    )
    print(
        "Unidade de fornecimento preenchida:",
        f"{supplied}/{total}",
        f"({(supplied / total * 100):.2f}%)" if total else "(0%)",
    )
    print(
        "Capacidade preenchida:",
        f"{capacities}/{total}",
        f"({(capacities / total * 100):.2f}%)" if total else "(0%)",
    )
    print(
        "Unidade comparável preenchida:",
        f"{comparable}/{total}",
        f"({(comparable / total * 100):.2f}%)" if total else "(0%)",
    )
    print("Conflitos por chave:", conflicts)
    print()
    print(f"Fato enriquecida: {args.output.resolve()}")
    print(f"Relatórios: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
