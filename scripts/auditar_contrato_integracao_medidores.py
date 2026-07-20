from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ALIASES: dict[str, list[str]] = {
    "codigo_catmat": [
        "codigo_catmat",
        "codigocatmat",
        "codigo",
        "codigo_item_catalogo",
        "codigoitemcatalogo",
        "catmat",
    ],
    "data": [
        "data",
        "data_referencia",
        "dataresultado",
        "data_resultado",
        "data_compra",
        "datacompra",
        "inicio_mes",
    ],
    "quantidade": [
        "quantidade",
        "quantidade_item",
        "quantidadeitem",
        "quantidade_prevista",
        "quantidadeprevista",
    ],
    "cenario": [
        "cenario",
        "cenario_key",
        "cenariokey",
        "nome_cenario",
        "nomecenario",
    ],
    "familia_codigo": [
        "familia_codigo",
        "codigofamilia",
        "codigo_familia",
    ],
    "familia_material": [
        "familia_material",
        "familiamaterial",
        "familia",
    ],
    "descricao": [
        "descricao",
        "descricao_catalogo",
        "descricaocatalogo",
        "descricao_material",
        "descricaomaterial",
        "descricao_item",
        "descricaoitem",
    ],
    "fornecedor_codigo": [
        "ni_fornecedor",
        "nifornecedor",
        "codigo_fornecedor",
        "codigofornecedor",
        "id_fornecedor",
        "idfornecedor",
        "fornecedor_id",
        "fornecedorid",
    ],
    "fornecedor_nome": [
        "nome_fornecedor",
        "nomefornecedor",
        "fornecedor",
    ],
    "uasg_codigo": [
        "codigo_uasg",
        "codigouasg",
        "uasg",
        "id_uasg",
        "iduasg",
    ],
    "uasg_nome": [
        "nome_uasg",
        "nomeuasg",
    ],
    "unidade_codigo": [
        "sigla_unidade_fornecimento",
        "siglaunidadefornecimento",
        "unidade_padronizada",
        "unidadepadronizada",
        "codigo_unidade",
        "codigounidade",
        "unidade",
    ],
    "unidade_nome": [
        "nome_unidade_fornecimento",
        "nomeunidadefornecimento",
        "descricao_unidade",
        "descricaounidade",
    ],
    "preco_unitario": [
        "preco_unitario",
        "precounitario",
        "preco_unitario_comparavel",
        "precounitariocomparavel",
    ],
    "valor_total": [
        "valor_total",
        "valortotal",
        "valor_total_calculado",
        "valortotalcalculado",
        "valor_total_estimado",
        "valortotalestimado",
    ],
    "modelo": [
        "modelo_selecionado",
        "modeloselecionado",
        "modelo",
    ],
    "confianca": [
        "confianca_previsao",
        "confiancaprevisao",
        "confianca",
    ],
}


REQUIRED_BY_TARGET = {
    "FatoCompras": [
        "codigo_catmat",
        "data",
        "quantidade",
    ],
    "FatoPrevisao": [
        "codigo_catmat",
        "data",
        "cenario",
        "quantidade",
    ],
    "DimMaterial": [
        "codigo_catmat",
        "descricao",
        "familia_material",
    ],
}


def normalize_header(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def save_psv(dataframe: pd.DataFrame, path: Path) -> None:
    output = dataframe.copy()

    for column in output.select_dtypes(include=["object", "string"]).columns:
        output[column] = (
            output[column]
            .astype("string")
            .str.replace(r"[\r\n\t]+", " ", regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.strip()
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def locate_concept(
    dataframe: pd.DataFrame,
    concept: str,
) -> str | None:
    normalized_to_original = {normalize_header(column): str(column) for column in dataframe.columns}

    for alias in ALIASES[concept]:
        normalized_alias = normalize_header(alias)
        if normalized_alias in normalized_to_original:
            return normalized_to_original[normalized_alias]

    return None


def table_inventory(
    name: str,
    path: Path,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    date_column = locate_concept(
        dataframe,
        "data",
    )

    minimum_date = None
    maximum_date = None

    if date_column is not None:
        dates = pd.to_datetime(
            dataframe[date_column],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

        if dates.notna().any():
            minimum_date = dates.min()
            maximum_date = dates.max()

    return {
        "tabela": name,
        "arquivo": str(path.resolve()),
        "linhas": len(dataframe),
        "colunas": len(dataframe.columns),
        "duplicados_exatos": int(dataframe.duplicated().sum()),
        "data_minima": minimum_date,
        "data_maxima": maximum_date,
    }


def schema_rows(
    name: str,
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for position, column in enumerate(
        dataframe.columns,
        start=1,
    ):
        series = dataframe[column]

        rows.append(
            {
                "tabela": name,
                "posicao": position,
                "coluna": str(column),
                "coluna_normalizada": (normalize_header(column)),
                "tipo": str(series.dtype),
                "nulos": int(series.isna().sum()),
                "percentual_nulos": (float(series.isna().mean()) if len(series) else 0.0),
                "valores_distintos": int(series.nunique(dropna=True)),
                "exemplo": (str(series.dropna().iloc[0])[:200] if series.notna().any() else None),
            }
        )

    return rows


def detected_key_rows(
    name: str,
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for concept in ALIASES:
        column = locate_concept(
            dataframe,
            concept,
        )

        rows.append(
            {
                "tabela": name,
                "conceito": concept,
                "coluna_detectada": column,
                "detectado": column is not None,
            }
        )

    return rows


def compatibility_rows(
    target_name: str,
    target: pd.DataFrame,
    source_name: str,
    source: pd.DataFrame,
) -> list[dict[str, Any]]:
    source_normalized = {normalize_header(column): str(column) for column in source.columns}

    rows: list[dict[str, Any]] = []

    for target_column in target.columns:
        normalized_target = normalize_header(target_column)
        source_column = source_normalized.get(normalized_target)
        strategy = "MESMO_NOME_NORMALIZADO"

        if source_column is None:
            matched_concept = None

            for concept, aliases in ALIASES.items():
                normalized_aliases = {normalize_header(alias) for alias in aliases}

                if normalized_target in normalized_aliases:
                    matched_concept = concept
                    break

            if matched_concept is not None:
                source_column = locate_concept(
                    source,
                    matched_concept,
                )
                strategy = f"ALIAS_{matched_concept.upper()}"

        if source_column is None:
            strategy = "SEM_MAPEAMENTO"

        rows.append(
            {
                "tabela_destino": target_name,
                "coluna_destino": str(target_column),
                "tipo_destino": str(target[target_column].dtype),
                "fonte": source_name,
                "coluna_fonte": source_column,
                "estrategia": strategy,
                "mapeada": source_column is not None,
            }
        )

    return rows


def required_validation_rows(
    name: str,
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for concept in REQUIRED_BY_TARGET.get(
        name,
        [],
    ):
        column = locate_concept(
            dataframe,
            concept,
        )

        rows.append(
            {
                "tabela": name,
                "conceito_obrigatorio": concept,
                "coluna_detectada": column,
                "resultado": ("OK" if column is not None else "AUSENTE"),
            }
        )

    return rows


def read_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{label} não encontrado: {path.resolve()}")

    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Audita o contrato das tabelas do Power BI antes da integração dos medidores.")
    )
    parser.add_argument(
        "--powerbi-dir",
        type=Path,
        default=Path("data/powerbi"),
    )
    parser.add_argument(
        "--pilot-purchases",
        type=Path,
        default=Path("data/pilot_medidores/silver/compras_medidores_piloto.parquet"),
    )
    parser.add_argument(
        "--pilot-forecast",
        type=Path,
        default=Path("data/pilot_medidores/previsao/previsao_catmat_long.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/staging_powerbi/auditoria_contrato_medidores"),
    )
    args = parser.parse_args()

    expected_powerbi_tables = [
        "FatoCompras",
        "FatoPrevisao",
        "DimMaterial",
        "DimFornecedor",
        "DimUASG",
        "DimUnidade",
        "DimCalendario",
        "DimCenario",
    ]

    tables: dict[str, tuple[Path, pd.DataFrame]] = {}

    for table_name in expected_powerbi_tables:
        path = args.powerbi_dir / f"{table_name}.parquet"

        if path.exists():
            tables[table_name] = (
                path,
                pd.read_parquet(path),
            )

    if "FatoCompras" not in tables:
        raise SystemExit(f"FatoCompras.parquet não foi encontrada em {args.powerbi_dir.resolve()}.")

    if "FatoPrevisao" not in tables:
        raise SystemExit(
            f"FatoPrevisao.parquet não foi encontrada em {args.powerbi_dir.resolve()}."
        )

    if "DimMaterial" not in tables:
        raise SystemExit(f"DimMaterial.parquet não foi encontrada em {args.powerbi_dir.resolve()}.")

    pilot_purchases = read_parquet(
        args.pilot_purchases,
        "Histórico piloto",
    )
    pilot_forecast = read_parquet(
        args.pilot_forecast,
        "Previsão piloto",
    )

    tables["PilotoCompras"] = (
        args.pilot_purchases,
        pilot_purchases,
    )
    tables["PilotoPrevisao"] = (
        args.pilot_forecast,
        pilot_forecast,
    )

    inventory: list[dict[str, Any]] = []
    schema: list[dict[str, Any]] = []
    detected_keys: list[dict[str, Any]] = []
    required_checks: list[dict[str, Any]] = []

    for name, (path, dataframe) in tables.items():
        inventory.append(
            table_inventory(
                name,
                path,
                dataframe,
            )
        )
        schema.extend(
            schema_rows(
                name,
                dataframe,
            )
        )
        detected_keys.extend(
            detected_key_rows(
                name,
                dataframe,
            )
        )
        required_checks.extend(
            required_validation_rows(
                name,
                dataframe,
            )
        )

    compatibility: list[dict[str, Any]] = []
    compatibility.extend(
        compatibility_rows(
            "FatoCompras",
            tables["FatoCompras"][1],
            "PilotoCompras",
            pilot_purchases,
        )
    )
    compatibility.extend(
        compatibility_rows(
            "FatoPrevisao",
            tables["FatoPrevisao"][1],
            "PilotoPrevisao",
            pilot_forecast,
        )
    )
    compatibility.extend(
        compatibility_rows(
            "DimMaterial",
            tables["DimMaterial"][1],
            "PilotoCompras",
            pilot_purchases,
        )
    )

    outputs = {
        "inventario_tabelas": pd.DataFrame(inventory),
        "esquema_colunas": pd.DataFrame(schema),
        "chaves_detectadas": pd.DataFrame(detected_keys),
        "campos_obrigatorios": pd.DataFrame(required_checks),
        "compatibilidade_integracao": pd.DataFrame(compatibility),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, dataframe in outputs.items():
        dataframe.to_parquet(
            args.output_dir / f"{name}.parquet",
            index=False,
        )
        save_psv(
            dataframe,
            args.output_dir / f"{name}.psv",
        )

    compatibility_frame = outputs["compatibilidade_integracao"]
    unmapped = compatibility_frame.loc[~compatibility_frame["mapeada"]]

    required_frame = outputs["campos_obrigatorios"]
    missing_required = required_frame.loc[required_frame["resultado"] != "OK"]

    print("Auditoria do contrato concluída.")
    print(
        "Tabelas avaliadas:",
        len(tables),
    )
    print(
        "Colunas sem mapeamento automático:",
        len(unmapped),
    )
    print(
        "Conceitos obrigatórios ausentes:",
        len(missing_required),
    )
    print()

    for target_name in [
        "FatoCompras",
        "FatoPrevisao",
        "DimMaterial",
    ]:
        target_rows = compatibility_frame.loc[compatibility_frame["tabela_destino"] == target_name]
        mapped = int(target_rows["mapeada"].sum())

        print(f"{target_name}: {mapped}/{len(target_rows)} colunas mapeadas automaticamente")

    print()
    print(
        "Diretório:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
