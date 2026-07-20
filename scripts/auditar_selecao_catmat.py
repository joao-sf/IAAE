from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

save_psv = import_module("src.reporting").save_psv


FAMILY_ALIASES = {
    "cabos_condutores": "Cabos Condutores",
    "disjuntores_chaves": "Disjuntores Chaves",
    "transformadores": "Transformadores",
    "medidores_energia": "Medidores Energia",
}

MEDIDOR_POSITIVE_TERMS = {
    "MEDIDOR": 20,
    "ENERGIA": 20,
    "ELETRICA": 10,
    "ELETRICO": 10,
    "ELETRONICO": 10,
    "MONOFASICO": 12,
    "BIFASICO": 12,
    "TRIFASICO": 12,
    "MULTIFUNCAO": 12,
    "WATTHORA": 15,
    "KWH": 15,
    "DIRETO": 8,
    "INDIRETO": 8,
}

MEDIDOR_PENALTY_TERMS = {
    "REATIVA": -18,
    "PORTATIL": -30,
    "ALICATE": -40,
    "ANALISADOR": -35,
    "TESTE": -25,
    "LABORATORIO": -25,
    "VAZAO": -60,
    "TEMPERATURA": -60,
    "PRESSAO": -60,
}

CODE_COLUMN_CANDIDATES = [
    "codigo",
    "codigo_item",
    "codigo_item_material",
    "codigo_material",
    "codigo_catmat",
    "catmat",
    "cod_item",
    "cod_material",
]

DESCRIPTION_COLUMN_CANDIDATES = [
    "descricao",
    "descricao_item",
    "descricao_material",
    "nome_item",
    "nome_material",
]

COLUMN_ALIASES = {
    "codigoitem": "codigo_item",
    "codigo_item_catmat": "codigo_catmat",
    "codigoitemmaterial": "codigo_item_material",
    "codigomaterial": "codigo_material",
    "descricaoitm": "descricao_item",
    "descricaoitem": "descricao_item",
    "descricaomaterial": "descricao_material",
    "codigopdm": "codigo_pdm",
    "nomepdm": "nome_pdm",
    "familia": "familia_material",
    "codigofamilia": "familia_codigo",
    "totalregistrosapi": "total_registros_api",
    "totalpaginasapi": "total_paginas_api",
    "scorecobertura": "score_cobertura",
    "statusconsulta": "status_consulta",
    "erroconsulta": "erro_consulta",
    "amostradataresultado": "amostra_data_resultado",
    "amostraunidade": "amostra_unidade",
    "amostrafornecedor": "amostra_fornecedor",
    "amostraestado": "amostra_estado",
    "amostradescricao": "amostra_descricao",
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(text.upper().split())


def normalize_column_name(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return COLUMN_ALIASES.get(text, text)


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]
    separators: list[str | None] = [
        ";",
        ",",
        "|",
        "\t",
        None,
    ]

    parsed: list[tuple[int, int, pd.DataFrame, str, str | None]] = []
    errors: list[str] = []

    for encoding in encodings:
        for separator in separators:
            options: dict[str, Any] = {
                "encoding": encoding,
                "low_memory": False,
            }

            if separator is None:
                options["sep"] = None
                options["engine"] = "python"
            else:
                options["sep"] = separator

            try:
                dataframe = pd.read_csv(path, **options)
                dataframe = normalize_columns(dataframe)

                non_empty_columns = sum(bool(str(column).strip()) for column in dataframe.columns)
                score = len(dataframe.columns)

                parsed.append(
                    (
                        score,
                        non_empty_columns,
                        dataframe,
                        encoding,
                        separator,
                    )
                )
            except Exception as exc:
                errors.append(f"encoding={encoding}, sep={separator!r}: {exc}")

    if not parsed:
        detail = "\n".join(errors[-5:])
        raise RuntimeError(f"Não foi possível ler {path}.\n{detail}")

    parsed.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    best = parsed[0]
    dataframe = best[2]

    if len(dataframe.columns) <= 1:
        raise ValueError(
            "O arquivo foi lido com apenas uma coluna. "
            f"Arquivo: {path.resolve()}. "
            f"Cabeçalho interpretado: {list(dataframe.columns)}"
        )

    return dataframe


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Não foi possível identificar a coluna de {label}. "
        f"Colunas encontradas: {list(dataframe.columns)}"
    )


def clean_code(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")


def candidate_family_name(
    dataframe: pd.DataFrame,
) -> pd.Series:
    if "familia_material" in dataframe.columns:
        family = dataframe["familia_material"].astype("string").str.strip()
    else:
        family = pd.Series(
            pd.NA,
            index=dataframe.index,
            dtype="string",
        )

    if "familia_codigo" in dataframe.columns:
        mapped = dataframe["familia_codigo"].astype("string").str.strip().map(FAMILY_ALIASES)
        family = family.fillna(mapped)

    return family


def score_medidor_description(
    description: Any,
) -> tuple[int, str, str]:
    normalized = normalize_text(description)
    score = 0
    positive_hits: list[str] = []
    penalty_hits: list[str] = []

    for term, points in MEDIDOR_POSITIVE_TERMS.items():
        if term in normalized:
            score += points
            positive_hits.append(term)

    for term, points in MEDIDOR_PENALTY_TERMS.items():
        if term in normalized:
            score += points
            penalty_hits.append(term)

    if "MEDIDOR" not in normalized:
        score -= 100

    return (
        score,
        ", ".join(positive_hits),
        ", ".join(penalty_hits),
    )


def diagnose_selected(
    fact: pd.DataFrame,
) -> pd.DataFrame:
    source = fact.copy()

    source["codigo_catmat"] = clean_code(source["codigo_catmat"])

    source["data_publicacao"] = pd.to_datetime(
        source["data_publicacao"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    source["quantidade"] = pd.to_numeric(
        source["quantidade"],
        errors="coerce",
    )

    description_column = next(
        (
            column
            for column in [
                "descricao_material",
                "descricao_item",
                "descricao",
            ]
            if column in source.columns
        ),
        None,
    )

    if description_column is None:
        source["descricao_referencia"] = pd.NA
    else:
        source["descricao_referencia"] = (
            source[description_column]
            .astype("string")
            .str.replace(
                r"[\r\n\t]+",
                " ",
                regex=True,
            )
            .str.replace(
                r"\s{2,}",
                " ",
                regex=True,
            )
            .str.strip()
        )

    supplier_column = next(
        (
            column
            for column in [
                "cnpj_fornecedor",
                "fornecedor_id",
                "fornecedor_key",
            ]
            if column in source.columns
        ),
        None,
    )

    uasg_column = next(
        (
            column
            for column in [
                "uasg",
                "codigo_uasg",
                "uasg_key",
            ]
            if column in source.columns
        ),
        None,
    )

    source["ano"] = source["data_publicacao"].dt.year
    source["mes_referencia"] = source["data_publicacao"].dt.to_period("M").astype("string")

    group_columns = [
        "familia_material",
        "codigo_catmat",
        "descricao_referencia",
    ]

    aggregation: dict[str, tuple[str, Any]] = {
        "registros": (
            "codigo_catmat",
            "size",
        ),
        "data_inicial": (
            "data_publicacao",
            "min",
        ),
        "data_final": (
            "data_publicacao",
            "max",
        ),
        "anos_ativos": (
            "ano",
            "nunique",
        ),
        "meses_ativos": (
            "mes_referencia",
            "nunique",
        ),
        "quantidade_total": (
            "quantidade",
            "sum",
        ),
        "unidades_comparaveis": (
            "unidade_comparavel",
            "nunique",
        ),
    }

    if supplier_column is not None:
        aggregation["fornecedores"] = (
            supplier_column,
            "nunique",
        )

    if uasg_column is not None:
        aggregation["uasgs"] = (
            uasg_column,
            "nunique",
        )

    result = source.groupby(
        group_columns,
        as_index=False,
        dropna=False,
    ).agg(**aggregation)

    start_period = result["data_inicial"].dt.to_period("M")
    end_period = result["data_final"].dt.to_period("M")

    result["cobertura_temporal_meses"] = [
        (end.ordinal - start.ordinal + 1 if pd.notna(start) and pd.notna(end) else np.nan)
        for start, end in zip(
            start_period,
            end_period,
            strict=False,
        )
    ]

    result["taxa_meses_ativos"] = np.where(
        result["cobertura_temporal_meses"] > 0,
        result["meses_ativos"] / result["cobertura_temporal_meses"],
        np.nan,
    )

    result["apto_previsao_catmat"] = (
        (result["registros"] >= 30)
        & (result["anos_ativos"] >= 3)
        & (result["meses_ativos"] >= 12)
        & (result["taxa_meses_ativos"] >= 0.25)
    )

    result["motivo_revisao"] = np.select(
        [
            result["registros"] < 30,
            result["anos_ativos"] < 3,
            result["meses_ativos"] < 12,
            result["taxa_meses_ativos"] < 0.25,
        ],
        [
            "Poucos registros no período",
            "Cobertura inferior a três anos",
            "Poucos meses com aquisição",
            "Série excessivamente esparsa",
        ],
        default="Apto para avaliação de previsão",
    )

    return result.sort_values(
        [
            "apto_previsao_catmat",
            "familia_material",
            "registros",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )


def prepare_catalog(
    dataframe: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    source = normalize_columns(dataframe)

    code_column = find_column(
        source,
        CODE_COLUMN_CANDIDATES,
        f"código no arquivo {source_label}",
    )

    if code_column != "codigo":
        source = source.rename(columns={code_column: "codigo"})

    description_column = next(
        (candidate for candidate in DESCRIPTION_COLUMN_CANDIDATES if candidate in source.columns),
        None,
    )

    if description_column is not None and description_column != "descricao":
        source = source.rename(columns={description_column: "descricao"})

    source["codigo"] = clean_code(source["codigo"])

    return source


def build_candidate_ranking(
    candidates: pd.DataFrame,
    coverage: pd.DataFrame | None,
) -> pd.DataFrame:
    source = prepare_catalog(
        candidates,
        "candidatos",
    )

    source["familia_material"] = candidate_family_name(source)

    if "descricao" not in source.columns:
        source["descricao"] = pd.NA

    if coverage is not None and not coverage.empty:
        coverage_source = prepare_catalog(
            coverage,
            "cobertura",
        )

        coverage_columns = [
            column
            for column in [
                "codigo",
                "total_registros_api",
                "total_paginas_api",
                "score_cobertura",
                "status_consulta",
                "erro_consulta",
                "amostra_data_resultado",
                "amostra_unidade",
                "amostra_fornecedor",
                "amostra_estado",
                "amostra_descricao",
            ]
            if column in coverage_source.columns
        ]

        source = source.merge(
            coverage_source[coverage_columns].drop_duplicates(
                subset=["codigo"],
                keep="last",
            ),
            on="codigo",
            how="left",
        )

    if "total_registros_api" not in source.columns:
        source["total_registros_api"] = 0

    source["total_registros_api"] = (
        pd.to_numeric(
            source["total_registros_api"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    scores = source["descricao"].apply(score_medidor_description)

    source[
        [
            "score_relevancia_tecnica",
            "termos_positivos",
            "termos_penalizados",
        ]
    ] = pd.DataFrame(
        scores.tolist(),
        index=source.index,
    )

    source["score_volume"] = (
        source["total_registros_api"]
        .apply(lambda value: math.log10(value + 1) * 10 if value > 0 else 0.0)
        .round(2)
    )

    source["score_preliminar"] = source["score_relevancia_tecnica"] + source["score_volume"]

    source["revisao_manual_obrigatoria"] = (source["familia_material"] == "Medidores Energia") & (
        (source["score_relevancia_tecnica"] < 35)
        | (source["termos_penalizados"].astype("string").str.len() > 0)
    )

    source["criterio_selecao"] = np.select(
        [
            source["total_registros_api"] == 0,
            source["score_relevancia_tecnica"] < 20,
            source["revisao_manual_obrigatoria"],
        ],
        [
            "Sem registros de preço",
            "Baixa aderência técnica",
            "Revisão técnica obrigatória",
        ],
        default="Candidato para auditoria temporal",
    )

    return source.sort_values(
        [
            "familia_material",
            "score_preliminar",
            "total_registros_api",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Audita os CATMATs selecionados e cria ranking preliminar dos candidatos.")
    )
    parser.add_argument(
        "--fact",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/silver/catmat_eletricos_candidatos.csv"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("data/silver/cobertura_catmat_eletricos.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gold/auditoria_selecao_catmat"),
    )
    args = parser.parse_args()

    if not args.fact.exists():
        raise SystemExit(f"Fato não encontrada: {args.fact.resolve()}")

    if not args.candidates.exists():
        raise SystemExit(f"Candidatos não encontrados: {args.candidates.resolve()}")

    fact = pd.read_parquet(args.fact)
    candidates = read_csv_flexible(args.candidates)
    coverage = read_csv_flexible(args.coverage) if args.coverage.exists() else None

    selected_diagnostic = diagnose_selected(fact)
    candidate_ranking = build_candidate_ranking(
        candidates,
        coverage,
    )

    medidor_candidates = (
        candidate_ranking.loc[candidate_ranking["familia_material"] == "Medidores Energia"]
        .head(100)
        .copy()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_psv(
        selected_diagnostic,
        args.output_dir / "diagnostico_catmat_selecionados.psv",
    )
    save_psv(
        medidor_candidates,
        args.output_dir / "candidatos_medidores_ranking.psv",
    )
    save_psv(
        candidate_ranking,
        args.output_dir / "ranking_todos_catmat.psv",
    )

    selected_diagnostic.to_parquet(
        args.output_dir / "diagnostico_catmat_selecionados.parquet",
        index=False,
    )

    print("Auditoria de seleção concluída.")
    print(
        "CATMATs atualmente selecionados:",
        selected_diagnostic["codigo_catmat"].nunique(),
    )
    print(
        "CATMATs não aptos à previsão:",
        int((~selected_diagnostic["apto_previsao_catmat"]).sum()),
    )
    print(
        "Candidatos de medidores avaliados:",
        len(medidor_candidates),
    )
    print()
    print("Resumo dos selecionados:")
    print(
        selected_diagnostic[
            [
                "familia_material",
                "codigo_catmat",
                "registros",
                "anos_ativos",
                "meses_ativos",
                "taxa_meses_ativos",
                "apto_previsao_catmat",
                "motivo_revisao",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Diretório:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
