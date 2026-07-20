from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def first_existing(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def read_mapping(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )
    dataframe.columns = [normalize_header(column) for column in dataframe.columns]

    required = {
        "codigo_catmat",
        "status_modelagem",
        "nivel_modelagem",
        "unidade_modelagem",
        "segmento_modelagem",
        "metodo_rateio",
    }
    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError("Colunas ausentes no mapeamento: " + ", ".join(sorted(missing)))

    dataframe["codigo_catmat"] = pd.to_numeric(
        dataframe["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    return dataframe


def prepare_fact(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    fact = source.copy()
    fact.columns = [normalize_header(column) for column in fact.columns]

    code_column = first_existing(
        fact,
        [
            "codigo_catmat",
            "codigo",
            "codigo_item_catalogo",
        ],
    )
    date_column = first_existing(
        fact,
        [
            "data_referencia",
            "data_resultado",
            "data_publicacao",
            "data_compra",
            "data",
        ],
    )
    quantity_column = first_existing(
        fact,
        [
            "quantidade",
            "quantidade_item",
        ],
    )
    purchase_column = first_existing(
        fact,
        [
            "id_compra",
            "idcompra",
            "id_item_compra",
        ],
    )

    if code_column is None:
        raise ValueError("Coluna de CATMAT não encontrada na fato piloto.")
    if date_column is None:
        raise ValueError("Coluna de data não encontrada na fato piloto.")
    if quantity_column is None:
        raise ValueError("Coluna de quantidade não encontrada na fato piloto.")

    rename_map = {
        code_column: "codigo_catmat",
        date_column: "data_referencia",
        quantity_column: "quantidade",
    }
    if purchase_column is not None:
        rename_map[purchase_column] = "id_compra"

    fact = fact.rename(columns=rename_map)

    if "id_compra" not in fact.columns:
        fact["id_compra"] = fact.index.astype(str)

    fact["codigo_catmat"] = pd.to_numeric(
        fact["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    fact["data_referencia"] = pd.to_datetime(
        fact["data_referencia"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    fact["quantidade"] = pd.to_numeric(
        fact["quantidade"],
        errors="coerce",
    )

    active_mapping = mapping.loc[
        mapping["status_modelagem"].astype(str).str.upper().eq("ATIVO")
    ].copy()

    fact = fact.merge(
        active_mapping,
        on="codigo_catmat",
        how="inner",
        validate="many_to_one",
    )

    fact = fact.loc[
        fact["data_referencia"].between(
            start_date,
            end_date,
            inclusive="both",
        )
        & fact["quantidade"].notna()
        & (fact["quantidade"] >= 0)
    ].copy()

    fact["inicio_mes"] = fact["data_referencia"].dt.to_period("M").dt.to_timestamp()

    return fact


def complete_monthly_series(
    fact: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    months = pd.date_range(
        start_date.to_period("M").to_timestamp(),
        end_date.to_period("M").to_timestamp(),
        freq="MS",
    )

    rows: list[pd.DataFrame] = []

    for code, group in fact.groupby(
        "codigo_catmat",
        dropna=False,
    ):
        monthly = (
            group.groupby("inicio_mes")
            .agg(
                quantidade=("quantidade", "sum"),
                registros=("quantidade", "size"),
                compras_distintas=(
                    "id_compra",
                    "nunique",
                ),
            )
            .reindex(months)
            .rename_axis("inicio_mes")
            .reset_index()
        )

        metadata = (
            group[
                [
                    "unidade_modelagem",
                    "nivel_modelagem",
                    "segmento_modelagem",
                    "metodo_rateio",
                ]
            ]
            .drop_duplicates()
            .iloc[0]
        )

        monthly["codigo_catmat"] = int(code)
        for column in [
            "unidade_modelagem",
            "nivel_modelagem",
            "segmento_modelagem",
            "metodo_rateio",
        ]:
            monthly[column] = metadata[column]

        for column in [
            "quantidade",
            "registros",
            "compras_distintas",
        ]:
            monthly[column] = pd.to_numeric(
                monthly[column],
                errors="coerce",
            ).fillna(0)

        rows.append(monthly)

    return pd.concat(
        rows,
        ignore_index=True,
    ).sort_values(
        [
            "codigo_catmat",
            "inicio_mes",
        ]
    )


def build_modeling_series(
    catmat_monthly: pd.DataFrame,
) -> pd.DataFrame:
    return (
        catmat_monthly.groupby(
            [
                "unidade_modelagem",
                "nivel_modelagem",
                "segmento_modelagem",
                "inicio_mes",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            quantidade=("quantidade", "sum"),
            registros=("registros", "sum"),
            compras_distintas=(
                "compras_distintas",
                "sum",
            ),
            catmats_ativos=(
                "codigo_catmat",
                "nunique",
            ),
        )
        .sort_values(
            [
                "unidade_modelagem",
                "inicio_mes",
            ]
        )
    )


def normalized_share(
    values: pd.Series,
) -> pd.Series:
    numeric = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )
    total = float(numeric.sum())

    if total <= 0:
        return pd.Series(
            np.repeat(
                1 / len(numeric),
                len(numeric),
            ),
            index=numeric.index,
        )

    return numeric / total


def build_allocation_weights(
    fact: pd.DataFrame,
) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []

    for unit, group in fact.groupby(
        "unidade_modelagem",
        dropna=False,
    ):
        level = str(group["nivel_modelagem"].iloc[0]).upper()

        summary = group.groupby(
            "codigo_catmat",
            as_index=False,
            dropna=False,
        ).agg(
            quantidade_historica=(
                "quantidade",
                "sum",
            ),
            compras_distintas=(
                "id_compra",
                "nunique",
            ),
            meses_ativos=(
                "inicio_mes",
                "nunique",
            ),
            registros_historicos=(
                "quantidade",
                "size",
            ),
        )

        summary["participacao_quantidade"] = normalized_share(summary["quantidade_historica"])
        summary["participacao_compras"] = normalized_share(summary["compras_distintas"])
        summary["participacao_meses"] = normalized_share(summary["meses_ativos"])

        if level == "INDIVIDUAL":
            summary["peso_rateio"] = 1.0
        else:
            summary["peso_rateio"] = (
                0.50 * summary["participacao_quantidade"]
                + 0.30 * summary["participacao_compras"]
                + 0.20 * summary["participacao_meses"]
            )
            summary["peso_rateio"] = summary["peso_rateio"] / summary["peso_rateio"].sum()

        summary["unidade_modelagem"] = unit
        summary["nivel_modelagem"] = group["nivel_modelagem"].iloc[0]
        summary["segmento_modelagem"] = group["segmento_modelagem"].iloc[0]
        summary["metodo_rateio"] = group["metodo_rateio"].iloc[0]

        summaries.append(summary)

    weights = pd.concat(
        summaries,
        ignore_index=True,
    )

    validation = weights.groupby("unidade_modelagem")["peso_rateio"].sum()

    invalid = validation.loc[
        ~np.isclose(
            validation,
            1.0,
            atol=1e-8,
        )
    ]

    if not invalid.empty:
        raise ValueError("Pesos não somam 1 para: " + ", ".join(invalid.index))

    return weights.sort_values(
        [
            "unidade_modelagem",
            "peso_rateio",
        ],
        ascending=[
            True,
            False,
        ],
    )


def build_diagnostic(
    fact: pd.DataFrame,
    modeling_series: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for unit, series in modeling_series.groupby(
        "unidade_modelagem",
        dropna=False,
    ):
        unit_fact = fact.loc[fact["unidade_modelagem"] == unit]
        positive_months = int((series["quantidade"] > 0).sum())

        rows.append(
            {
                "unidade_modelagem": unit,
                "nivel_modelagem": (series["nivel_modelagem"].iloc[0]),
                "segmento_modelagem": (series["segmento_modelagem"].iloc[0]),
                "catmats": int(unit_fact["codigo_catmat"].nunique()),
                "registros": int(len(unit_fact)),
                "quantidade_total": float(unit_fact["quantidade"].sum()),
                "anos_ativos": int(unit_fact["data_referencia"].dt.year.nunique()),
                "meses_ativos": positive_months,
                "percentual_meses_sem_demanda": float((series["quantidade"] == 0).mean()),
                "peso_minimo": float(
                    weights.loc[
                        weights["unidade_modelagem"] == unit,
                        "peso_rateio",
                    ].min()
                ),
                "peso_maximo": float(
                    weights.loc[
                        weights["unidade_modelagem"] == unit,
                        "peso_rateio",
                    ].max()
                ),
                "estrategia_previsao": (
                    "Modelo individual por CATMAT"
                    if (series["nivel_modelagem"].iloc[0] == "INDIVIDUAL")
                    else ("Modelo agregado por segmento com distribuição robusta aos CATMATs")
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "nivel_modelagem",
            "registros",
        ],
        ascending=[
            True,
            False,
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara as três unidades de modelagem dos medidores "
            "e calcula pesos robustos de distribuição por CATMAT."
        )
    )
    parser.add_argument(
        "--fact",
        type=Path,
        default=Path("data/pilot_medidores/silver/compras_medidores_piloto.parquet"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/modelagem_medidores.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pilot_medidores/modelagem"),
    )
    parser.add_argument(
        "--start-date",
        default="2021-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
    )
    args = parser.parse_args()

    if not args.fact.exists():
        raise SystemExit(f"Fato piloto não encontrada: {args.fact.resolve()}")

    if not args.mapping.exists():
        raise SystemExit(f"Mapeamento não encontrado: {args.mapping.resolve()}")

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)

    mapping = read_mapping(args.mapping)
    fact = prepare_fact(
        pd.read_parquet(args.fact),
        mapping,
        start_date,
        end_date,
    )

    if fact.empty:
        raise SystemExit("Nenhum registro permaneceu após o mapeamento.")

    catmat_monthly = complete_monthly_series(
        fact,
        start_date,
        end_date,
    )
    modeling_series = build_modeling_series(catmat_monthly)
    weights = build_allocation_weights(fact)
    diagnostic = build_diagnostic(
        fact,
        modeling_series,
        weights,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "serie_mensal_catmat": catmat_monthly,
        "serie_unidade_modelagem": modeling_series,
        "pesos_rateio_catmat": weights,
        "diagnostico_unidades_modelagem": diagnostic,
    }

    for name, dataframe in outputs.items():
        dataframe.to_parquet(
            args.output_dir / f"{name}.parquet",
            index=False,
        )
        save_psv(
            dataframe,
            args.output_dir / f"{name}.psv",
        )

    print("Preparação hierárquica concluída.")
    print(
        "CATMATs ativos:",
        fact["codigo_catmat"].nunique(),
    )
    print(
        "Unidades de modelagem:",
        modeling_series["unidade_modelagem"].nunique(),
    )
    print()
    print("Estratégias:")
    print(
        diagnostic[
            [
                "unidade_modelagem",
                "nivel_modelagem",
                "catmats",
                "registros",
                "meses_ativos",
                "estrategia_previsao",
            ]
        ].to_string(index=False)
    )
    print()
    print("Pesos de rateio:")
    print(
        weights[
            [
                "unidade_modelagem",
                "codigo_catmat",
                "participacao_quantidade",
                "participacao_compras",
                "participacao_meses",
                "peso_rateio",
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
