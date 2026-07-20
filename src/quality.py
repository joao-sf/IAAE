from __future__ import annotations

import math

import numpy as np
import pandas as pd

OUTLIER_MIN_GROUP_SIZE = 8


def _build_item_key(df: pd.DataFrame) -> pd.Series:
    key = pd.Series(pd.NA, index=df.index, dtype="string")

    for column in ("id_compra_item", "id_item_compra", "purchase_id"):
        if column not in df.columns:
            continue

        values = (
            df[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
        key = key.combine_first(values)

    return key


def apply_quality_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()

    if result.empty:
        return result, pd.DataFrame(columns=["regra", "quantidade", "percentual"])

    result["data_publicacao"] = pd.to_datetime(
        result["data_publicacao"],
        errors="coerce",
        utc=True,
    )

    for column in ("quantidade", "valor_unitario", "valor_total"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["dq_codigo_catmat_ausente"] = result["codigo_catmat"].isna()
    result["dq_data_invalida"] = result["data_publicacao"].isna()
    result["dq_quantidade_invalida"] = result["quantidade"].isna() | (result["quantidade"] <= 0)
    result["dq_valor_unitario_invalido"] = result["valor_unitario"].isna() | (
        result["valor_unitario"] <= 0
    )
    result["dq_valor_total_invalido"] = result["valor_total"].isna() | (result["valor_total"] <= 0)

    supply_abbreviation = result.get(
        "sigla_unidade_fornecimento",
        pd.Series(pd.NA, index=result.index, dtype="string"),
    )
    supply_name = result.get(
        "nome_unidade_fornecimento",
        pd.Series(pd.NA, index=result.index, dtype="string"),
    )
    comparable_unit = result.get(
        "unidade_comparavel",
        pd.Series(pd.NA, index=result.index, dtype="string"),
    )

    supply_abbreviation = supply_abbreviation.astype("string").str.strip()
    supply_name = supply_name.astype("string").str.strip()
    comparable_unit = comparable_unit.astype("string").str.strip()

    result["dq_unidade_fornecimento_ausente"] = (
        supply_abbreviation.isna() | supply_abbreviation.eq("")
    ) & (supply_name.isna() | supply_name.eq(""))
    result["dq_unidade_comparavel_ausente"] = comparable_unit.isna() | comparable_unit.eq("")

    expected = result["quantidade"] * result["valor_unitario"]
    denominator = result["valor_total"].replace(0, np.nan).abs()
    result["diferenca_valor_pct"] = (result["valor_total"] - expected).abs() / denominator
    result["dq_inconsistencia_valor"] = result["diferenca_valor_pct"] > 0.02

    result["chave_item_qualidade"] = _build_item_key(result)
    valid_item_key = result["chave_item_qualidade"].notna()

    result["hist_item_repetido"] = False
    result.loc[valid_item_key, "hist_item_repetido"] = result.loc[
        valid_item_key,
        "chave_item_qualidade",
    ].duplicated(keep=False)

    observation_columns = [
        column
        for column in [
            "id_compra_item",
            "id_item_compra",
            "id_compra",
            "numero_item",
            "codigo_catmat",
            "data_publicacao",
            "cnpj_fornecedor",
            "uasg",
            "quantidade",
            "valor_unitario",
            "unidade_comparavel",
        ]
        if column in result.columns
    ]

    result["dq_duplicado"] = False
    if observation_columns:
        result["dq_duplicado"] = result.duplicated(
            subset=observation_columns,
            keep=False,
        )

    quality_columns = [column for column in result.columns if column.startswith("dq_")]
    result["dq_possui_erro"] = result[quality_columns].any(axis=1)

    report = pd.DataFrame(
        {
            "regra": quality_columns,
            "quantidade": [int(result[column].sum()) for column in quality_columns],
        }
    )
    report["percentual"] = report["quantidade"] / len(result)

    return result, report


def _severity(price: float, median: float) -> str:
    if not np.isfinite(price) or not np.isfinite(median) or median <= 0:
        return "REVISAR"

    ratio = price / median

    if ratio >= 5 or ratio <= 0.20:
        return "CRITICO"
    if ratio >= 3 or ratio <= 0.33:
        return "ALTO"
    return "MODERADO"


def flag_price_outliers(
    df: pd.DataFrame,
    minimum_group_size: int = OUTLIER_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    result = df.copy()
    result["ano"] = pd.to_datetime(
        result["data_publicacao"],
        errors="coerce",
        utc=True,
    ).dt.year.astype("Int64")

    result["is_price_outlier"] = False
    result["outlier_q1"] = np.nan
    result["outlier_mediana"] = np.nan
    result["outlier_q3"] = np.nan
    result["outlier_limite_inferior"] = np.nan
    result["outlier_limite_superior"] = np.nan
    result["outlier_razao_mediana"] = np.nan
    result["outlier_gravidade"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    eligible = (
        result["codigo_catmat"].notna()
        & result["unidade_comparavel"].notna()
        & result["ano"].notna()
        & result["valor_unitario"].notna()
        & (result["valor_unitario"] > 0)
        & ~result["dq_duplicado"]
    )

    group_columns = ["codigo_catmat", "unidade_comparavel", "ano"]

    for _, indexes in (
        result.loc[eligible]
        .groupby(
            group_columns,
            dropna=False,
        )
        .groups.items()
    ):
        values = result.loc[indexes, "valor_unitario"].dropna()

        if len(values) < minimum_group_size:
            continue

        q1 = float(values.quantile(0.25))
        median = float(values.median())
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        if not math.isfinite(iqr) or iqr <= 0:
            continue

        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        prices = result.loc[indexes, "valor_unitario"]
        flags = ~prices.between(lower, upper)

        result.loc[indexes, "outlier_q1"] = q1
        result.loc[indexes, "outlier_mediana"] = median
        result.loc[indexes, "outlier_q3"] = q3
        result.loc[indexes, "outlier_limite_inferior"] = lower
        result.loc[indexes, "outlier_limite_superior"] = upper
        result.loc[indexes, "outlier_razao_mediana"] = prices / median
        result.loc[indexes, "is_price_outlier"] = flags

        flagged_indexes = prices.index[flags]
        result.loc[flagged_indexes, "outlier_gravidade"] = [
            _severity(float(result.at[index, "valor_unitario"]), median)
            for index in flagged_indexes
        ]

    return result
