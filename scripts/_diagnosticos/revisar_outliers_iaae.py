from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_GROUP_SIZE = 8
TRUE_VALUES = {"true", "1", "sim", "yes"}


def read_fact(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    required = {
        "data",
        "familia",
        "codigo_catmat",
        "descricao",
        "unidade",
        "quantidade",
        "preco_unitario",
        "valor_total",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(sorted(missing)))

    df["data"] = pd.to_datetime(df["data"], errors="coerce", utc=True)
    df["ano"] = df["data"].dt.year.astype("Int64")
    df["codigo_catmat"] = pd.to_numeric(df["codigo_catmat"], errors="coerce").astype("Int64")

    for column in ["quantidade", "preco_unitario", "valor_total"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [
        "familia",
        "descricao",
        "unidade",
        "fornecedor_nome",
        "fornecedor_id",
        "uasg",
        "estado",
        "id_compra",
    ]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype("string").str.strip()

    df["unidade_analise"] = (
        df["unidade"].fillna("UNIDADE_NAO_INFORMADA").replace("", "UNIDADE_NAO_INFORMADA")
    )

    return df


def iqr_limits(series: pd.Series) -> tuple[float, float, float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < MIN_GROUP_SIZE:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    q1 = float(values.quantile(0.25))
    median = float(values.median())
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr <= 0:
        return q1, median, q3, math.nan, math.nan

    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    return q1, median, q3, lower, upper


def add_group_statistics(
    df: pd.DataFrame,
    group_columns: list[str],
    suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        q1, median, q3, lower, upper = iqr_limits(group["preco_unitario"])

        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                f"amostra_{suffix}": len(group),
                f"q1_{suffix}": q1,
                f"mediana_{suffix}": median,
                f"q3_{suffix}": q3,
                f"limite_inferior_{suffix}": lower,
                f"limite_superior_{suffix}": upper,
            }
        )
        rows.append(row)

    stats = pd.DataFrame(rows)
    return df.merge(stats, on=group_columns, how="left")


def choose_reference(row: pd.Series) -> pd.Series:
    annual_valid = pd.notna(row["limite_superior_anual"]) and row["amostra_anual"] >= MIN_GROUP_SIZE

    if annual_valid:
        return pd.Series(
            {
                "nivel_referencia": "CATMAT_UNIDADE_ANO",
                "amostra_referencia": row["amostra_anual"],
                "q1_referencia": row["q1_anual"],
                "mediana_referencia": row["mediana_anual"],
                "q3_referencia": row["q3_anual"],
                "limite_inferior_referencia": row["limite_inferior_anual"],
                "limite_superior_referencia": row["limite_superior_anual"],
            }
        )

    return pd.Series(
        {
            "nivel_referencia": "CATMAT_UNIDADE_HISTORICO",
            "amostra_referencia": row["amostra_historica"],
            "q1_referencia": row["q1_historica"],
            "mediana_referencia": row["mediana_historica"],
            "q3_referencia": row["q3_historica"],
            "limite_inferior_referencia": row["limite_inferior_historica"],
            "limite_superior_referencia": row["limite_superior_historica"],
        }
    )


def classify_severity(row: pd.Series) -> str:
    if not row["flag_outlier_preco"]:
        return "SEM_ALERTA"

    price = row["preco_unitario"]
    median = row["mediana_referencia"]

    if pd.isna(price) or pd.isna(median) or median <= 0:
        return "REVISAR"

    ratio = price / median

    if ratio >= 5 or ratio <= 0.20:
        return "CRITICO"
    if ratio >= 3 or ratio <= 0.33:
        return "ALTO"
    return "MODERADO"


def classify_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    if pd.isna(row["unidade"]) or not str(row["unidade"]).strip():
        reasons.append("UNIDADE_NAO_INFORMADA")

    if pd.isna(row["quantidade"]) or row["quantidade"] <= 0:
        reasons.append("QUANTIDADE_INVALIDA")

    if pd.isna(row["preco_unitario"]) or row["preco_unitario"] <= 0:
        reasons.append("PRECO_INVALIDO")

    if row["duplicado_exato"]:
        reasons.append("POSSIVEL_DUPLICIDADE")

    if row["flag_outlier_preco"]:
        if row["preco_unitario"] > row["limite_superior_referencia"]:
            reasons.append("PRECO_ACIMA_IQR")
        elif row["preco_unitario"] < row["limite_inferior_referencia"]:
            reasons.append("PRECO_ABAIXO_IQR")

    return "|".join(reasons) if reasons else "SEM_ALERTA"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Revisa outliers de preço por CATMAT, unidade e ano, sem excluir registros.")
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/gold/eda/fact_compras_normalizada.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gold/eda"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input.resolve()}")

    df = read_fact(args.input)

    valid_price = (
        df["preco_unitario"].notna() & (df["preco_unitario"] > 0) & df["codigo_catmat"].notna()
    )

    working = df.loc[valid_price].copy()

    if working.empty:
        raise SystemExit("Nenhum preço válido disponível para análise.")

    duplicate_columns = [
        column
        for column in [
            "id_compra",
            "codigo_catmat",
            "quantidade",
            "preco_unitario",
            "fornecedor_id",
            "data",
        ]
        if column in working.columns
    ]

    working["duplicado_exato"] = working.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    working = add_group_statistics(
        working,
        ["codigo_catmat", "unidade_analise"],
        "historica",
    )

    working = add_group_statistics(
        working,
        ["codigo_catmat", "unidade_analise", "ano"],
        "anual",
    )

    reference = working.apply(
        choose_reference,
        axis=1,
    )
    working = pd.concat(
        [working.reset_index(drop=True), reference.reset_index(drop=True)],
        axis=1,
    )

    valid_limits = (
        working["limite_inferior_referencia"].notna()
        & working["limite_superior_referencia"].notna()
    )

    working["flag_outlier_preco"] = valid_limits & (
        (working["preco_unitario"] < working["limite_inferior_referencia"])
        | (working["preco_unitario"] > working["limite_superior_referencia"])
    )

    working["razao_preco_mediana"] = np.where(
        working["mediana_referencia"] > 0,
        working["preco_unitario"] / working["mediana_referencia"],
        np.nan,
    )

    working["gravidade_alerta"] = working.apply(
        classify_severity,
        axis=1,
    )
    working["motivo_alerta"] = working.apply(
        classify_reason,
        axis=1,
    )

    outliers = working.loc[working["flag_outlier_preco"] | working["duplicado_exato"]].copy()

    detail_columns = [
        "data",
        "ano",
        "familia",
        "codigo_catmat",
        "descricao",
        "unidade",
        "quantidade",
        "preco_unitario",
        "valor_total",
        "fornecedor_nome",
        "fornecedor_id",
        "uasg",
        "estado",
        "id_compra",
        "nivel_referencia",
        "amostra_referencia",
        "q1_referencia",
        "mediana_referencia",
        "q3_referencia",
        "limite_inferior_referencia",
        "limite_superior_referencia",
        "razao_preco_mediana",
        "duplicado_exato",
        "flag_outlier_preco",
        "gravidade_alerta",
        "motivo_alerta",
    ]

    outliers = outliers[detail_columns].sort_values(
        [
            "gravidade_alerta",
            "familia",
            "codigo_catmat",
            "data",
        ]
    )

    summary = (
        outliers.groupby(
            [
                "familia",
                "codigo_catmat",
                "descricao",
                "unidade",
                "gravidade_alerta",
                "motivo_alerta",
            ],
            dropna=False,
        )
        .agg(
            alertas=("codigo_catmat", "size"),
            data_inicial=("data", "min"),
            data_final=("data", "max"),
            preco_minimo=("preco_unitario", "min"),
            preco_mediano=("preco_unitario", "median"),
            preco_maximo=("preco_unitario", "max"),
            fornecedores=("fornecedor_id", "nunique"),
            compras=("id_compra", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["alertas", "familia"],
            ascending=[False, True],
        )
    )

    counts = (
        outliers.groupby(
            ["familia", "gravidade_alerta"],
            dropna=False,
        )
        .size()
        .rename("alertas")
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(
        outliers,
        args.output_dir / "outliers_preco_revisados.csv",
    )
    save_csv(
        summary,
        args.output_dir / "resumo_outliers_revisados.csv",
    )
    save_csv(
        counts,
        args.output_dir / "contagem_outliers_gravidade.csv",
    )

    working.to_parquet(
        args.output_dir / "base_preco_com_flags.parquet",
        index=False,
    )

    print("Revisão de outliers concluída.")
    print(f"Registros com preço válido: {len(working)}")
    print(f"Alertas revisados: {len(outliers)}")
    print(f"Taxa de alertas: {(len(outliers) / len(working) * 100):.2f}%")

    print("\nPor gravidade:")
    if outliers.empty:
        print("Nenhum alerta identificado.")
    else:
        print(outliers["gravidade_alerta"].value_counts().to_string())

    print("\nPor família:")
    if outliers.empty:
        print("Nenhum alerta identificado.")
    else:
        print(outliers.groupby("familia").size().sort_values(ascending=False).to_string())

    print(f"\nDiretório: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
