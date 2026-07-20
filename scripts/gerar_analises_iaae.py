from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.reporting import save_psv  # noqa: E402

REQUIRED_COLUMNS = {
    "purchase_id",
    "data_publicacao",
    "codigo_catmat",
    "descricao_item",
    "familia_material",
    "quantidade",
    "valor_unitario",
    "valor_total",
    "sigla_unidade_fornecimento",
    "capacidade_unidade_fornecimento",
    "unidade_comparavel",
    "cnpj_fornecedor",
    "nome_fornecedor",
    "uasg",
    "uf_uasg",
    "dq_possui_erro",
    "dq_duplicado",
    "is_price_outlier",
    "outlier_gravidade",
}


def save_report(df: pd.DataFrame, path: Path) -> Path:
    """Salva relatório humano em PSV, sem quebras de linha internas."""

    return save_psv(df, path)


def prepare_fact(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "A fato não segue o esquema final. Colunas ausentes: " + ", ".join(sorted(missing))
        )

    df["data_publicacao"] = pd.to_datetime(df["data_publicacao"], errors="coerce", utc=True)
    df["ano"] = df["data_publicacao"].dt.year.astype("Int64")
    df["ano_mes"] = df["data_publicacao"].dt.to_period("M").astype("string")

    for column in ("quantidade", "valor_unitario", "valor_total"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def supplier_key(df: pd.DataFrame) -> pd.Series:
    identifier = df["cnpj_fornecedor"].astype("string").str.strip()
    name = df["nome_fornecedor"].astype("string").str.strip()
    key = identifier.mask(identifier.isna() | identifier.eq(""), name)
    return key.mask(key.isna() | key.eq(""), "NAO_IDENTIFICADO")


def executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "registros": len(df),
                "data_inicial": df["data_publicacao"].min(),
                "data_final": df["data_publicacao"].max(),
                "familias": df["familia_material"].nunique(dropna=True),
                "catmats": df["codigo_catmat"].nunique(dropna=True),
                "unidades_comparaveis": df["unidade_comparavel"].nunique(dropna=True),
                "fornecedores": supplier_key(df).nunique(dropna=True),
                "uasgs": df["uasg"].nunique(dropna=True),
                "estados": df["uf_uasg"].nunique(dropna=True),
                "quantidade_total": df["quantidade"].sum(),
                "valor_total": df["valor_total"].sum(),
                "erros_qualidade": int(df["dq_possui_erro"].sum()),
                "duplicados_confirmados": int(df["dq_duplicado"].sum()),
                "outliers_preco": int(df["is_price_outlier"].sum()),
            }
        ]
    )


def monthly_family(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["ano_mes", "familia_material"], dropna=False)
        .agg(
            registros=("purchase_id", "size"),
            catmats=("codigo_catmat", "nunique"),
            quantidade_total=("quantidade", "sum"),
            valor_total=("valor_total", "sum"),
            fornecedores=("cnpj_fornecedor", "nunique"),
        )
        .reset_index()
        .sort_values(["ano_mes", "familia_material"])
    )


def monthly_catmat(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "ano_mes",
                "familia_material",
                "codigo_catmat",
                "descricao_item",
                "unidade_comparavel",
            ],
            dropna=False,
        )
        .agg(
            registros=("purchase_id", "size"),
            quantidade_total=("quantidade", "sum"),
            valor_total=("valor_total", "sum"),
            preco_mediano=("valor_unitario", "median"),
            fornecedores=("cnpj_fornecedor", "nunique"),
        )
        .reset_index()
        .sort_values(["ano_mes", "familia_material", "codigo_catmat"])
    )


def price_statistics(df: pd.DataFrame) -> pd.DataFrame:
    benchmark = df.loc[
        ~df["dq_possui_erro"]
        & ~df["is_price_outlier"]
        & df["unidade_comparavel"].notna()
        & df["valor_unitario"].gt(0)
    ].copy()

    if benchmark.empty:
        return pd.DataFrame()

    stats = (
        benchmark.groupby(
            [
                "familia_material",
                "codigo_catmat",
                "descricao_item",
                "unidade_comparavel",
            ],
            dropna=False,
        )
        .agg(
            registros=("purchase_id", "size"),
            preco_minimo=("valor_unitario", "min"),
            preco_p25=("valor_unitario", lambda x: x.quantile(0.25)),
            preco_mediano=("valor_unitario", "median"),
            preco_medio=("valor_unitario", "mean"),
            preco_p75=("valor_unitario", lambda x: x.quantile(0.75)),
            preco_maximo=("valor_unitario", "max"),
            desvio_padrao=("valor_unitario", "std"),
            quantidade_total=("quantidade", "sum"),
            valor_total=("valor_total", "sum"),
        )
        .reset_index()
    )
    stats["preco_medio_ponderado"] = stats["valor_total"] / stats["quantidade_total"].replace(
        0, np.nan
    )
    stats["coeficiente_variacao"] = stats["desvio_padrao"] / stats["preco_medio"].replace(0, np.nan)
    return stats


def supplier_concentration(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[~df["dq_possui_erro"]].copy()
    valid["fornecedor_chave"] = supplier_key(valid)
    grouped = (
        valid.groupby(
            ["ano", "familia_material", "fornecedor_chave"],
            dropna=False,
        )["valor_total"]
        .sum()
        .reset_index()
    )

    rows = []
    for (year, family), group in grouped.groupby(["ano", "familia_material"], dropna=False):
        total = group["valor_total"].sum()
        shares = group["valor_total"] / total if total > 0 else pd.Series(dtype=float)
        hhi = float((shares.pow(2).sum()) * 10000) if len(shares) else np.nan
        rows.append(
            {
                "ano": year,
                "familia_material": family,
                "fornecedores": group["fornecedor_chave"].nunique(),
                "valor_total": total,
                "hhi": round(hhi, 2) if pd.notna(hhi) else np.nan,
                "participacao_top1": shares.nlargest(1).sum(),
                "participacao_top3": shares.nlargest(3).sum(),
                "classificacao_hhi": (
                    "ALTA" if hhi >= 2500 else "MODERADA" if hhi >= 1500 else "BAIXA"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gold/eda_final"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input.resolve()}")

    fact = prepare_fact(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_report(executive_summary(fact), args.output_dir / "resumo_executivo.psv")
    save_report(monthly_family(fact), args.output_dir / "serie_mensal_familia.psv")
    save_report(monthly_catmat(fact), args.output_dir / "serie_mensal_catmat.psv")
    save_report(
        price_statistics(fact),
        args.output_dir / "estatisticas_preco_catmat.psv",
    )
    save_report(
        supplier_concentration(fact),
        args.output_dir / "concentracao_fornecedores_hhi.psv",
    )
    save_report(
        fact.loc[fact["is_price_outlier"]],
        args.output_dir / "outliers_preco.psv",
    )
    save_report(
        fact.loc[fact["dq_possui_erro"]],
        args.output_dir / "registros_qualidade.psv",
    )

    print("Análises concluídas.")
    print("Registros:", len(fact))
    print("Famílias:", fact["familia_material"].nunique())
    print("CATMATs:", fact["codigo_catmat"].nunique())
    print("Duplicados confirmados:", int(fact["dq_duplicado"].sum()))
    print("Outliers de preço:", int(fact["is_price_outlier"].sum()))
    print("Diretório:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
