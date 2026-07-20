from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "data": ["data_publicacao", "data_resultado", "data_compra", "data"],
    "codigo_catmat": ["codigo_catmat", "codigo_item_catalogo", "codigo_item", "codigo"],
    "descricao": ["descricao_material", "descricao_item", "descricao"],
    "familia": ["familia_material", "familia", "familia_codigo"],
    "quantidade": ["quantidade", "quantidade_item", "qtd"],
    "preco_unitario": ["preco_unitario_praticado", "valor_unitario", "preco_unitario", "preco"],
    "valor_total": ["valor_total", "valor_total_item", "valor_item"],
    "unidade": [
        "sigla_unidade_fornecimento",
        "nome_unidade_fornecimento",
        "sigla_unidade_medida",
        "nome_unidade_medida",
        "unidade_fornecimento",
        "unidade_medida",
        "unidade",
    ],
    "fornecedor_nome": ["fornecedor_nome", "nome_fornecedor", "fornecedor"],
    "fornecedor_id": ["fornecedor_id", "cnpj_fornecedor", "ni_fornecedor", "cpf_cnpj_fornecedor"],
    "uasg": ["uasg", "codigo_uasg"],
    "estado": ["estado", "uf", "sigla_uf"],
    "purchase_id": ["purchase_id"],
    "id_item_compra": ["id_item_compra"],
    "id_compra_item": ["id_compra_item"],
    "id_compra": ["id_compra"],
    "numero_item": ["numero_item", "numero_item_compra"],
}


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def normalize_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.replace(r"[R$\s]", "", regex=True)
    both = text.str.contains(",", regex=False, na=False) & text.str.contains(
        ".", regex=False, na=False
    )
    comma_only = text.str.contains(",", regex=False, na=False) & ~text.str.contains(
        ".", regex=False, na=False
    )
    text.loc[both] = (
        text.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    text.loc[comma_only] = text.loc[comma_only].str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def resolve_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    normalized = {normalize_name(col): col for col in df.columns}
    mapping: dict[str, str | None] = {}
    for canonical, candidates in ALIASES.items():
        mapping[canonical] = next(
            (normalized[normalize_name(c)] for c in candidates if normalize_name(c) in normalized),
            None,
        )
    return mapping


def build_canonical(source: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    result = pd.DataFrame(index=source.index)
    for canonical, original in mapping.items():
        if original is not None:
            result[canonical] = source[original]

    required = ["data", "codigo_catmat", "familia", "quantidade", "preco_unitario"]
    missing = [col for col in required if col not in result.columns]
    if missing:
        raise ValueError("Colunas obrigatórias não identificadas: " + ", ".join(missing))

    result["data"] = pd.to_datetime(result["data"], errors="coerce", utc=True)
    result["codigo_catmat"] = pd.to_numeric(result["codigo_catmat"], errors="coerce").astype(
        "Int64"
    )
    for col in ["quantidade", "preco_unitario", "valor_total"]:
        result[col] = normalize_numeric(result[col]) if col in result.columns else np.nan
    result["valor_total"] = result["valor_total"].fillna(
        result["quantidade"] * result["preco_unitario"]
    )

    text_cols = [
        "descricao",
        "familia",
        "unidade",
        "fornecedor_nome",
        "fornecedor_id",
        "uasg",
        "estado",
        "purchase_id",
        "id_item_compra",
        "id_compra_item",
        "id_compra",
        "numero_item",
    ]
    for col in text_cols:
        if col not in result.columns:
            result[col] = pd.NA
        result[col] = result[col].astype("string").str.strip()

    result["ano"] = result["data"].dt.year.astype("Int64")
    result["unidade_informada"] = result["unidade"].notna() & result["unidade"].ne("")
    return result


def choose_unique_key(df: pd.DataFrame) -> str | None:
    best = None
    best_ratio = -1.0
    for col in ["purchase_id", "id_item_compra", "id_compra_item"]:
        s = df[col].dropna().astype("string").str.strip()
        s = s[s.ne("")]
        if s.empty:
            continue
        ratio = s.nunique() / len(s)
        if ratio > best_ratio:
            best = col
            best_ratio = ratio
    return best


def add_duplicate_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    result = df.copy()
    unique_key = choose_unique_key(result)
    result["duplicado_confirmado"] = False
    if unique_key:
        key = result[unique_key].astype("string").str.strip()
        valid = key.notna() & key.ne("")
        result.loc[valid, "duplicado_confirmado"] = key.loc[valid].duplicated(keep=False)

    result["duplicado_integral"] = result.duplicated(keep=False)
    result["repeticao_comercial"] = result.duplicated(
        subset=[
            "data",
            "codigo_catmat",
            "quantidade",
            "preco_unitario",
            "fornecedor_id",
            "purchase_id",
        ],
        keep=False,
    )
    result["mesma_compra_multiplos_itens"] = (
        result["id_compra"].notna()
        & result["id_compra"].ne("")
        & result["id_compra"].duplicated(keep=False)
    )
    return result, unique_key


def calculate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df.loc[
        df["unidade_informada"]
        & df["preco_unitario"].notna()
        & (df["preco_unitario"] > 0)
        & ~df["duplicado_confirmado"]
        & ~df["duplicado_integral"]
    ].copy()

    rows: list[pd.DataFrame] = []
    for _, group in eligible.groupby(["codigo_catmat", "unidade", "ano"], dropna=False):
        if len(group) < 8:
            continue
        q1 = float(group["preco_unitario"].quantile(0.25))
        median = float(group["preco_unitario"].median())
        q3 = float(group["preco_unitario"].quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        flagged = group.loc[
            (group["preco_unitario"] < lower) | (group["preco_unitario"] > upper)
        ].copy()
        if flagged.empty:
            continue
        flagged["q1_referencia"] = q1
        flagged["mediana_referencia"] = median
        flagged["q3_referencia"] = q3
        flagged["limite_inferior_referencia"] = lower
        flagged["limite_superior_referencia"] = upper
        flagged["razao_preco_mediana"] = (
            flagged["preco_unitario"] / median if median > 0 else np.nan
        )
        rows.append(flagged)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["tipo_outlier"] = np.where(
        result["preco_unitario"] > result["limite_superior_referencia"], "ACIMA", "ABAIXO"
    )
    ratio = result["razao_preco_mediana"]
    result["gravidade_alerta"] = np.select(
        [(ratio >= 5) | (ratio <= 0.20), (ratio >= 3) | (ratio <= 0.33)],
        ["CRITICO", "ALTO"],
        default="MODERADO",
    )
    return result.sort_values(["gravidade_alerta", "familia", "codigo_catmat", "data"])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="|", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corrige unidade, duplicidades e outliers do IAAE."
    )
    parser.add_argument("--input", type=Path, default=Path("data/gold/fact_compras.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gold/eda_v2"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input.resolve()}")

    source = pd.read_parquet(args.input)
    mapping = resolve_mapping(source)
    fact = build_canonical(source, mapping)
    fact, unique_key = add_duplicate_flags(fact)
    outliers = calculate_outliers(fact)

    duplicate_report = pd.DataFrame(
        [
            {"metrica": "registros_totais", "valor": len(fact), "interpretacao": "Total de linhas"},
            {
                "metrica": "chave_unica_utilizada",
                "valor": unique_key or "NAO_IDENTIFICADA",
                "interpretacao": "Chave de item",
            },
            {
                "metrica": "duplicados_confirmados",
                "valor": int(fact["duplicado_confirmado"].sum()),
                "interpretacao": "Mesma chave única repetida",
            },
            {
                "metrica": "duplicados_integrais",
                "valor": int(fact["duplicado_integral"].sum()),
                "interpretacao": "Linhas integralmente iguais",
            },
            {
                "metrica": "repeticoes_comerciais",
                "valor": int(fact["repeticao_comercial"].sum()),
                "interpretacao": "Mesma assinatura comercial",
            },
            {
                "metrica": "linhas_em_compras_com_multiplos_itens",
                "valor": int(fact["mesma_compra_multiplos_itens"].sum()),
                "interpretacao": "Repetição esperada de id_compra",
            },
        ]
    )

    mapping_report = pd.DataFrame(
        [{"campo_canonico": key, "coluna_origem": value} for key, value in mapping.items()]
    )
    quality_report = pd.DataFrame(
        [
            {"metrica": "registros", "valor": len(fact)},
            {"metrica": "unidade_informada", "valor": int(fact["unidade_informada"].sum())},
            {"metrica": "unidade_nao_informada", "valor": int((~fact["unidade_informada"]).sum())},
            {
                "metrica": "percentual_unidade_informada",
                "valor": round(float(fact["unidade_informada"].mean() * 100), 2),
            },
            {"metrica": "outliers_recalculados", "valor": len(outliers)},
            {
                "metrica": "outliers_criticos",
                "valor": int((outliers["gravidade_alerta"] == "CRITICO").sum())
                if not outliers.empty
                else 0,
            },
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(args.output_dir / "fact_compras_normalizada_v2.parquet", index=False)
    save_csv(mapping_report, args.output_dir / "mapeamento_colunas_v2.csv")
    save_csv(duplicate_report, args.output_dir / "diagnostico_duplicidades_v2.csv")
    save_csv(quality_report, args.output_dir / "resumo_qualidade_v2.csv")
    save_csv(outliers, args.output_dir / "outliers_preco_v2.csv")

    print("Reprocessamento concluído.")
    print("Coluna de unidade reconhecida:", mapping.get("unidade"))
    print("Chave única utilizada:", unique_key)
    print("Duplicados confirmados:", int(fact["duplicado_confirmado"].sum()))
    print("Duplicados integrais:", int(fact["duplicado_integral"].sum()))
    print("Repetições comerciais:", int(fact["repeticao_comercial"].sum()))
    print("Unidade não informada:", int((~fact["unidade_informada"]).sum()))
    print("Outliers recalculados:", len(outliers))
    if not outliers.empty:
        print("\nOutliers por gravidade:")
        print(outliers["gravidade_alerta"].value_counts().to_string())
    print(f"\nDiretório: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
