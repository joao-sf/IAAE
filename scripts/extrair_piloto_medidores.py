from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PRICE_URL = "https://dadosabertos.compras.gov.br/modulo-pesquisa-preco/1_consultarMaterial"

TRUE_VALUES = {"true", "1", "sim", "yes", "y"}


def normalize_header(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    separators: list[str | None] = [",", ";", "|", "\t", None]

    successful: list[tuple[int, int, pd.DataFrame, str, str | None]] = []

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
            except Exception:
                continue

            columns = [normalize_header(column) for column in dataframe.columns]

            expected_hits = len(
                {
                    "codigo",
                    "descricao",
                    "familia_codigo",
                    "total_registros_api",
                    "status_consulta",
                }.intersection(columns)
            )

            successful.append(
                (
                    expected_hits,
                    len(dataframe.columns),
                    dataframe,
                    encoding,
                    separator,
                )
            )

    if not successful:
        raise RuntimeError(f"Não foi possível ler {path.resolve()}.")

    successful.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    expected_hits, column_count, dataframe, encoding, separator = successful[0]

    if column_count <= 1:
        raise ValueError("O arquivo foi interpretado como uma única coluna.")

    print(
        "Entrada interpretada com "
        f"encoding={encoding}, "
        f"delimitador={separator!r}, "
        f"colunas={column_count}.",
        flush=True,
    )

    dataframe.columns = [normalize_header(column) for column in dataframe.columns]

    return dataframe


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "IAAE/1.0 (piloto medidores)",
        }
    )

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = re.sub(
        r"[\r\n\t]+",
        " ",
        str(value),
    )
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text or None


def save_psv(dataframe: pd.DataFrame, path: Path) -> None:
    output = dataframe.copy()

    for column in output.select_dtypes(include=["object", "string"]).columns:
        output[column] = output[column].map(clean_text)

    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def prepare_candidates(path: Path) -> pd.DataFrame:
    dataframe = read_csv_flexible(path)

    aliases = {
        "codigo_catmat": "codigo",
        "codigoitem": "codigo",
        "codigo_item": "codigo",
        "descricao_item": "descricao",
        "descricaoitem": "descricao",
    }
    dataframe = dataframe.rename(columns=aliases)

    required = {
        "codigo",
        "descricao",
        "familia_codigo",
        "total_registros_api",
        "status_consulta",
    }
    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(sorted(missing)))

    dataframe["codigo"] = pd.to_numeric(
        dataframe["codigo"],
        errors="coerce",
    )
    dataframe["total_registros_api"] = pd.to_numeric(
        dataframe["total_registros_api"],
        errors="coerce",
    ).fillna(0)

    valid = (
        dataframe["codigo"].notna()
        & (dataframe["status_consulta"].astype(str).str.strip().str.upper() == "OK")
        & (dataframe["total_registros_api"] > 0)
    )

    if "selecionado" in dataframe.columns:
        selected = dataframe["selecionado"].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        valid &= selected

    dataframe = dataframe.loc[valid].copy()
    dataframe["codigo"] = dataframe["codigo"].astype(int)

    return (
        dataframe.drop_duplicates(subset=["codigo"])
        .sort_values(
            "total_registros_api",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def fetch_catmat(
    session: requests.Session,
    code: int,
    bronze_dir: Path,
    page_size: int,
    timeout: float,
    delay: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1

    while True:
        params = {
            "pagina": page,
            "tamanhoPagina": page_size,
            "codigoItemCatalogo": code,
            "dataResultado": "true",
        }

        response = session.get(
            PRICE_URL,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        payload = response.json()
        records = payload.get("resultado") or []

        bronze_path = bronze_dir / f"catmat_{code}" / f"pagina_{page:04d}.json"
        bronze_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        bronze_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"  página {page}: {len(records)} registro(s)",
            flush=True,
        )

        rows.extend(records)

        total_pages = int(payload.get("totalPaginas") or 0)

        if not records:
            break

        if total_pages and page >= total_pages:
            break

        if len(records) < page_size:
            break

        page += 1
        time.sleep(delay)

    return rows


def first_existing(
    row: pd.Series,
    names: list[str],
) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if pd.notna(value):
                return value

    return None


def prepare_fact(
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if raw.empty:
        return raw

    fact = raw.copy()
    fact.columns = [normalize_header(column) for column in fact.columns]

    aliases = {
        "codigoitemcatalogo": "codigo_catmat",
        "codigo_item_catalogo": "codigo_catmat",
        "dataresultado": "data_resultado",
        "datacompra": "data_compra",
        "idcompra": "id_compra",
        "iditemcompra": "id_item_compra",
        "quantidadeitem": "quantidade",
        "precounitario": "preco_unitario",
        "nomefornecedor": "nome_fornecedor",
        "nifornecedor": "ni_fornecedor",
        "codigouasg": "codigo_uasg",
        "nomeuasg": "nome_uasg",
        "siglaunidadefornecimento": ("sigla_unidade_fornecimento"),
        "nomeunidadefornecimento": ("nome_unidade_fornecimento"),
        "capacidadeunidadefornecimento": ("capacidade_unidade_fornecimento"),
        "descricaoitem": "descricao_item",
    }
    fact = fact.rename(columns=aliases)

    if "quantidade" not in fact.columns:
        fact["quantidade"] = np.nan

    if "preco_unitario" not in fact.columns:
        fact["preco_unitario"] = np.nan

    fact["codigo_catmat"] = pd.to_numeric(
        fact["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    fact["quantidade"] = pd.to_numeric(
        fact["quantidade"],
        errors="coerce",
    )

    fact["preco_unitario"] = pd.to_numeric(
        fact["preco_unitario"],
        errors="coerce",
    )

    for date_column in [
        "data_resultado",
        "data_compra",
    ]:
        if date_column not in fact.columns:
            fact[date_column] = pd.NaT

        fact[date_column] = pd.to_datetime(
            fact[date_column],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

    fact["data_referencia"] = fact["data_resultado"].fillna(fact["data_compra"])

    fact = fact.loc[
        fact["data_referencia"].between(
            start_date,
            end_date,
            inclusive="both",
        )
    ].copy()

    metadata_columns = [
        column
        for column in [
            "codigo",
            "descricao",
            "familia_codigo",
            "familia_material",
            "categoria_tecnica",
            "segmento_analitico",
            "prioridade",
            "total_registros_api",
        ]
        if column in candidates.columns
    ]

    metadata = candidates[metadata_columns].rename(
        columns={
            "codigo": "codigo_catmat",
            "descricao": "descricao_catalogo",
        }
    )

    fact = fact.merge(
        metadata,
        on="codigo_catmat",
        how="left",
    )

    fact["valor_total_calculado"] = fact["quantidade"] * fact["preco_unitario"]

    fact["ano"] = fact["data_referencia"].dt.year
    fact["mes_referencia"] = fact["data_referencia"].dt.to_period("M").astype(str)

    preferred_order = [
        "codigo_catmat",
        "descricao_catalogo",
        "categoria_tecnica",
        "segmento_analitico",
        "prioridade",
        "data_referencia",
        "ano",
        "mes_referencia",
        "id_compra",
        "id_item_compra",
        "descricao_item",
        "quantidade",
        "preco_unitario",
        "valor_total_calculado",
        "ni_fornecedor",
        "nome_fornecedor",
        "codigo_uasg",
        "nome_uasg",
        "sigla_unidade_fornecimento",
        "nome_unidade_fornecimento",
        "capacidade_unidade_fornecimento",
    ]

    existing_order = [column for column in preferred_order if column in fact.columns]
    remaining = [column for column in fact.columns if column not in existing_order]

    return fact[existing_order + remaining].sort_values(
        [
            "codigo_catmat",
            "data_referencia",
        ]
    )


def concentration_share(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0)

    total = float(numeric.sum())
    if total <= 0:
        return np.nan

    return float(numeric.max() / total)


def build_diagnostic(
    fact: pd.DataFrame,
) -> pd.DataFrame:
    if fact.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for code, group in fact.groupby(
        "codigo_catmat",
        dropna=False,
    ):
        valid_quantity = group.loc[group["quantidade"].notna() & (group["quantidade"] >= 0)].copy()

        purchase_totals = (
            valid_quantity.groupby(
                "id_compra",
                dropna=False,
            )["quantidade"].sum()
            if "id_compra" in valid_quantity.columns
            else pd.Series(dtype=float)
        )

        supplier_totals = (
            valid_quantity.groupby(
                "ni_fornecedor",
                dropna=False,
            )["quantidade"].sum()
            if "ni_fornecedor" in valid_quantity.columns
            else pd.Series(dtype=float)
        )

        months_active = int(group["mes_referencia"].dropna().nunique())
        years_active = int(group["ano"].dropna().nunique())
        records = int(len(group))

        individual_candidate = records >= 24 and years_active >= 3 and months_active >= 12

        rows.append(
            {
                "codigo_catmat": int(code),
                "descricao_catalogo": (
                    group["descricao_catalogo"].dropna().iloc[0]
                    if (
                        "descricao_catalogo" in group.columns
                        and group["descricao_catalogo"].notna().any()
                    )
                    else None
                ),
                "categoria_tecnica": (
                    group["categoria_tecnica"].dropna().iloc[0]
                    if (
                        "categoria_tecnica" in group.columns
                        and group["categoria_tecnica"].notna().any()
                    )
                    else None
                ),
                "segmento_analitico": (
                    group["segmento_analitico"].dropna().iloc[0]
                    if (
                        "segmento_analitico" in group.columns
                        and group["segmento_analitico"].notna().any()
                    )
                    else None
                ),
                "registros_periodo": records,
                "data_inicial": (group["data_referencia"].min()),
                "data_final": (group["data_referencia"].max()),
                "anos_ativos": years_active,
                "meses_ativos": months_active,
                "quantidade_total": float(valid_quantity["quantidade"].sum()),
                "compras_distintas": int(group["id_compra"].dropna().nunique())
                if "id_compra" in group.columns
                else 0,
                "fornecedores": int(group["ni_fornecedor"].dropna().nunique())
                if "ni_fornecedor" in group.columns
                else 0,
                "uasgs": int(group["codigo_uasg"].dropna().nunique())
                if "codigo_uasg" in group.columns
                else 0,
                "unidades_fornecimento": int(group["sigla_unidade_fornecimento"].dropna().nunique())
                if ("sigla_unidade_fornecimento" in group.columns)
                else 0,
                "maior_compra_participacao": (concentration_share(purchase_totals)),
                "maior_fornecedor_participacao": (concentration_share(supplier_totals)),
                "candidato_modelo_individual": (individual_candidate),
                "modelagem_recomendada": (
                    "CATMAT individual"
                    if individual_candidate
                    else "Modelo hierárquico por segmento"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "candidato_modelo_individual",
            "registros_periodo",
            "quantidade_total",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )


def build_segment_diagnostic(
    fact: pd.DataFrame,
) -> pd.DataFrame:
    if fact.empty or "segmento_analitico" not in fact.columns:
        return pd.DataFrame()

    return (
        fact.groupby(
            "segmento_analitico",
            as_index=False,
            dropna=False,
        )
        .agg(
            catmats=(
                "codigo_catmat",
                "nunique",
            ),
            registros_periodo=(
                "codigo_catmat",
                "size",
            ),
            quantidade_total=(
                "quantidade",
                "sum",
            ),
            data_inicial=(
                "data_referencia",
                "min",
            ),
            data_final=(
                "data_referencia",
                "max",
            ),
            anos_ativos=(
                "ano",
                "nunique",
            ),
            meses_ativos=(
                "mes_referencia",
                "nunique",
            ),
            compras_distintas=(
                "id_compra",
                "nunique",
            ),
            fornecedores=(
                "ni_fornecedor",
                "nunique",
            ),
            uasgs=(
                "codigo_uasg",
                "nunique",
            ),
        )
        .sort_values(
            [
                "registros_periodo",
                "quantidade_total",
            ],
            ascending=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai, em área isolada, o histórico dos CATMATs de medidores com cobertura positiva."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/silver/descoberta_medidores/cobertura_medidores_validada.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pilot_medidores"),
    )
    parser.add_argument(
        "--start-date",
        default="2021-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.30,
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input.resolve()}")

    if not 10 <= args.page_size <= 500:
        raise SystemExit("--page-size deve estar entre 10 e 500.")

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)

    candidates = prepare_candidates(args.input)

    if candidates.empty:
        raise SystemExit("Nenhum CATMAT com cobertura positiva.")

    print(
        "CATMATs para o piloto:",
        len(candidates),
    )
    print(
        "Registros indicados pela cobertura:",
        int(candidates["total_registros_api"].sum()),
    )

    session = build_session()
    raw_rows: list[dict[str, Any]] = []

    bronze_dir = args.output_dir / "bronze"

    for position, record in enumerate(
        candidates.to_dict(orient="records"),
        start=1,
    ):
        code = int(record["codigo"])

        print()
        print(
            f"[{position}/{len(candidates)}] CATMAT {code}",
            flush=True,
        )

        rows = fetch_catmat(
            session=session,
            code=code,
            bronze_dir=bronze_dir,
            page_size=args.page_size,
            timeout=args.timeout,
            delay=args.delay,
        )
        raw_rows.extend(rows)

    raw = pd.DataFrame(raw_rows)
    fact = prepare_fact(
        raw=raw,
        candidates=candidates,
        start_date=start_date,
        end_date=end_date,
    )

    diagnostic = build_diagnostic(fact)
    segment_diagnostic = build_segment_diagnostic(fact)

    silver_dir = args.output_dir / "silver"
    gold_dir = args.output_dir / "gold"
    silver_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    gold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fact.to_parquet(
        silver_dir / "compras_medidores_piloto.parquet",
        index=False,
    )
    save_psv(
        fact,
        silver_dir / "compras_medidores_piloto.psv",
    )

    diagnostic.to_parquet(
        gold_dir / "diagnostico_catmat_medidores.parquet",
        index=False,
    )
    save_psv(
        diagnostic,
        gold_dir / "diagnostico_catmat_medidores.psv",
    )

    if not segment_diagnostic.empty:
        segment_diagnostic.to_parquet(
            gold_dir / "diagnostico_segmentos_medidores.parquet",
            index=False,
        )
        save_psv(
            segment_diagnostic,
            gold_dir / "diagnostico_segmentos_medidores.psv",
        )

    print()
    print("Extração piloto concluída.")
    print(
        "Registros brutos retornados:",
        len(raw),
    )
    print(
        f"Registros entre {args.start_date} e {args.end_date}:",
        len(fact),
    )
    print(
        "CATMATs com histórico no período:",
        fact["codigo_catmat"].nunique() if not fact.empty else 0,
    )
    print(
        "Quantidade física total:",
        round(
            pd.to_numeric(
                fact["quantidade"],
                errors="coerce",
            ).sum(),
            2,
        )
        if not fact.empty
        else 0,
    )
    print(
        "Diretório:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
