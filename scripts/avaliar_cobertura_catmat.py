from __future__ import annotations

import argparse
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PRICE_URL = "https://dadosabertos.compras.gov.br/modulo-pesquisa-preco/1_consultarMaterial"

TRUE_VALUES = {"true", "1", "sim", "yes", "y"}
MIN_PAGE_SIZE = 10


def normalize_header(value: Any) -> str:
    """Normaliza cabeçalhos vindos de Python, Excel ou PowerShell."""
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Testa combinações de codificação e delimitador e escolhe
    a leitura que realmente separou o maior número de colunas.
    """
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]
    separators: list[str | None] = [
        ",",
        ";",
        "|",
        "\t",
        None,
    ]

    successful_reads: list[tuple[int, int, pd.DataFrame, str, str | None]] = []
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

                normalized_columns = [normalize_header(column) for column in dataframe.columns]

                expected_hits = len(
                    {
                        "codigo",
                        "codigoitem",
                        "codigo_item",
                        "descricao",
                        "descricaoitem",
                        "descricao_item",
                        "familia_codigo",
                    }.intersection(normalized_columns)
                )

                successful_reads.append(
                    (
                        expected_hits,
                        len(dataframe.columns),
                        dataframe,
                        encoding,
                        separator,
                    )
                )
            except Exception as exc:
                errors.append(f"encoding={encoding}, sep={separator!r}: {exc}")

    if not successful_reads:
        detail = "\n".join(errors[-5:])
        raise RuntimeError(f"Não foi possível ler {path.resolve()}.\n{detail}")

    successful_reads.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    expected_hits, column_count, dataframe, encoding, separator = successful_reads[0]

    print(
        "Entrada interpretada com "
        f"encoding={encoding}, "
        f"delimitador={separator!r}, "
        f"colunas={column_count}.",
        flush=True,
    )

    if column_count <= 1:
        raise ValueError(
            "O arquivo foi interpretado como uma única coluna. "
            f"Cabeçalho encontrado: {list(dataframe.columns)}"
        )

    return dataframe


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes, códigos e seleção opcional."""
    dataframe = df.copy()
    dataframe.columns = [normalize_header(column) for column in dataframe.columns]

    aliases = {
        "codigoitem": "codigo",
        "codigo_item": "codigo",
        "codigo_item_material": "codigo",
        "codigo_catmat": "codigo",
        "catmat": "codigo",
        "descricaoitem": "descricao",
        "descricao_item": "descricao",
        "descricao_material": "descricao",
        "familia": "familia_codigo",
        "codigo_familia": "familia_codigo",
    }

    dataframe = dataframe.rename(columns=aliases)

    required = {
        "codigo",
        "descricao",
        "familia_codigo",
    }
    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(sorted(missing))
            + ". Colunas encontradas: "
            + ", ".join(dataframe.columns)
        )

    dataframe["codigo"] = pd.to_numeric(
        dataframe["codigo"],
        errors="coerce",
    )
    dataframe = dataframe.loc[dataframe["codigo"].notna()].copy()
    dataframe["codigo"] = dataframe["codigo"].astype("int64")

    if "selecionado" in dataframe.columns:
        selected = dataframe["selecionado"].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        dataframe = dataframe.loc[selected].copy()

    return dataframe.drop_duplicates(subset=["codigo"]).reset_index(drop=True)


def build_session() -> requests.Session:
    """Cria sessão HTTP com tentativas automáticas para erros transitórios."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": ("IAAE/1.0 (projeto de dados públicos)"),
        }
    )

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def get_coverage(
    session: requests.Session,
    code: int,
    timeout: float,
) -> dict[str, Any]:
    """Consulta uma página mínima válida e retorna a cobertura do CATMAT."""
    params = {
        "pagina": 1,
        "tamanhoPagina": MIN_PAGE_SIZE,
        "codigoItemCatalogo": code,
        "dataResultado": "true",
    }

    response = session.get(
        PRICE_URL,
        params=params,
        timeout=timeout,
    )

    if not response.ok:
        body = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(f"HTTP {response.status_code} — {body or 'sem corpo de resposta'}")

    payload = response.json()

    total_records = payload.get("totalRegistros", 0) or 0
    total_pages = payload.get("totalPaginas", 0) or 0
    results = payload.get("resultado") or []
    sample = results[0] if results else {}

    return {
        "total_registros_api": int(total_records),
        "total_paginas_api": int(total_pages),
        "amostra_data_resultado": (sample.get("dataResultado")),
        "amostra_preco_unitario": (sample.get("precoUnitario")),
        "amostra_unidade": (sample.get("siglaUnidadeFornecimento")),
        "amostra_fornecedor": (sample.get("nomeFornecedor")),
        "amostra_estado": (sample.get("estado")),
        "amostra_descricao": (sample.get("descricaoItem")),
    }


def score_coverage(total: int) -> float:
    """Pontuação logarítmica para ordenar cobertura."""
    if total <= 0:
        return 0.0

    return round(
        math.log10(total + 1) * 10,
        2,
    )


def save_checkpoint(
    previous: pd.DataFrame,
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    """Salva resultados anteriores e parciais sem perder a retomada."""
    current = pd.DataFrame(rows)

    if previous.empty and current.empty:
        return

    combined = pd.concat(
        [
            previous,
            current,
        ],
        ignore_index=True,
    )

    if "codigo" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["codigo"],
            keep="last",
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    combined.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )


def load_processed_codes(
    output: Path,
) -> set[int]:
    """Lê códigos já processados em uma execução anterior."""
    if not output.exists():
        return set()

    try:
        existing = read_csv_flexible(output)
    except Exception:
        return set()

    existing.columns = [normalize_header(column) for column in existing.columns]

    if "codigo" not in existing.columns:
        return set()

    codes = pd.to_numeric(
        existing["codigo"],
        errors="coerce",
    ).dropna()
    return {int(code) for code in codes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Avalia a disponibilidade de preços praticados para CATMATs candidatos.")
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/silver/catmat_eletricos_candidatos.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/silver/cobertura_catmat_eletricos.csv"),
    )
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=Path("data/silver/shortlist_catmat_eletricos.csv"),
    )
    parser.add_argument(
        "--top-per-family",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=("Ignora CATMATs já presentes no arquivo de saída."),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input.resolve()}")

    candidates = normalize_columns(read_csv_flexible(args.input))

    if candidates.empty:
        raise SystemExit("Nenhum CATMAT selecionado permaneceu após a leitura.")

    if args.max_candidates:
        candidates = candidates.head(args.max_candidates)

    previous = (
        read_csv_flexible(args.output) if (args.resume and args.output.exists()) else pd.DataFrame()
    )

    processed_codes = load_processed_codes(args.output) if args.resume else set()

    if processed_codes:
        candidates = candidates.loc[~candidates["codigo"].isin(processed_codes)].reset_index(
            drop=True
        )

    print(f"CATMATs candidatos nesta execução: {len(candidates)}")

    if candidates.empty:
        print("Nenhum CATMAT pendente para consulta.")
        return

    session = build_session()
    rows: list[dict[str, Any]] = []
    consecutive_errors = 0

    try:
        for position, record in enumerate(
            candidates.to_dict(orient="records"),
            start=1,
        ):
            code = int(record["codigo"])

            try:
                coverage = get_coverage(
                    session=session,
                    code=code,
                    timeout=args.timeout,
                )
                status = "OK"
                error = None
                consecutive_errors = 0
            except Exception as exc:
                coverage = {
                    "total_registros_api": 0,
                    "total_paginas_api": 0,
                    "amostra_data_resultado": None,
                    "amostra_preco_unitario": None,
                    "amostra_unidade": None,
                    "amostra_fornecedor": None,
                    "amostra_estado": None,
                    "amostra_descricao": None,
                }
                status = "ERRO"
                error = f"{type(exc).__name__}: {exc}"
                consecutive_errors += 1

            total = int(coverage["total_registros_api"])

            output_row = {
                **record,
                **coverage,
                "score_cobertura": (score_coverage(total)),
                "status_consulta": status,
                "erro_consulta": error,
            }
            rows.append(output_row)

            message = (
                f"[{position}/{len(candidates)}] CATMAT {code}: {total} registro(s) — {status}"
            )
            if error:
                message += f" | {error}"
            print(
                message,
                flush=True,
            )

            if position % args.checkpoint_every == 0:
                save_checkpoint(
                    previous,
                    rows,
                    args.output,
                )
                print(
                    f"Checkpoint salvo: {args.output.resolve()}",
                    flush=True,
                )

            if consecutive_errors >= args.max_consecutive_errors:
                save_checkpoint(
                    previous,
                    rows,
                    args.output,
                )
                raise SystemExit(
                    "\nExecução interrompida "
                    "após "
                    f"{consecutive_errors} "
                    "erros consecutivos. "
                    "Consulte erro_consulta "
                    "no arquivo parcial."
                )

            time.sleep(args.delay)

    except KeyboardInterrupt:
        save_checkpoint(
            previous,
            rows,
            args.output,
        )
        print(f"\nExecução interrompida. Parcial salvo em: {args.output.resolve()}")
        return

    if args.resume and not previous.empty:
        result = pd.concat(
            [
                previous,
                pd.DataFrame(rows),
            ],
            ignore_index=True,
        )
        result = result.drop_duplicates(
            subset=["codigo"],
            keep="last",
        )
    else:
        result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "familia_codigo",
            "total_registros_api",
            "codigo",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    eligible = result.loc[
        (result["status_consulta"] == "OK") & (result["total_registros_api"] > 0)
    ].copy()

    shortlist = (
        eligible.groupby(
            "familia_codigo",
            group_keys=False,
        )
        .head(args.top_per_family)
        .reset_index(drop=True)
    )

    shortlist["selecionado_final"] = False

    shortlist.to_csv(
        args.shortlist,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nResumo de cobertura por família:")

    if result.empty:
        print("Nenhum CATMAT foi avaliado.")
    else:
        summary = (
            result.groupby("familia_codigo")
            .agg(
                catmats_avaliados=(
                    "codigo",
                    "nunique",
                ),
                catmats_com_preco=(
                    "total_registros_api",
                    lambda values: int((values > 0).sum()),
                ),
                registros_disponiveis=(
                    "total_registros_api",
                    "sum",
                ),
                erros=(
                    "status_consulta",
                    lambda values: int((values == "ERRO").sum()),
                ),
            )
            .sort_index()
        )
        print(summary.to_string())

    print(f"\nArquivo completo: {args.output.resolve()}")
    print(f"Shortlist: {args.shortlist.resolve()}")
    print(f"CATMATs na shortlist: {len(shortlist)}")


if __name__ == "__main__":
    main()
