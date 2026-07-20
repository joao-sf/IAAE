from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api_client import ComprasAPIClient, extract_records  # noqa: E402
from src.settings import Settings  # noqa: E402

PDM_URL = "https://dadosabertos.compras.gov.br/modulo-material/3_consultarPdmMaterial"
ITEM_URL = "https://dadosabertos.compras.gov.br/modulo-material/4_consultarItemMaterial"

FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "disjuntores_chaves": (
        "DISJUNTOR",
        "CHAVE SECCIONADORA",
        "CHAVE FUSIVEL",
    ),
    "transformadores": ("TRANSFORMADOR",),
    "medidores_energia": (
        "MEDIDOR DE ENERGIA",
        "MEDIDOR ENERGIA",
        "MEDIDOR ELETRICO",
    ),
    "cabos_condutores": (
        "CABO ELETRICO",
        "CONDUTOR ELETRICO",
        "CABO DE POTENCIA",
    ),
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.upper().split())


def fetch_all_pages(
    client: ComprasAPIClient,
    url: str,
    params: dict[str, Any],
    page_size: int = 500,
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    """Consulta páginas com proteção contra repetição e execução infinita."""

    records: list[dict[str, Any]] = []
    seen_pages: set[str] = set()

    for page_number in range(1, max_pages + 1):
        request_params = {
            **params,
            "pagina": page_number,
            "tamanhoPagina": page_size,
        }

        payload = client.get_json(
            url,
            request_params,
        )

        page_records = extract_records(payload)

        total_records = None
        total_pages = None

        if isinstance(payload, dict):
            total_records = payload.get("totalRegistros")
            total_pages = payload.get("totalPaginas")

        print(
            f"Página {page_number}: "
            f"{len(page_records)} registro(s) | "
            f"totalRegistros={total_records} | "
            f"totalPaginas={total_pages}",
            flush=True,
        )

        if not page_records:
            print(
                "Consulta encerrada: página sem registros.",
                flush=True,
            )
            break

        fingerprint = "|".join(
            str(record.get("codigoPdm") or record.get("codigoItem") or record)
            for record in page_records
        )

        if fingerprint in seen_pages:
            print(
                "Consulta encerrada: a API repetiu uma página.",
                flush=True,
            )
            break

        seen_pages.add(fingerprint)
        records.extend(page_records)

        if total_pages is not None:
            try:
                if page_number >= int(total_pages):
                    print(
                        "Consulta encerrada: última página informada pela API.",
                        flush=True,
                    )
                    break
            except TypeError, ValueError:
                pass

        if len(page_records) < page_size:
            print(
                "Consulta encerrada: página parcial.",
                flush=True,
            )
            break

        time.sleep(client.delay_seconds)

    else:
        print(
            f"Aviso: limite de {max_pages} páginas atingido.",
            flush=True,
        )

    return records


def classify_pdm(name: str) -> tuple[str | None, str | None, int]:
    normalized_name = normalize_text(name)

    best_family: str | None = None
    best_keyword: str | None = None
    best_score = 0

    for family, keywords in FAMILY_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)

            if normalized_name == normalized_keyword:
                score = 100
            elif normalized_name.startswith(normalized_keyword):
                score = 80
            elif normalized_keyword in normalized_name:
                score = 60
            else:
                continue

            if score > best_score:
                best_family = family
                best_keyword = keyword
                best_score = score

    return best_family, best_keyword, best_score


def scan_pdms(settings: Settings, client: ComprasAPIClient) -> Path:
    print("Consultando PDMs ativos do CATMAT...")

    records = fetch_all_pages(
        client,
        PDM_URL,
        params={"statusPdm": "true"},
        page_size=500,
        max_pages=200,
    )

    rows: list[dict[str, Any]] = []

    for record in records:
        family, keyword, score = classify_pdm(record.get("nomePdm", ""))
        if family is None:
            continue

        rows.append(
            {
                "familia_codigo": family,
                "palavra_encontrada": keyword,
                "score_selecao": score,
                "codigo_grupo": record.get("codigoGrupo"),
                "nome_grupo": record.get("nomeGrupo"),
                "codigo_classe": record.get("codigoClasse"),
                "nome_classe": record.get("nomeClasse"),
                "codigo_pdm": record.get("codigoPdm"),
                "nome_pdm": record.get("nomePdm"),
                "status_pdm": record.get("statusPdm"),
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise SystemExit(
            "Nenhum PDM elétrico candidato foi localizado. Revise as palavras-chave do script."
        )

    dataframe = dataframe.sort_values(
        ["familia_codigo", "score_selecao", "nome_pdm"],
        ascending=[True, False, True],
    ).drop_duplicates(subset=["codigo_pdm"])

    output = settings.silver_dir / "pdm_eletricos_candidatos.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False, encoding="utf-8-sig")

    print("\nResumo por família:")
    print(dataframe.groupby("familia_codigo").size().to_string())
    print(f"\nArquivo gerado: {output.resolve()}")
    return output


def load_selected_pdms(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)

    required = {"familia_codigo", "codigo_pdm"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise SystemExit(
            "O arquivo de PDMs não contém as colunas obrigatórias: " + ", ".join(sorted(missing))
        )

    if "selecionado" in dataframe.columns:
        selected = (
            dataframe["selecionado"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "sim", "yes"})
        )
        dataframe = dataframe.loc[selected].copy()

    dataframe["codigo_pdm"] = pd.to_numeric(dataframe["codigo_pdm"], errors="coerce")
    dataframe = dataframe.dropna(subset=["codigo_pdm"])
    dataframe["codigo_pdm"] = dataframe["codigo_pdm"].astype(int)

    if dataframe.empty:
        raise SystemExit("Nenhum PDM foi selecionado para consulta.")

    return dataframe


def scan_items(
    settings: Settings,
    client: ComprasAPIClient,
    pdm_file: Path,
) -> Path:
    pdms = load_selected_pdms(pdm_file)
    rows: list[dict[str, Any]] = []

    for _, pdm in pdms.iterrows():
        pdm_code = int(pdm["codigo_pdm"])
        family = str(pdm["familia_codigo"])

        print(f"\nConsultando itens do PDM {pdm_code} ({pdm.get('nome_pdm', '')})...")

        records = fetch_all_pages(
            client,
            ITEM_URL,
            params={
                "codigoPdm": pdm_code,
                "statusItem": "true",
            },
            page_size=500,
            max_pages=50,
        )

        for record in records:
            rows.append(
                {
                    "codigo": record.get("codigoItem"),
                    "descricao": record.get("descricaoItem"),
                    "familia_codigo": family,
                    "familia_material": family.replace("_", " ").title(),
                    "codigo_grupo": record.get("codigoGrupo"),
                    "nome_grupo": record.get("nomeGrupo"),
                    "codigo_classe": record.get("codigoClasse"),
                    "nome_classe": record.get("nomeClasse"),
                    "codigo_pdm": record.get("codigoPdm"),
                    "nome_pdm": record.get("nomePdm"),
                    "status_item": record.get("statusItem"),
                    "item_sustentavel": record.get("itemSustentavel"),
                    "codigo_ncm": record.get("codigo_ncm"),
                    "classificacao_score": 1,
                    "classificacao_motivo": "Selecionado por PDM revisado",
                    "material_relevante": True,
                    "selecionado": False,
                    "observacao_revisao": "",
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise SystemExit("Nenhum item CATMAT foi retornado para os PDMs selecionados.")

    dataframe["codigo"] = pd.to_numeric(dataframe["codigo"], errors="coerce")
    dataframe = dataframe.dropna(subset=["codigo"])
    dataframe["codigo"] = dataframe["codigo"].astype(int)
    dataframe = dataframe.sort_values(
        ["familia_codigo", "nome_pdm", "descricao", "codigo"]
    ).drop_duplicates(subset=["codigo"])

    output = settings.silver_dir / "catmat_eletricos_candidatos.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False, encoding="utf-8-sig")

    print("\nResumo por família:")
    print(dataframe.groupby("familia_codigo").size().to_string())
    print(f"\nArquivo gerado: {output.resolve()}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Navegação controlada PDM → CATMAT para o projeto IAAE."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "pdms",
        help="Lista PDMs ativos e filtra candidatos elétricos.",
    )

    items = subparsers.add_parser(
        "items",
        help="Lista itens CATMAT dos PDMs selecionados.",
    )
    items.add_argument(
        "--pdm-file",
        type=Path,
        required=True,
        help="CSV com familia_codigo, codigo_pdm e opcionalmente selecionado.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    settings.ensure_directories()

    client = ComprasAPIClient(
        timeout_seconds=settings.timeout_seconds,
        delay_seconds=settings.request_delay_seconds,
        page_size=settings.page_size,
        max_pages=settings.max_pages,
    )

    if args.command == "pdms":
        scan_pdms(settings, client)
        return

    if args.command == "items":
        scan_items(settings, client, args.pdm_file)
        return

    raise SystemExit(f"Comando não reconhecido: {args.command}")


if __name__ == "__main__":
    main()
