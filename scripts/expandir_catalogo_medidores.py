from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api_client import ComprasAPIClient, extract_records  # noqa: E402
from src.settings import Settings  # noqa: E402

save_psv = import_module("src.reporting").save_psv

PDM_URL = "https://dadosabertos.compras.gov.br/modulo-material/3_consultarPdmMaterial"
ITEM_URL = "https://dadosabertos.compras.gov.br/modulo-material/4_consultarItemMaterial"

TRUE_VALUES = {
    "true",
    "1",
    "sim",
    "yes",
    "y",
}

PDM_MEASUREMENT_TERMS = {
    "MEDIDOR": 25,
    "MEDICAO": 18,
    "INSTRUMENTO": 6,
    "CONTADOR": 10,
    "REGISTRADOR": 8,
}

PDM_ELECTRICAL_TERMS = {
    "ENERGIA": 25,
    "ELETRIC": 22,
    "WATTHORA": 35,
    "KWH": 35,
    "POTENCIA": 12,
    "CONSUMO": 8,
    "MULTIFUNCAO": 16,
    "ELETRONICO": 10,
}

PDM_EXCLUSION_TERMS = {
    "AGUA": -45,
    "VAZAO": -45,
    "PRESSAO": -45,
    "TEMPERATURA": -45,
    "UMIDADE": -45,
    "GAS": -45,
    "COMBUSTIVEL": -45,
    "RADIACAO": -35,
    "ODONTOLOG": -50,
    "HOSPITAL": -40,
    "DISTANCIA": -30,
    "VELOCIDADE": -30,
    "ESPESSURA": -30,
    "PH": -30,
}

ITEM_POSITIVE_TERMS = {
    "MEDIDOR": 25,
    "MEDICAO": 12,
    "ENERGIA": 25,
    "ELETRIC": 18,
    "ELETRONICO": 12,
    "MONOFASICO": 16,
    "BIFASICO": 16,
    "TRIFASICO": 16,
    "MULTIFUNCAO": 18,
    "WATTHORA": 25,
    "KWH": 25,
    "ATIVA": 10,
    "REATIVA": 6,
    "DIRETA": 8,
    "DIRETO": 8,
    "INDIRETA": 8,
    "INDIRETO": 8,
    "INTELIGENTE": 20,
    "SMART": 20,
    "TELEMEDICAO": 18,
    "AMI": 18,
}

ITEM_REVIEW_TERMS = {
    "REATIVA": -4,
    "PORTATIL": -18,
    "ALICATE": -25,
    "ANALISADOR": -22,
    "LABORATORIO": -22,
    "TESTE": -18,
    "CALIBRADOR": -25,
    "PADRAO": -15,
    "BANCO DE CAPACITOR": -20,
}

ITEM_HARD_EXCLUSION_TERMS = {
    "VAZAO",
    "PRESSAO",
    "TEMPERATURA",
    "UMIDADE",
    "AGUA",
    "GAS",
    "COMBUSTIVEL",
    "DISTANCIA",
    "ESPESSURA",
    "RADIOLOG",
    "ODONTOLOG",
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\s+", " ", text.upper()).strip()
    return text


def has_word(text: str, word: str) -> bool:
    pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
    return re.search(pattern, text) is not None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts = [
        {
            "encoding": "utf-8-sig",
            "sep": ",",
        },
        {
            "encoding": "utf-8-sig",
            "sep": ";",
        },
        {
            "encoding": "utf-8",
            "sep": ",",
        },
        {
            "encoding": "utf-8",
            "sep": ";",
        },
        {
            "encoding": "cp1252",
            "sep": None,
            "engine": "python",
        },
    ]

    candidates: list[pd.DataFrame] = []
    errors: list[str] = []

    for options in attempts:
        try:
            dataframe = pd.read_csv(
                path,
                low_memory=False,
                **options,
            )
            candidates.append(dataframe)
        except Exception as exc:
            errors.append(str(exc))

    if not candidates:
        detail = " | ".join(errors[-3:])
        raise RuntimeError(f"Não foi possível ler {path}: {detail}")

    return max(
        candidates,
        key=lambda dataframe: len(dataframe.columns),
    )


def fetch_all_pages(
    client: ComprasAPIClient,
    url: str,
    params: dict[str, Any],
    page_size: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    page_number = 1

    while True:
        request_params = {
            **params,
            "pagina": page_number,
        }

        if page_size is not None:
            request_params["tamanhoPagina"] = page_size

        payload = client.get_json(
            url,
            request_params,
        )
        page_records = extract_records(payload)

        print(
            f"Página {page_number}: {len(page_records)} registro(s)",
            flush=True,
        )

        if not page_records:
            break

        records.extend(page_records)

        total_pages = 0
        if isinstance(payload, dict):
            try:
                total_pages = int(payload.get("totalPaginas") or 0)
            except TypeError, ValueError:
                total_pages = 0

        if total_pages and page_number >= total_pages:
            break

        if page_size is not None and len(page_records) < page_size:
            break

        page_number += 1
        time.sleep(client.delay_seconds)

    return records


def term_score(
    text: str,
    terms: dict[str, int],
) -> tuple[int, list[str]]:
    score = 0
    hits: list[str] = []

    for term, points in terms.items():
        if term in text:
            score += points
            hits.append(term)

    return score, hits


def classify_pdm(
    record: dict[str, Any],
    minimum_score: int,
) -> dict[str, Any]:
    name = normalize_text(record.get("nomePdm"))
    class_name = normalize_text(record.get("nomeClasse"))
    group_name = normalize_text(record.get("nomeGrupo"))

    combined = " | ".join(
        [
            name,
            class_name,
            group_name,
        ]
    )

    measurement_score, measurement_hits = term_score(
        combined,
        PDM_MEASUREMENT_TERMS,
    )
    electrical_score, electrical_hits = term_score(
        combined,
        PDM_ELECTRICAL_TERMS,
    )
    exclusion_score, exclusion_hits = term_score(
        combined,
        PDM_EXCLUSION_TERMS,
    )

    total_score = measurement_score + electrical_score + exclusion_score

    has_measurement_context = bool(measurement_hits)
    has_electrical_context = bool(electrical_hits)

    eligible = has_measurement_context and has_electrical_context and total_score >= minimum_score

    return {
        "codigo_grupo": record.get("codigoGrupo"),
        "nome_grupo": record.get("nomeGrupo"),
        "codigo_classe": record.get("codigoClasse"),
        "nome_classe": record.get("nomeClasse"),
        "codigo_pdm": record.get("codigoPdm"),
        "nome_pdm": record.get("nomePdm"),
        "status_pdm": record.get("statusPdm"),
        "score_medicao": measurement_score,
        "score_eletrico": electrical_score,
        "score_exclusao": exclusion_score,
        "score_total": total_score,
        "termos_medicao": ", ".join(measurement_hits),
        "termos_eletricos": ", ".join(electrical_hits),
        "termos_exclusao": ", ".join(exclusion_hits),
        "apto_consulta": eligible,
        "selecionado": False,
    }


def item_category(
    text: str,
) -> str:
    if any(
        term in text
        for term in [
            "INTELIGENTE",
            "SMART",
            "TELEMEDICAO",
            "AMI",
        ]
    ):
        return "Medidor inteligente / telemedição"

    if "MULTIFUNCAO" in text:
        return "Medidor multifunção"

    if "TRIFASICO" in text:
        return "Medidor trifásico"

    if "BIFASICO" in text:
        return "Medidor bifásico"

    if "MONOFASICO" in text:
        return "Medidor monofásico"

    has_active = has_word(text, "ATIVA")
    has_reactive = has_word(text, "REATIVA")

    if has_active and has_reactive:
        return "Medidor de energia ativa e reativa"

    if has_reactive and not has_active:
        return "Medidor especializado em energia reativa"

    return "Medidor de energia genérico"


def classify_item(
    record: dict[str, Any],
) -> dict[str, Any]:
    description = normalize_text(record.get("descricaoItem"))

    positive_score, positive_hits = term_score(
        description,
        ITEM_POSITIVE_TERMS,
    )
    review_score, review_hits = term_score(
        description,
        ITEM_REVIEW_TERMS,
    )

    hard_exclusions = [term for term in ITEM_HARD_EXCLUSION_TERMS if term in description]

    has_meter = any(
        term in description
        for term in [
            "MEDIDOR",
            "MEDICAO",
            "CONTADOR",
        ]
    )
    has_energy = any(
        term in description
        for term in [
            "ENERGIA",
            "ELETRIC",
            "WATTHORA",
            "KWH",
            "POTENCIA ATIVA",
            "POTENCIA REATIVA",
        ]
    )

    score = positive_score + review_score

    has_active = has_word(description, "ATIVA")
    has_reactive = has_word(description, "REATIVA")

    reactive_only = has_reactive and not has_active

    instrument_only = any(
        term in description
        for term in [
            "ANALISADOR",
            "ALICATE",
            "PORTATIL",
            "CALIBRADOR",
            "LABORATORIO",
        ]
    )

    if hard_exclusions or not has_meter or not has_energy:
        status = "DESCARTAR"
    elif reactive_only or instrument_only or "BANCO DE CAPACITOR" in description:
        status = "ESPECIALIZADO"
    elif score >= 65:
        status = "ADERENTE"
    elif score >= 40:
        status = "REVISAR"
    else:
        status = "REVISAR"

    return {
        "codigo": record.get("codigoItem"),
        "descricao": record.get("descricaoItem"),
        "familia_codigo": ("medidores_energia"),
        "familia_material": ("Medidores Energia"),
        "codigo_grupo": record.get("codigoGrupo"),
        "nome_grupo": record.get("nomeGrupo"),
        "codigo_classe": record.get("codigoClasse"),
        "nome_classe": record.get("nomeClasse"),
        "codigo_pdm": record.get("codigoPdm"),
        "nome_pdm": record.get("nomePdm"),
        "status_item": record.get("statusItem"),
        "item_sustentavel": record.get("itemSustentavel"),
        "codigo_ncm": (record.get("codigo_ncm") or record.get("codigoNcm")),
        "categoria_tecnica": (item_category(description)),
        "score_relevancia_tecnica": score,
        "termos_positivos": ", ".join(positive_hits),
        "termos_revisao": ", ".join(review_hits),
        "termos_exclusao": ", ".join(hard_exclusions),
        "status_classificacao": status,
        "material_relevante": (status != "DESCARTAR"),
        "selecionado": False,
        "observacao_revisao": "",
    }


def scan_pdms(
    settings: Settings,
    client: ComprasAPIClient,
    minimum_score: int,
    output_dir: Path,
) -> Path:
    print("Consultando todos os PDMs ativos...")

    records = fetch_all_pages(
        client,
        PDM_URL,
        params={
            "statusPdm": "true",
        },
    )

    rows = [
        classify_pdm(
            record,
            minimum_score,
        )
        for record in records
    ]

    dataframe = pd.DataFrame(rows)

    candidates = dataframe.loc[
        (dataframe["score_medicao"] > 0) | (dataframe["score_eletrico"] > 0)
    ].copy()

    candidates["codigo_pdm"] = pd.to_numeric(
        candidates["codigo_pdm"],
        errors="coerce",
    )
    candidates = candidates.dropna(subset=["codigo_pdm"])
    candidates["codigo_pdm"] = candidates["codigo_pdm"].astype(int)

    candidates = (
        candidates.sort_values(
            [
                "apto_consulta",
                "score_total",
                "nome_pdm",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .drop_duplicates(subset=["codigo_pdm"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output = output_dir / "pdm_medidores_ampliado.csv"
    psv_output = output_dir / "pdm_medidores_ampliado.psv"

    candidates.to_csv(
        csv_output,
        index=False,
        encoding="utf-8-sig",
    )
    save_psv(
        candidates,
        psv_output,
    )

    print()
    print(f"PDMs com sinais de medição ou eletricidade: {len(candidates)}")
    print(
        "PDMs aptos para consulta de itens:",
        int(candidates["apto_consulta"].sum()),
    )
    print(
        "Arquivo:",
        csv_output.resolve(),
    )

    return csv_output


def load_pdms_for_items(
    path: Path,
) -> pd.DataFrame:
    dataframe = read_csv_flexible(path)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    required = {
        "codigo_pdm",
    }
    missing = required.difference(dataframe.columns)

    if missing:
        raise SystemExit("Arquivo de PDMs sem coluna codigo_pdm.")

    if "selecionado" in dataframe.columns:
        selected = dataframe["selecionado"].astype(str).str.strip().str.lower().isin(TRUE_VALUES)

        if selected.any():
            dataframe = dataframe.loc[selected].copy()
        elif "apto_consulta" in dataframe.columns:
            eligible = (
                dataframe["apto_consulta"].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
            )
            dataframe = dataframe.loc[eligible].copy()

    elif "apto_consulta" in dataframe.columns:
        eligible = dataframe["apto_consulta"].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        dataframe = dataframe.loc[eligible].copy()

    dataframe["codigo_pdm"] = pd.to_numeric(
        dataframe["codigo_pdm"],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["codigo_pdm"])
    dataframe["codigo_pdm"] = dataframe["codigo_pdm"].astype(int)

    if dataframe.empty:
        raise SystemExit("Nenhum PDM selecionado ou apto para consulta.")

    return dataframe.drop_duplicates(subset=["codigo_pdm"])


def scan_items(
    settings: Settings,
    client: ComprasAPIClient,
    pdm_file: Path,
    output_dir: Path,
) -> Path:
    pdms = load_pdms_for_items(pdm_file)

    print(f"PDMs que serão consultados: {len(pdms)}")

    rows: list[dict[str, Any]] = []

    for position, pdm in enumerate(
        pdms.to_dict(orient="records"),
        start=1,
    ):
        pdm_code = int(pdm["codigo_pdm"])

        print()
        print(f"[{position}/{len(pdms)}] PDM {pdm_code} — {pdm.get('nome_pdm', '')}")

        records = fetch_all_pages(
            client,
            ITEM_URL,
            params={
                "codigoPdm": pdm_code,
                "statusItem": "true",
            },
            page_size=500,
        )

        for record in records:
            classified = classify_item(record)

            if not classified.get("nome_pdm"):
                classified["nome_pdm"] = pdm.get("nome_pdm")

            rows.append(classified)

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise SystemExit("Nenhum item foi retornado para os PDMs consultados.")

    dataframe["codigo"] = pd.to_numeric(
        dataframe["codigo"],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["codigo"])
    dataframe["codigo"] = dataframe["codigo"].astype(int)

    dataframe = (
        dataframe.sort_values(
            [
                "status_classificacao",
                "score_relevancia_tecnica",
                "categoria_tecnica",
                "descricao",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )
        .drop_duplicates(subset=["codigo"])
        .reset_index(drop=True)
    )

    candidates = dataframe.loc[
        dataframe["status_classificacao"].isin(
            [
                "ADERENTE",
                "REVISAR",
                "ESPECIALIZADO",
            ]
        )
    ].copy()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_psv = output_dir / "catmat_medidores_ampliado_completo.psv"
    candidate_psv = output_dir / "catmat_medidores_candidatos.psv"
    candidate_csv = output_dir / "catmat_medidores_candidatos.csv"

    save_psv(
        dataframe,
        complete_psv,
    )
    save_psv(
        candidates,
        candidate_psv,
    )
    candidates.to_csv(
        candidate_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Itens retornados pelos PDMs:",
        len(dataframe),
    )
    print(
        "Candidatos mantidos para revisão:",
        len(candidates),
    )
    print()
    print("Resumo por classificação:")
    print(dataframe["status_classificacao"].value_counts().to_string())
    print()
    print("Resumo por categoria técnica:")
    print(candidates["categoria_tecnica"].value_counts().to_string())
    print()
    print(
        "Arquivo para cobertura:",
        candidate_csv.resolve(),
    )

    return candidate_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Amplia a descoberta de PDMs e CATMATs de medidores de energia.")
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    pdms = subparsers.add_parser(
        "pdms",
        help=("Consulta todos os PDMs ativos e gera candidatos ampliados."),
    )
    pdms.add_argument(
        "--minimum-score",
        type=int,
        default=30,
    )
    pdms.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/silver/descoberta_medidores"),
    )

    items = subparsers.add_parser(
        "items",
        help=("Consulta os itens dos PDMs selecionados ou aptos."),
    )
    items.add_argument(
        "--pdm-file",
        type=Path,
        required=True,
    )
    items.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/silver/descoberta_medidores"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = Settings()
    settings.ensure_directories()

    client = ComprasAPIClient(
        timeout_seconds=(settings.timeout_seconds),
        delay_seconds=(settings.request_delay_seconds),
        page_size=settings.page_size,
        max_pages=settings.max_pages,
    )

    if args.command == "pdms":
        scan_pdms(
            settings,
            client,
            minimum_score=args.minimum_score,
            output_dir=args.output_dir,
        )
        return

    if args.command == "items":
        scan_items(
            settings,
            client,
            pdm_file=args.pdm_file,
            output_dir=args.output_dir,
        )
        return

    raise SystemExit(f"Comando desconhecido: {args.command}")


if __name__ == "__main__":
    main()
