from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.api_client import extract_records
from src.pipeline import ProcurementPipeline
from src.settings import Settings


def read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8-sig", "sep": "|"},
        {"encoding": "cp1252", "sep": None, "engine": "python"},
    ]

    last_error: Exception | None = None
    for options in attempts:
        try:
            dataframe = pd.read_csv(path, **options)
            if len(dataframe.columns) > 1:
                return dataframe
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Não foi possível ler {path}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline IAAE de preços públicos de materiais elétricos"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Executar extração, qualidade e publicação",
    )
    run.add_argument("--start-date", required=True, type=date.fromisoformat)
    run.add_argument("--end-date", required=True, type=date.fromisoformat)
    run.add_argument("--max-materials", type=int, default=None)
    run.add_argument("--state", default=None)
    run.add_argument("--uasg", type=int, default=None)
    run.add_argument("--catalog-file", required=True, type=Path)
    run.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove somente saídas operacionais de preços antes da nova consulta",
    )

    rebuild = subparsers.add_parser(
        "rebuild",
        help="Reconstruir Gold a partir da Silver existente, sem consultar a API",
    )
    rebuild.add_argument("--catalog-file", required=True, type=Path)
    rebuild.add_argument(
        "--silver-file",
        type=Path,
        default=Path("data/silver/precos_praticados.parquet"),
    )

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--code", type=int, default=610532)

    price_smoke = subparsers.add_parser("price-smoke-test")
    price_smoke.add_argument("--code", type=int, default=610539)
    price_smoke.add_argument("--uasg", type=int, default=986001)
    price_smoke.add_argument("--page-size", type=int, default=10)

    return parser.parse_args()


def configure_logging(settings: Settings) -> None:
    settings.ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                settings.log_dir / "pipeline.log",
                encoding="utf-8",
            ),
        ],
    )


def normalize_material_relevance(materials: pd.DataFrame) -> pd.DataFrame:
    result = materials.copy()

    if "material_relevante" not in result.columns:
        result["material_relevante"] = True
        return result

    if result["material_relevante"].dtype == bool:
        return result

    result["material_relevante"] = (
        result["material_relevante"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "sim": True,
                "yes": True,
                "false": False,
                "0": False,
                "nao": False,
                "não": False,
                "no": False,
            }
        )
        .fillna(False)
    )
    return result


def main() -> None:
    args = parse_args()
    settings = Settings()
    configure_logging(settings)
    pipeline = ProcurementPipeline(settings)

    if args.command == "smoke-test":
        payload = pipeline.client.get_json(
            settings.materials_url,
            {
                "pagina": 1,
                "tamanhoPagina": 10,
                "codigoItem": args.code,
            },
        )
        records = extract_records(payload)
        print("Conexão OK.")
        print(f"Registros recebidos: {len(records)}")
        if records:
            print("Campos encontrados:", sorted(records[0].keys()))
        return

    if args.command == "price-smoke-test":
        if not 10 <= args.page_size <= 500:
            raise SystemExit("--page-size deve estar entre 10 e 500.")

        payload = pipeline.client.get_json(
            settings.practiced_prices_url,
            {
                "pagina": 1,
                "tamanhoPagina": args.page_size,
                "codigoItemCatalogo": args.code,
                "codigoUasg": args.uasg,
                "dataResultado": "true",
            },
        )
        records = extract_records(payload)
        raw_path = (
            settings.bronze_dir / "diagnostico" / f"precos_catmat_{args.code}_uasg_{args.uasg}.json"
        )
        pipeline.client.save_raw_payload(raw_path, payload)

        print("Conexão com preços praticados OK.")
        print(f"Registros recebidos: {len(records)}")
        if records:
            first = records[0]
            print("Unidade de fornecimento:", first.get("siglaUnidadeFornecimento"))
            print("Nome da unidade:", first.get("nomeUnidadeFornecimento"))
            print("Capacidade:", first.get("capacidadeUnidadeFornecimento"))
            print("Unidade de medida:", first.get("siglaUnidadeMedida"))
        print(f"Resposta salva em: {raw_path.resolve()}")
        return

    if args.command == "run" and args.start_date > args.end_date:
        raise SystemExit("A data inicial não pode ser posterior à data final.")
    if not args.catalog_file.exists():
        raise SystemExit(f"Catálogo não encontrado: {args.catalog_file.resolve()}")

    if args.catalog_file.suffix.lower() == ".parquet":
        materials = pd.read_parquet(args.catalog_file)
    else:
        materials = read_csv_flexible(args.catalog_file)

    materials = normalize_material_relevance(materials)

    required_catalog_columns = {
        "codigo",
        "descricao",
        "familia_codigo",
        "familia_material",
        "material_relevante",
    }
    missing = required_catalog_columns.difference(materials.columns)
    if missing:
        raise SystemExit("Colunas ausentes no catálogo: " + ", ".join(sorted(missing)))

    if args.command == "rebuild":
        if not args.silver_file.exists():
            raise SystemExit(f"Silver não encontrada: {args.silver_file.resolve()}")
        purchases = pd.read_parquet(args.silver_file)
        print(
            "Reconstruindo Gold a partir da Silver:",
            args.silver_file.resolve(),
        )
    else:
        if args.clean_output:
            pipeline.reset_price_outputs()

        purchases = pipeline.extract_practiced_prices(
            materials,
            start_date=args.start_date,
            end_date=args.end_date,
            max_materials=args.max_materials,
            state=args.state,
            uasg=args.uasg,
        )

    paths = pipeline.process_and_publish(materials, purchases)

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
