from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .api_client import ComprasAPIClient
from .quality import apply_quality_rules, flag_price_outliers
from .reporting import save_psv
from .settings import Settings
from .storage import build_duckdb, save_parquet
from .transform import build_practiced_price_row

LOGGER = logging.getLogger(__name__)


class ProcurementPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.client = ComprasAPIClient(
            timeout_seconds=settings.timeout_seconds,
            delay_seconds=settings.request_delay_seconds,
            page_size=settings.page_size,
            max_pages=settings.max_pages,
        )

    def reset_price_outputs(self) -> None:
        paths = [
            self.settings.bronze_dir / "precos_praticados",
            self.settings.silver_dir / "precos_praticados.parquet",
            self.settings.silver_dir / "precos_praticados.csv",
            self.settings.silver_dir / "precos_praticados.psv",
            self.settings.silver_dir / "relatorio_extracao.csv",
            self.settings.silver_dir / "relatorio_extracao.psv",
            self.settings.gold_dir / "fact_compras.parquet",
            self.settings.gold_dir / "dim_material.parquet",
            self.settings.gold_dir / "relatorio_qualidade.csv",
            self.settings.gold_dir / "relatorio_qualidade.psv",
            self.settings.gold_dir / "outliers_preco.csv",
            self.settings.gold_dir / "outliers_preco.psv",
            self.settings.gold_dir / "historico_itens.psv",
            self.settings.gold_dir / "iaae.duckdb",
        ]

        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

        LOGGER.info("Saídas operacionais anteriores removidas.")

    @staticmethod
    def _material_lookup(
        materials: pd.DataFrame,
        max_materials: int | None,
    ) -> dict[int, dict[str, Any]]:
        relevant = materials.loc[
            materials["material_relevante"].fillna(False) & materials["codigo"].notna()
        ].copy()

        if max_materials:
            relevant = relevant.head(max_materials)

        lookup: dict[int, dict[str, Any]] = {}

        for _, row in relevant.iterrows():
            code = int(row["codigo"])
            lookup[code] = {str(key): value for key, value in row.to_dict().items()}

        if not lookup:
            raise RuntimeError("Nenhum CATMAT relevante foi encontrado no catálogo.")

        return lookup

    def extract_practiced_prices(
        self,
        materials: pd.DataFrame,
        start_date: date,
        end_date: date,
        max_materials: int | None = None,
        state: str | None = None,
        uasg: int | None = None,
    ) -> pd.DataFrame:
        material_lookup = self._material_lookup(materials, max_materials)
        purchases: list[dict[str, Any]] = []
        extraction_rows: list[dict[str, Any]] = []

        start_timestamp = pd.Timestamp(start_date, tz="UTC")
        end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

        for material_code, material in material_lookup.items():
            LOGGER.info(
                "Consultando preços do CATMAT %s — %s",
                material_code,
                material.get("descricao"),
            )

            params: dict[str, Any] = {
                "codigoItemCatalogo": material_code,
                "dataResultado": "true",
            }
            if state:
                params["estado"] = state.upper()
            if uasg:
                params["codigoUasg"] = uasg

            material_rows: list[dict[str, Any]] = []
            pages = 0
            raw_records = 0

            for page_number, payload, records in self.client.paginate(
                self.settings.practiced_prices_url,
                params=params,
            ):
                pages += 1
                raw_records += len(records)
                raw_path = (
                    self.settings.bronze_dir
                    / "precos_praticados"
                    / f"catmat_{material_code}"
                    / f"pagina_{page_number:04d}.json"
                )
                self.client.save_raw_payload(raw_path, payload)
                material_rows.extend(
                    build_practiced_price_row(record, material_lookup) for record in records
                )

            material_df = pd.DataFrame(material_rows)

            if material_df.empty:
                extraction_rows.append(
                    {
                        "codigo_catmat": material_code,
                        "descricao": material.get("descricao"),
                        "paginas": pages,
                        "registros_api": raw_records,
                        "registros_periodo": 0,
                        "fora_periodo": 0,
                        "sem_data": 0,
                        "observacoes_duplicadas": 0,
                        "linhas_com_item_historico": 0,
                        "itens_unicos_com_historico": 0,
                    }
                )
                continue

            dates = pd.to_datetime(
                material_df["data_publicacao"],
                errors="coerce",
                utc=True,
            )
            valid_period = dates.ge(start_timestamp) & dates.lt(end_exclusive)
            without_date = dates.isna()
            observation_duplicates = material_df["purchase_id"].duplicated(keep=False)

            item_key = pd.Series(
                pd.NA,
                index=material_df.index,
                dtype="string",
            )
            for column in (
                "id_compra_item",
                "id_item_compra",
            ):
                if column not in material_df.columns:
                    continue
                values = (
                    material_df[column]
                    .astype("string")
                    .str.strip()
                    .replace(
                        {
                            "": pd.NA,
                            "nan": pd.NA,
                            "None": pd.NA,
                        }
                    )
                )
                item_key = item_key.combine_first(values)

            item_history = item_key.notna() & item_key.duplicated(keep=False)
            history_counts = item_key.loc[item_history].value_counts()

            extraction_rows.append(
                {
                    "codigo_catmat": material_code,
                    "descricao": material.get("descricao"),
                    "paginas": pages,
                    "registros_api": raw_records,
                    "registros_periodo": int(valid_period.sum()),
                    "fora_periodo": int((~valid_period & ~without_date).sum()),
                    "sem_data": int(without_date.sum()),
                    "observacoes_duplicadas": int(observation_duplicates.sum()),
                    "linhas_com_item_historico": int(item_history.sum()),
                    "itens_unicos_com_historico": int(len(history_counts)),
                }
            )

            purchases.extend(material_df.loc[valid_period].to_dict(orient="records"))

        dataframe = pd.DataFrame(purchases)
        extraction_report = pd.DataFrame(extraction_rows)
        save_psv(
            extraction_report,
            self.settings.silver_dir / "relatorio_extracao.psv",
        )

        if dataframe.empty:
            LOGGER.warning("Nenhum preço praticado foi mantido no período.")
            return dataframe

        dataframe = dataframe.sort_values(
            ["data_publicacao", "codigo_catmat", "purchase_id"]
        ).reset_index(drop=True)

        save_parquet(
            dataframe,
            self.settings.silver_dir / "precos_praticados.parquet",
        )
        save_psv(
            dataframe,
            self.settings.silver_dir / "precos_praticados.psv",
        )

        LOGGER.info(
            "%s preços mantidos entre %s e %s. Nenhuma linha foi deduplicada silenciosamente.",
            len(dataframe),
            start_date,
            end_date,
        )
        return dataframe

    def process_and_publish(
        self,
        materials: pd.DataFrame,
        purchases: pd.DataFrame,
    ) -> dict[str, Path]:
        if purchases.empty:
            raise RuntimeError(
                "Nenhum item de compra foi extraído. Consulte o relatório de extração."
            )

        quality_df, quality_report = apply_quality_rules(purchases)
        fact = flag_price_outliers(quality_df)
        dim_material = materials.loc[materials["material_relevante"].fillna(False)].copy()

        fact_path = self.settings.gold_dir / "fact_compras.parquet"
        dim_path = self.settings.gold_dir / "dim_material.parquet"
        report_path = self.settings.gold_dir / "relatorio_qualidade.psv"
        outliers_path = self.settings.gold_dir / "outliers_preco.psv"
        history_path = self.settings.gold_dir / "historico_itens.psv"
        database_path = self.settings.gold_dir / "iaae.duckdb"

        save_parquet(fact, fact_path)
        save_parquet(dim_material, dim_path)
        report_path = save_psv(quality_report, report_path)
        outliers_path = save_psv(
            fact.loc[fact["is_price_outlier"]],
            outliers_path,
        )
        history_path = save_psv(
            fact.loc[fact["hist_item_repetido"]],
            history_path,
        )
        build_duckdb(self.settings.gold_dir, database_path)

        return {
            "fact_compras": fact_path,
            "dim_material": dim_path,
            "relatorio_qualidade": report_path,
            "outliers_preco": outliers_path,
            "historico_itens": history_path,
            "relatorio_extracao": self.settings.silver_dir / "relatorio_extracao.psv",
            "duckdb": database_path,
        }
