from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

save_psv = import_module("src.reporting").save_psv


HISTORY_REQUIRED_COLUMNS = {
    "purchase_id",
    "data_publicacao",
    "codigo_catmat",
    "familia_material",
    "quantidade",
    "valor_unitario",
    "valor_total",
    "unidade_comparavel",
}

FORECAST_REQUIRED_COLUMNS = {
    "mes",
    "codigo_catmat",
    "familia_material",
    "modelo_selecionado",
    "confianca_previsao",
    "tipo_demanda",
    "quantidade_cenario_inferior",
    "quantidade_prevista_base",
    "quantidade_cenario_superior",
}

SUMMARY_REQUIRED_COLUMNS = {
    "codigo_catmat",
    "familia_material",
    "modelo_selecionado",
    "confianca_previsao",
    "tipo_demanda",
    "previsao_12m_cenario_inferior",
    "previsao_12m_base",
    "previsao_12m_cenario_superior",
}


def validate_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes em {name}: " + ", ".join(missing))


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def first_existing(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    return next(
        (column for column in candidates if column in dataframe.columns),
        None,
    )


def series_or_default(
    dataframe: pd.DataFrame,
    candidates: list[str],
    default: Any = pd.NA,
) -> pd.Series:
    column = first_existing(dataframe, candidates)
    if column is None:
        return pd.Series(
            default,
            index=dataframe.index,
        )
    return dataframe[column]


def mode_or_first(series: pd.Series) -> Any:
    values = clean_text(series).dropna()
    if values.empty:
        return pd.NA

    modes = values.mode()
    if not modes.empty:
        return modes.iloc[0]

    return values.iloc[0]


def normalize_key(
    series: pd.Series,
    missing_value: str,
) -> pd.Series:
    return clean_text(series).fillna(missing_value)


def prepare_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    validate_columns(
        history,
        HISTORY_REQUIRED_COLUMNS,
        "fact_compras",
    )

    source = history.copy()

    source["data_publicacao"] = pd.to_datetime(
        source["data_publicacao"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    source["codigo_catmat"] = pd.to_numeric(
        source["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    for column in [
        "quantidade",
        "valor_unitario",
        "valor_total",
    ]:
        source[column] = pd.to_numeric(
            source[column],
            errors="coerce",
        )

    valid = (
        source["purchase_id"].notna()
        & source["data_publicacao"].notna()
        & source["codigo_catmat"].notna()
    )
    source = source.loc[valid].copy()

    source["FornecedorKey"] = normalize_key(
        series_or_default(
            source,
            [
                "cnpj_fornecedor",
                "fornecedor_id",
                "ni_fornecedor",
            ],
        ),
        "FORNECEDOR_NAO_INFORMADO",
    )

    source["UASGKey"] = normalize_key(
        series_or_default(
            source,
            [
                "uasg",
                "codigo_uasg",
            ],
        ),
        "UASG_NAO_INFORMADA",
    )

    source["UnidadeKey"] = (
        source["codigo_catmat"].astype("string")
        + "|"
        + normalize_key(
            source["unidade_comparavel"],
            "UNIDADE_NAO_INFORMADA",
        )
    )

    fact = pd.DataFrame(
        {
            "CompraID": clean_text(source["purchase_id"]),
            "IdCompra": clean_text(
                series_or_default(
                    source,
                    ["id_compra"],
                )
            ),
            "IdItemCompra": clean_text(
                series_or_default(
                    source,
                    ["id_item_compra"],
                )
            ),
            "IdCompraItem": clean_text(
                series_or_default(
                    source,
                    ["id_compra_item"],
                )
            ),
            "NumeroItem": pd.to_numeric(
                series_or_default(
                    source,
                    [
                        "numero_item",
                        "numero_item_compra",
                    ],
                ),
                errors="coerce",
            ).astype("Int64"),
            "Data": source["data_publicacao"].dt.normalize(),
            "DataKey": source["data_publicacao"].dt.strftime("%Y%m%d").astype(int),
            "MesKey": source["data_publicacao"].dt.strftime("%Y%m").astype(int),
            "CodigoCATMAT": source["codigo_catmat"].astype("Int64"),
            "FornecedorKey": source["FornecedorKey"],
            "UASGKey": source["UASGKey"],
            "UnidadeKey": source["UnidadeKey"],
            "Quantidade": source["quantidade"],
            "ValorUnitario": source["valor_unitario"],
            "ValorTotal": source["valor_total"],
            "TipoPreco": clean_text(
                series_or_default(
                    source,
                    ["tipo_preco"],
                )
            ),
            "Estado": clean_text(
                series_or_default(
                    source,
                    [
                        "uf_uasg",
                        "estado",
                        "uf",
                        "sigla_uf",
                    ],
                )
            ),
            "Municipio": clean_text(
                series_or_default(
                    source,
                    [
                        "municipio",
                        "nome_municipio",
                    ],
                )
            ),
            "Poder": clean_text(
                series_or_default(
                    source,
                    ["poder"],
                )
            ),
            "Esfera": clean_text(
                series_or_default(
                    source,
                    ["esfera"],
                )
            ),
            "FormaCompra": clean_text(
                series_or_default(
                    source,
                    [
                        "forma_compra",
                        "forma_de_compra",
                    ],
                )
            ),
            "Modalidade": clean_text(
                series_or_default(
                    source,
                    ["modalidade"],
                )
            ),
            "CriterioJulgamento": clean_text(
                series_or_default(
                    source,
                    [
                        "criterio_julgamento",
                        "criterio_de_julgamento",
                    ],
                )
            ),
            "HistoricoItem": (
                series_or_default(
                    source,
                    ["hist_item_repetido"],
                    False,
                )
                .fillna(False)
                .astype(bool)
            ),
            "DQDuplicado": (
                series_or_default(
                    source,
                    ["dq_duplicado"],
                    False,
                )
                .fillna(False)
                .astype(bool)
            ),
            "DQPossuiErro": (
                series_or_default(
                    source,
                    ["dq_possui_erro"],
                    False,
                )
                .fillna(False)
                .astype(bool)
            ),
            "AlertaPreco": (
                series_or_default(
                    source,
                    ["is_price_outlier"],
                    False,
                )
                .fillna(False)
                .astype(bool)
            ),
            "GravidadeAlertaPreco": clean_text(
                series_or_default(
                    source,
                    [
                        "outlier_gravidade",
                        "gravidade_alerta",
                    ],
                )
            ),
            "Fonte": clean_text(
                series_or_default(
                    source,
                    ["fonte"],
                )
            ),
        }
    )

    return fact


def build_material_dimension(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    history_base = pd.DataFrame(
        {
            "CodigoCATMAT": pd.to_numeric(
                history["codigo_catmat"],
                errors="coerce",
            ).astype("Int64"),
            "FamiliaMaterial": clean_text(history["familia_material"]),
            "DescricaoMaterial": clean_text(
                series_or_default(
                    history,
                    [
                        "descricao_material",
                        "descricao_item",
                        "descricao",
                    ],
                )
            ),
            "CodigoPDM": clean_text(
                series_or_default(
                    history,
                    ["codigo_pdm"],
                )
            ),
            "CodigoClasse": clean_text(
                series_or_default(
                    history,
                    ["codigo_classe"],
                )
            ),
        }
    )

    forecast_base = pd.DataFrame(
        {
            "CodigoCATMAT": pd.to_numeric(
                forecast["codigo_catmat"],
                errors="coerce",
            ).astype("Int64"),
            "FamiliaMaterial": clean_text(forecast["familia_material"]),
            "DescricaoMaterial": clean_text(
                series_or_default(
                    forecast,
                    [
                        "descricao_catmat",
                        "descricao_material",
                    ],
                )
            ),
            "CodigoPDM": pd.NA,
            "CodigoClasse": pd.NA,
        }
    )

    combined = pd.concat(
        [history_base, forecast_base],
        ignore_index=True,
    )
    combined = combined.loc[combined["CodigoCATMAT"].notna()].copy()

    dimension = (
        combined.groupby(
            "CodigoCATMAT",
            as_index=False,
            dropna=False,
        )
        .agg(
            FamiliaMaterial=(
                "FamiliaMaterial",
                mode_or_first,
            ),
            DescricaoMaterial=(
                "DescricaoMaterial",
                mode_or_first,
            ),
            CodigoPDM=(
                "CodigoPDM",
                mode_or_first,
            ),
            CodigoClasse=(
                "CodigoClasse",
                mode_or_first,
            ),
        )
        .sort_values("CodigoCATMAT")
    )

    dimension["MaterialLabel"] = (
        dimension["CodigoCATMAT"].astype("string")
        + " - "
        + dimension["DescricaoMaterial"].fillna("Descrição não informada")
    )

    return dimension


def build_supplier_dimension(
    history: pd.DataFrame,
) -> pd.DataFrame:
    supplier_key = normalize_key(
        series_or_default(
            history,
            [
                "cnpj_fornecedor",
                "fornecedor_id",
                "ni_fornecedor",
            ],
        ),
        "FORNECEDOR_NAO_INFORMADO",
    )
    supplier_name = clean_text(
        series_or_default(
            history,
            [
                "fornecedor_nome",
                "nome_fornecedor",
                "fornecedor",
            ],
        )
    )

    base = pd.DataFrame(
        {
            "FornecedorKey": supplier_key,
            "FornecedorNome": supplier_name,
        }
    )

    dimension = (
        base.groupby(
            "FornecedorKey",
            as_index=False,
        )
        .agg(
            FornecedorNome=(
                "FornecedorNome",
                mode_or_first,
            )
        )
        .sort_values("FornecedorKey")
    )

    dimension["FornecedorNome"] = dimension["FornecedorNome"].fillna("Fornecedor não informado")

    return dimension


def build_uasg_dimension(
    history: pd.DataFrame,
) -> pd.DataFrame:
    uasg_key = normalize_key(
        series_or_default(
            history,
            [
                "uasg",
                "codigo_uasg",
            ],
        ),
        "UASG_NAO_INFORMADA",
    )

    base = pd.DataFrame(
        {
            "UASGKey": uasg_key,
            "Orgao": clean_text(
                series_or_default(
                    history,
                    [
                        "orgao",
                        "nome_orgao",
                    ],
                )
            ),
            "Estado": clean_text(
                series_or_default(
                    history,
                    [
                        "uf_uasg",
                        "estado",
                        "uf",
                        "sigla_uf",
                    ],
                )
            ),
            "Municipio": clean_text(
                series_or_default(
                    history,
                    [
                        "municipio",
                        "nome_municipio",
                    ],
                )
            ),
            "Poder": clean_text(
                series_or_default(
                    history,
                    ["poder"],
                )
            ),
            "Esfera": clean_text(
                series_or_default(
                    history,
                    ["esfera"],
                )
            ),
        }
    )

    dimension = (
        base.groupby(
            "UASGKey",
            as_index=False,
        )
        .agg(
            Orgao=("Orgao", mode_or_first),
            Estado=("Estado", mode_or_first),
            Municipio=("Municipio", mode_or_first),
            Poder=("Poder", mode_or_first),
            Esfera=("Esfera", mode_or_first),
        )
        .sort_values("UASGKey")
    )

    return dimension


def build_unit_dimension(
    history: pd.DataFrame,
) -> pd.DataFrame:
    catmat = pd.to_numeric(
        history["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")
    comparable = normalize_key(
        history["unidade_comparavel"],
        "UNIDADE_NAO_INFORMADA",
    )

    base = pd.DataFrame(
        {
            "UnidadeKey": (catmat.astype("string") + "|" + comparable),
            "CodigoCATMAT": catmat,
            "UnidadeComparavel": comparable,
            "SiglaUnidadeFornecimento": clean_text(
                series_or_default(
                    history,
                    [
                        "sigla_unidade_fornecimento",
                    ],
                )
            ),
            "NomeUnidadeFornecimento": clean_text(
                series_or_default(
                    history,
                    [
                        "nome_unidade_fornecimento",
                    ],
                )
            ),
            "CapacidadeUnidadeFornecimento": pd.to_numeric(
                series_or_default(
                    history,
                    [
                        "capacidade_unidade_fornecimento",
                    ],
                ),
                errors="coerce",
            ),
        }
    )

    dimension = (
        base.groupby(
            "UnidadeKey",
            as_index=False,
        )
        .agg(
            CodigoCATMAT=(
                "CodigoCATMAT",
                "first",
            ),
            UnidadeComparavel=(
                "UnidadeComparavel",
                mode_or_first,
            ),
            SiglaUnidadeFornecimento=(
                "SiglaUnidadeFornecimento",
                mode_or_first,
            ),
            NomeUnidadeFornecimento=(
                "NomeUnidadeFornecimento",
                mode_or_first,
            ),
            CapacidadeUnidadeFornecimento=(
                "CapacidadeUnidadeFornecimento",
                "first",
            ),
        )
        .sort_values(
            [
                "CodigoCATMAT",
                "UnidadeComparavel",
            ]
        )
    )

    return dimension


def prepare_forecast(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    validate_columns(
        forecast,
        FORECAST_REQUIRED_COLUMNS,
        "previsao_catmat_v2",
    )

    source = forecast.copy()
    source["mes"] = pd.to_datetime(
        source["mes"],
        errors="coerce",
    ).dt.normalize()
    source["codigo_catmat"] = pd.to_numeric(
        source["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    scenario_columns = {
        "INFERIOR": ("quantidade_cenario_inferior"),
        "BASE": ("quantidade_prevista_base"),
        "SUPERIOR": ("quantidade_cenario_superior"),
    }

    rows: list[pd.DataFrame] = []

    common = pd.DataFrame(
        {
            "Data": source["mes"],
            "DataKey": source["mes"].dt.strftime("%Y%m%d").astype("Int64"),
            "MesKey": source["mes"].dt.strftime("%Y%m").astype("Int64"),
            "CodigoCATMAT": source["codigo_catmat"],
            "ModeloSelecionado": clean_text(source["modelo_selecionado"]),
            "ConfiancaPrevisao": clean_text(source["confianca_previsao"]),
            "TipoDemanda": clean_text(source["tipo_demanda"]),
            "ErroReferenciaMensal": pd.to_numeric(
                series_or_default(
                    source,
                    [
                        "erro_referencia_mensal",
                    ],
                ),
                errors="coerce",
            ),
        }
    )

    for scenario_key, source_column in scenario_columns.items():
        scenario = common.copy()
        scenario["CenarioKey"] = scenario_key
        scenario["QuantidadePrevista"] = pd.to_numeric(
            source[source_column],
            errors="coerce",
        )
        rows.append(scenario)

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    return result.loc[result["Data"].notna() & result["CodigoCATMAT"].notna()].copy()


def prepare_forecast_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    validate_columns(
        summary,
        SUMMARY_REQUIRED_COLUMNS,
        "resumo_previsao_catmat_v2",
    )

    result = pd.DataFrame(
        {
            "CodigoCATMAT": pd.to_numeric(
                summary["codigo_catmat"],
                errors="coerce",
            ).astype("Int64"),
            "ModeloSelecionado": clean_text(summary["modelo_selecionado"]),
            "ConfiancaPrevisao": clean_text(summary["confianca_previsao"]),
            "TipoDemanda": clean_text(summary["tipo_demanda"]),
            "Previsao12MCenarioInferior": pd.to_numeric(
                summary["previsao_12m_cenario_inferior"],
                errors="coerce",
            ),
            "Previsao12MBase": pd.to_numeric(
                summary["previsao_12m_base"],
                errors="coerce",
            ),
            "Previsao12MCenarioSuperior": pd.to_numeric(
                summary["previsao_12m_cenario_superior"],
                errors="coerce",
            ),
            "QuantidadeHistoricaTotal": pd.to_numeric(
                series_or_default(
                    summary,
                    [
                        "quantidade_historica_total",
                    ],
                ),
                errors="coerce",
            ),
            "MediaMensalPrevistaBase": pd.to_numeric(
                series_or_default(
                    summary,
                    [
                        "media_mensal_prevista_base",
                    ],
                ),
                errors="coerce",
            ),
            "WAPEValidacaoPercentual": pd.to_numeric(
                series_or_default(
                    summary,
                    [
                        "wape_validacao_percentual",
                    ],
                ),
                errors="coerce",
            ),
            "MAEValidacao": pd.to_numeric(
                series_or_default(
                    summary,
                    ["mae_validacao"],
                ),
                errors="coerce",
            ),
            "MesesComDemanda": pd.to_numeric(
                series_or_default(
                    summary,
                    ["meses_com_demanda"],
                ),
                errors="coerce",
            ).astype("Int64"),
            "PercentualMesesSemDemanda": pd.to_numeric(
                series_or_default(
                    summary,
                    [
                        "percentual_meses_sem_demanda",
                    ],
                ),
                errors="coerce",
            ),
            "JanelasValidacao": pd.to_numeric(
                series_or_default(
                    summary,
                    ["janelas_validacao"],
                ),
                errors="coerce",
            ).astype("Int64"),
            "ProximosPassos": clean_text(
                series_or_default(
                    summary,
                    ["proximos_passos"],
                )
            ),
        }
    )

    return result.loc[result["CodigoCATMAT"].notna()].drop_duplicates(
        subset=["CodigoCATMAT"],
        keep="first",
    )


def build_calendar_dimension(
    history_fact: pd.DataFrame,
    forecast_fact: pd.DataFrame,
) -> pd.DataFrame:
    start = min(
        history_fact["Data"].min(),
        forecast_fact["Data"].min(),
    )
    end = max(
        history_fact["Data"].max(),
        forecast_fact["Data"].max(),
    )

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
    )

    month_names = {
        1: "Jan",
        2: "Fev",
        3: "Mar",
        4: "Abr",
        5: "Mai",
        6: "Jun",
        7: "Jul",
        8: "Ago",
        9: "Set",
        10: "Out",
        11: "Nov",
        12: "Dez",
    }

    dimension = pd.DataFrame(
        {
            "Data": dates,
        }
    )

    dimension["DataKey"] = dimension["Data"].dt.strftime("%Y%m%d").astype(int)
    dimension["Ano"] = dimension["Data"].dt.year
    dimension["NumeroMes"] = dimension["Data"].dt.month
    dimension["Mes"] = dimension["NumeroMes"].map(month_names)
    dimension["MesAno"] = dimension["Mes"] + "/" + dimension["Ano"].astype(str)
    dimension["MesAnoOrdenacao"] = dimension["Data"].dt.strftime("%Y%m").astype(int)
    dimension["InicioMes"] = dimension["Data"].dt.to_period("M").dt.to_timestamp()
    dimension["Trimestre"] = "T" + dimension["Data"].dt.quarter.astype(str)
    dimension["AnoTrimestre"] = dimension["Ano"].astype(str) + "-" + dimension["Trimestre"]
    dimension["Semestre"] = np.where(
        dimension["NumeroMes"] <= 6,
        "S1",
        "S2",
    )
    dimension["Dia"] = dimension["Data"].dt.day
    dimension["DiaSemanaNumero"] = dimension["Data"].dt.dayofweek + 1
    day_names = {
        1: "Seg",
        2: "Ter",
        3: "Qua",
        4: "Qui",
        5: "Sex",
        6: "Sáb",
        7: "Dom",
    }
    dimension["DiaSemana"] = dimension["DiaSemanaNumero"].map(day_names)
    dimension["FimDeSemana"] = dimension["DiaSemanaNumero"] >= 6

    return dimension


def scenario_dimension() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CenarioKey": "INFERIOR",
                "Cenario": "Inferior",
                "OrdemCenario": 1,
            },
            {
                "CenarioKey": "BASE",
                "Cenario": "Base",
                "OrdemCenario": 2,
            },
            {
                "CenarioKey": "SUPERIOR",
                "Cenario": "Superior",
                "OrdemCenario": 3,
            },
        ]
    )


def save_table(
    dataframe: pd.DataFrame,
    output_dir: Path,
    table_name: str,
) -> dict[str, str]:
    parquet_path = output_dir / f"{table_name}.parquet"
    psv_path = output_dir / f"{table_name}.psv"

    dataframe.to_parquet(
        parquet_path,
        index=False,
    )
    save_psv(
        dataframe,
        psv_path,
    )

    return {
        "parquet": str(parquet_path.resolve()),
        "psv": str(psv_path.resolve()),
        "linhas": str(len(dataframe)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Prepara tabelas estrela para o modelo analítico do Power BI.")
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--forecast",
        type=Path,
        default=Path("data/gold/previsao_catmat_v2/previsao_catmat_v2.parquet"),
    )
    parser.add_argument(
        "--forecast-summary",
        type=Path,
        default=Path("data/gold/previsao_catmat_v2/resumo_previsao_catmat_v2.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/powerbi"),
    )
    args = parser.parse_args()

    for path in (
        args.history,
        args.forecast,
        args.forecast_summary,
    ):
        if not path.exists():
            raise SystemExit(f"Arquivo não encontrado: {path.resolve()}")

    history = pd.read_parquet(args.history)
    forecast = pd.read_parquet(args.forecast)
    summary = pd.read_parquet(args.forecast_summary)

    history_fact = prepare_history(history)
    forecast_fact = prepare_forecast(forecast)
    forecast_summary = prepare_forecast_summary(summary)

    dim_material = build_material_dimension(
        history,
        forecast,
    )
    dim_supplier = build_supplier_dimension(
        history,
    )
    dim_uasg = build_uasg_dimension(history)
    dim_unit = build_unit_dimension(history)
    dim_calendar = build_calendar_dimension(
        history_fact,
        forecast_fact,
    )
    dim_scenario = scenario_dimension()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = {
        "FatoCompras": history_fact,
        "FatoPrevisao": forecast_fact,
        "ResumoPrevisao": forecast_summary,
        "DimCalendario": dim_calendar,
        "DimMaterial": dim_material,
        "DimFornecedor": dim_supplier,
        "DimUASG": dim_uasg,
        "DimUnidade": dim_unit,
        "DimCenario": dim_scenario,
    }

    manifest: dict[str, Any] = {
        "tabelas": {},
        "relacionamentos": [
            {
                "de": "DimCalendario[Data]",
                "para": "FatoCompras[Data]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimCalendario[Data]",
                "para": "FatoPrevisao[Data]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimMaterial[CodigoCATMAT]",
                "para": "FatoCompras[CodigoCATMAT]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimMaterial[CodigoCATMAT]",
                "para": "FatoPrevisao[CodigoCATMAT]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimMaterial[CodigoCATMAT]",
                "para": "ResumoPrevisao[CodigoCATMAT]",
                "cardinalidade": "1:1",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimFornecedor[FornecedorKey]",
                "para": "FatoCompras[FornecedorKey]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimUASG[UASGKey]",
                "para": "FatoCompras[UASGKey]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimUnidade[UnidadeKey]",
                "para": "FatoCompras[UnidadeKey]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
            {
                "de": "DimCenario[CenarioKey]",
                "para": "FatoPrevisao[CenarioKey]",
                "cardinalidade": "1:*",
                "direcao": "Única",
                "ativo": True,
            },
        ],
    }

    for table_name, dataframe in tables.items():
        manifest["tabelas"][table_name] = save_table(
            dataframe,
            args.output_dir,
            table_name,
        )

    manifest_path = args.output_dir / "manifesto_modelo_powerbi.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Modelo Power BI preparado.")
    print(
        "Diretório:",
        args.output_dir.resolve(),
    )
    print()
    for table_name, dataframe in tables.items():
        print(f"{table_name}: {len(dataframe)} linha(s)")
    print()
    print(
        "Manifesto:",
        manifest_path.resolve(),
    )


if __name__ == "__main__":
    main()
