from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NEW_METER_CODES = {
    422736,
    383951,
    383953,
    286852,
    287355,
    287357,
    356362,
}

EXCLUDED_METER_CODES = {
    383952,
    365598,
    42838,
}

POWERBI_TABLES = [
    "FatoCompras",
    "FatoPrevisao",
    "ResumoPrevisao",
    "DimCalendario",
    "DimMaterial",
    "DimFornecedor",
    "DimUASG",
    "DimUnidade",
    "DimCenario",
]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", text.upper()).strip()


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
    column = first_existing(
        dataframe,
        candidates,
    )

    if column is None:
        return pd.Series(
            default,
            index=dataframe.index,
        )

    return dataframe[column]


def format_capacity(value: Any) -> str | None:
    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric) or float(numeric) <= 0:
        return None

    numeric = float(numeric)

    if numeric.is_integer():
        return str(int(numeric))

    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def build_comparable_unit(
    unit: Any,
    unit_name: Any,
    capacity: Any,
) -> str | None:
    selected = unit

    if selected is None or pd.isna(selected) or not str(selected).strip():
        selected = unit_name

    if selected is None or pd.isna(selected) or not str(selected).strip():
        return None

    selected = str(selected).strip().upper()
    formatted_capacity = format_capacity(capacity)

    if formatted_capacity and formatted_capacity != "1":
        return f"{selected}|CAP={formatted_capacity}"

    return selected


def safe_scalar_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError, ValueError:
        pass

    return str(value)


def stable_purchase_id(
    row: pd.Series,
    position: int,
) -> str:
    parts = [
        safe_scalar_text(row.get("codigo_catmat")),
        safe_scalar_text(row.get("id_compra")),
        safe_scalar_text(row.get("id_item_compra")),
        safe_scalar_text(row.get("id_compra_item")),
        safe_scalar_text(row.get("data_referencia")),
        str(position),
    ]

    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]

    return f"MED-{digest}"


def object_value_to_reference_text(
    value: Any,
) -> str | bytes | None:
    """Converte valores de colunas object para representação estável."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except TypeError, ValueError:
        pass

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat) and not isinstance(value, str):
        try:
            return str(isoformat())
        except TypeError, ValueError:
            pass

    return str(value)


def coerce_column_to_reference(
    series: pd.Series,
    reference_series: pd.Series,
) -> pd.Series:
    """Converte uma coluna para o tipo lógico usado na referência."""
    dtype = reference_series.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype):
        converted = pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        return converted.dt.tz_convert(None)

    if pd.api.types.is_bool_dtype(dtype):
        if pd.api.types.is_bool_dtype(series.dtype):
            return series.fillna(False).astype(dtype)

        normalized = series.astype("string").str.strip().str.lower()
        return normalized.isin(
            {
                "true",
                "1",
                "sim",
                "yes",
                "y",
            }
        ).astype(dtype)

    if pd.api.types.is_integer_dtype(dtype):
        converted = pd.to_numeric(
            series,
            errors="coerce",
        )

        try:
            return converted.astype(dtype)
        except TypeError, ValueError:
            return converted.astype("Int64")

    if pd.api.types.is_float_dtype(dtype):
        converted = pd.to_numeric(
            series,
            errors="coerce",
        )
        return converted.astype(dtype)

    if pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
        return clean_text(series).astype(dtype)

    if pd.api.types.is_object_dtype(dtype):
        reference_non_null = reference_series.dropna()

        reference_uses_bytes = bool(
            not reference_non_null.empty
            and isinstance(
                reference_non_null.iloc[0],
                (bytes, bytearray),
            )
        )

        converted = series.map(object_value_to_reference_text)

        if reference_uses_bytes:
            return converted.map(
                lambda value: (
                    value
                    if isinstance(value, bytes)
                    else (str(value).encode("utf-8") if value is not None else None)
                )
            ).astype("object")

        return converted.map(
            lambda value: (
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            )
        ).astype("object")

    try:
        return series.astype(dtype)
    except TypeError, ValueError:
        return series


def align_to_schema(
    source: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Alinha nomes, ordem e tipos lógicos ao DataFrame de referência."""
    aligned = pd.DataFrame(index=source.index)

    for column in reference.columns:
        dtype = reference[column].dtype

        if column in source.columns:
            aligned[column] = coerce_column_to_reference(
                source[column],
                reference[column],
            )
            continue

        if pd.api.types.is_bool_dtype(dtype):
            aligned[column] = pd.Series(
                False,
                index=source.index,
                dtype=dtype,
            )
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            aligned[column] = pd.Series(
                pd.NaT,
                index=source.index,
                dtype=dtype,
            )
        elif pd.api.types.is_integer_dtype(dtype):
            nullable_dtype = dtype if pd.api.types.is_extension_array_dtype(dtype) else "Int64"
            aligned[column] = pd.Series(
                pd.NA,
                index=source.index,
                dtype=nullable_dtype,
            )
        elif pd.api.types.is_float_dtype(dtype):
            aligned[column] = pd.Series(
                np.nan,
                index=source.index,
                dtype=dtype,
            )
        elif pd.api.types.is_object_dtype(dtype):
            aligned[column] = pd.Series(
                None,
                index=source.index,
                dtype="object",
            )
        else:
            aligned[column] = pd.Series(
                pd.NA,
                index=source.index,
            )

    return aligned[list(reference.columns)]


def identify_meter_codes(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    summary: pd.DataFrame,
) -> set[int]:
    codes: set[int] = set(EXCLUDED_METER_CODES)

    for dataframe in [
        history,
        forecast,
        summary,
    ]:
        if "codigo_catmat" not in dataframe.columns:
            continue

        family_column = first_existing(
            dataframe,
            [
                "familia_material",
                "familia",
            ],
        )

        if family_column is None:
            continue

        family_mask = (
            dataframe[family_column]
            .map(normalize_text)
            .str.contains(
                "MEDIDOR",
                na=False,
            )
        )

        found = pd.to_numeric(
            dataframe.loc[
                family_mask,
                "codigo_catmat",
            ],
            errors="coerce",
        ).dropna()

        codes.update(int(value) for value in found)

    return codes


def resolve_family_label(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    meter_codes: set[int],
) -> str:
    values: list[str] = []

    for dataframe in [
        history,
        forecast,
    ]:
        if "codigo_catmat" not in dataframe.columns or "familia_material" not in dataframe.columns:
            continue

        codes = pd.to_numeric(
            dataframe["codigo_catmat"],
            errors="coerce",
        )

        candidates = clean_text(
            dataframe.loc[
                codes.isin(meter_codes),
                "familia_material",
            ]
        ).dropna()

        values.extend(candidates.tolist())

    if not values:
        return "Medidores Energia"

    modes = pd.Series(
        values,
        dtype="string",
    ).mode()

    if not modes.empty:
        return str(modes.iloc[0])

    return str(values[0])


def build_pilot_history(
    pilot: pd.DataFrame,
    reference: pd.DataFrame,
    family_label: str,
) -> pd.DataFrame:
    source = pilot.copy()

    for column in source.columns:
        if pd.api.types.is_object_dtype(source[column].dtype):
            source[column] = clean_text(source[column])

    source["codigo_catmat"] = pd.to_numeric(
        series_or_default(
            source,
            [
                "codigo_catmat",
                "codigo_item_catalogo",
            ],
        ),
        errors="coerce",
    ).astype("Int64")

    source["data_referencia"] = pd.to_datetime(
        series_or_default(
            source,
            [
                "data_referencia",
                "data_resultado",
                "data_compra",
            ],
        ),
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    source["quantidade"] = pd.to_numeric(
        series_or_default(
            source,
            [
                "quantidade",
                "quantidade_item",
            ],
        ),
        errors="coerce",
    )

    source["preco_unitario"] = pd.to_numeric(
        series_or_default(
            source,
            [
                "preco_unitario",
                "valor_unitario",
            ],
        ),
        errors="coerce",
    )

    source["valor_total_calculado"] = pd.to_numeric(
        series_or_default(
            source,
            [
                "valor_total_calculado",
                "valor_total",
            ],
        ),
        errors="coerce",
    )

    missing_total = source["valor_total_calculado"].isna()

    source.loc[
        missing_total,
        "valor_total_calculado",
    ] = (
        source.loc[
            missing_total,
            "quantidade",
        ]
        * source.loc[
            missing_total,
            "preco_unitario",
        ]
    )

    source["sigla_unidade_fornecimento"] = clean_text(
        series_or_default(
            source,
            [
                "sigla_unidade_fornecimento",
                "unidade_fornecimento",
            ],
        )
    )
    source["nome_unidade_fornecimento"] = clean_text(
        series_or_default(
            source,
            [
                "nome_unidade_fornecimento",
            ],
        )
    )
    source["capacidade_unidade_fornecimento"] = pd.to_numeric(
        series_or_default(
            source,
            [
                "capacidade_unidade_fornecimento",
            ],
        ),
        errors="coerce",
    )

    source["unidade_comparavel"] = [
        build_comparable_unit(
            unit,
            name,
            capacity,
        )
        for unit, name, capacity in zip(
            source["sigla_unidade_fornecimento"],
            source["nome_unidade_fornecimento"],
            source["capacidade_unidade_fornecimento"],
            strict=False,
        )
    ]

    purchase_ids = [
        stable_purchase_id(
            row,
            position,
        )
        for position, (
            _,
            row,
        ) in enumerate(
            source.iterrows(),
            start=1,
        )
    ]

    canonical = source.copy()
    canonical["purchase_id"] = purchase_ids
    canonical["data_publicacao"] = source["data_referencia"]
    canonical["familia_material"] = family_label
    canonical["valor_unitario"] = source["preco_unitario"]
    canonical["valor_total"] = source["valor_total_calculado"]

    canonical["descricao_material"] = clean_text(
        series_or_default(
            source,
            [
                "descricao_catalogo",
                "descricao_item",
                "descricao",
            ],
        )
    )

    canonical["cnpj_fornecedor"] = clean_text(
        series_or_default(
            source,
            [
                "ni_fornecedor",
                "cnpj_fornecedor",
                "fornecedor_id",
            ],
        )
    )
    canonical["fornecedor_nome"] = clean_text(
        series_or_default(
            source,
            [
                "nome_fornecedor",
                "fornecedor_nome",
                "fornecedor",
            ],
        )
    )
    canonical["uasg"] = clean_text(
        series_or_default(
            source,
            [
                "codigo_uasg",
                "uasg",
            ],
        )
    )
    canonical["orgao"] = clean_text(
        series_or_default(
            source,
            [
                "nome_uasg",
                "orgao",
                "nome_orgao",
            ],
        )
    )

    canonical["uf_uasg"] = clean_text(
        series_or_default(
            source,
            [
                "uf_uasg",
                "estado",
                "uf",
                "sigla_uf",
            ],
        )
    )
    canonical["municipio"] = clean_text(
        series_or_default(
            source,
            [
                "municipio",
                "nome_municipio",
            ],
        )
    )

    canonical["tipo_preco"] = "PRATICADO"
    canonical["fonte"] = "Compras.gov.br"
    canonical["unidade_fornecimento_informada"] = (
        canonical["sigla_unidade_fornecimento"].notna()
        | canonical["nome_unidade_fornecimento"].notna()
    )

    canonical["hist_item_repetido"] = False
    canonical["dq_duplicado"] = False
    canonical["dq_possui_erro"] = (
        canonical["data_publicacao"].isna()
        | canonical["codigo_catmat"].isna()
        | canonical["quantidade"].isna()
        | canonical["valor_unitario"].isna()
        | canonical["unidade_comparavel"].isna()
    )
    canonical["is_price_outlier"] = False
    canonical["outlier_gravidade"] = "SEM_ALERTA"

    canonical = canonical.loc[canonical["codigo_catmat"].isin(NEW_METER_CODES)].copy()

    aligned = align_to_schema(
        canonical,
        reference,
    )

    return aligned


def build_description_lookup(
    pilot_history: pd.DataFrame,
) -> dict[int, str]:
    code_column = first_existing(
        pilot_history,
        [
            "codigo_catmat",
            "codigo_item_catalogo",
        ],
    )
    description_column = first_existing(
        pilot_history,
        [
            "descricao_catalogo",
            "descricao_item",
            "descricao",
        ],
    )

    if code_column is None or description_column is None:
        return {}

    base = pd.DataFrame(
        {
            "codigo": pd.to_numeric(
                pilot_history[code_column],
                errors="coerce",
            ),
            "descricao": clean_text(pilot_history[description_column]),
        }
    ).dropna(
        subset=[
            "codigo",
            "descricao",
        ]
    )

    result: dict[int, str] = {}

    for code, group in base.groupby("codigo"):
        modes = group["descricao"].mode()

        result[int(code)] = str(modes.iloc[0] if not modes.empty else group["descricao"].iloc[0])

    return result


def build_pilot_forecast(
    pilot_forecast: pd.DataFrame,
    reference: pd.DataFrame,
    family_label: str,
    descriptions: dict[int, str],
) -> pd.DataFrame:
    source = pilot_forecast.copy()

    source["codigo_catmat"] = pd.to_numeric(
        source["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")
    source["mes"] = pd.to_datetime(
        source["data"],
        errors="coerce",
    ).dt.normalize()
    source["familia_material"] = family_label

    source["tipo_demanda"] = np.where(
        source["nivel_modelagem"].astype(str).str.upper().eq("INDIVIDUAL"),
        "INTERMITENTE_INDIVIDUAL",
        "INTERMITENTE_SEGMENTO",
    )

    source["descricao_catmat"] = source["codigo_catmat"].map(descriptions)

    source["erro_referencia_mensal"] = pd.to_numeric(
        source["quantidade_prevista_base"],
        errors="coerce",
    ) * pd.to_numeric(
        source["fator_incerteza"],
        errors="coerce",
    )

    source = source.loc[source["codigo_catmat"].isin(NEW_METER_CODES)].copy()

    return align_to_schema(
        source,
        reference,
    )


def build_pilot_summary(
    pilot_summary: pd.DataFrame,
    unit_summary: pd.DataFrame,
    weights: pd.DataFrame,
    reference: pd.DataFrame,
    family_label: str,
    descriptions: dict[int, str],
) -> pd.DataFrame:
    summary = pilot_summary.copy()
    units = unit_summary.copy()
    allocation = weights.copy()

    summary["codigo_catmat"] = pd.to_numeric(
        summary["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    allocation["codigo_catmat"] = pd.to_numeric(
        allocation["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")

    merged = summary.merge(
        units[
            [
                "unidade_modelagem",
                "wape_validacao",
                "mae_validacao",
                "janelas_validacao",
                "meses_com_demanda",
                "percentual_meses_sem_demanda",
            ]
        ],
        on="unidade_modelagem",
        how="left",
        validate="many_to_one",
    )

    merged = merged.merge(
        allocation[
            [
                "codigo_catmat",
                "quantidade_historica",
            ]
        ],
        on="codigo_catmat",
        how="left",
        validate="one_to_one",
    )

    merged["familia_material"] = family_label
    merged["descricao_catmat"] = merged["codigo_catmat"].map(descriptions)

    merged["tipo_demanda"] = np.where(
        merged["nivel_modelagem"].astype(str).str.upper().eq("INDIVIDUAL"),
        "INTERMITENTE_INDIVIDUAL",
        "INTERMITENTE_SEGMENTO",
    )

    merged["previsao_12m_cenario_inferior"] = merged["previsao_12m_inferior"]
    merged["previsao_12m_cenario_superior"] = merged["previsao_12m_superior"]
    merged["quantidade_historica_total"] = merged["quantidade_historica"]
    merged["wape_validacao_percentual"] = (
        pd.to_numeric(
            merged["wape_validacao"],
            errors="coerce",
        )
        * 100
    )
    merged["proximos_passos"] = np.where(
        merged["nivel_modelagem"].astype(str).str.upper().eq("INDIVIDUAL"),
        ("Recalibrar com consumo interno, estoque e plano de obras."),
        ("Recalibrar o segmento com consumo interno e revisar os pesos de rateio."),
    )

    merged = merged.loc[merged["codigo_catmat"].isin(NEW_METER_CODES)].copy()

    return align_to_schema(
        merged,
        reference,
    )


def save_psv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    output = dataframe.copy()

    for column in output.select_dtypes(
        include=[
            "object",
            "string",
        ]
    ).columns:
        output[column] = clean_text(output[column])

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def compare_schemas(
    production_dir: Path,
    staging_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for table in POWERBI_TABLES:
        production_path = production_dir / f"{table}.parquet"
        staging_path = staging_dir / f"{table}.parquet"

        if not production_path.exists():
            rows.append(
                {
                    "tabela": table,
                    "resultado": ("SEM_REFERENCIA_PRODUCAO"),
                    "colunas_producao": None,
                    "colunas_staging": None,
                    "ordem_colunas_igual": None,
                    "tipos_iguais": None,
                }
            )
            continue

        if not staging_path.exists():
            rows.append(
                {
                    "tabela": table,
                    "resultado": ("AUSENTE_NO_STAGING"),
                    "colunas_producao": None,
                    "colunas_staging": None,
                    "ordem_colunas_igual": False,
                    "tipos_iguais": False,
                }
            )
            continue

        production = pd.read_parquet(production_path)
        staging = pd.read_parquet(staging_path)

        same_columns = list(production.columns) == list(staging.columns)
        same_types = production.dtypes.astype(str).to_dict() == staging.dtypes.astype(str).to_dict()

        rows.append(
            {
                "tabela": table,
                "resultado": ("OK" if same_columns else "DIVERGENTE"),
                "colunas_producao": (len(production.columns)),
                "colunas_staging": (len(staging.columns)),
                "ordem_colunas_igual": (same_columns),
                "tipos_iguais": same_types,
                "linhas_producao": (len(production)),
                "linhas_staging": (len(staging)),
            }
        )

    return pd.DataFrame(rows)


def validate_staging(
    production_dir: Path,
    staging_dir: Path,
    old_meter_codes: set[int],
) -> pd.DataFrame:
    purchases = pd.read_parquet(staging_dir / "FatoCompras.parquet")
    forecast = pd.read_parquet(staging_dir / "FatoPrevisao.parquet")
    materials = pd.read_parquet(staging_dir / "DimMaterial.parquet")
    suppliers = pd.read_parquet(staging_dir / "DimFornecedor.parquet")
    uasgs = pd.read_parquet(staging_dir / "DimUASG.parquet")
    units = pd.read_parquet(staging_dir / "DimUnidade.parquet")
    calendar = pd.read_parquet(staging_dir / "DimCalendario.parquet")
    scenarios = pd.read_parquet(staging_dir / "DimCenario.parquet")

    purchase_codes = set(
        pd.to_numeric(
            purchases["CodigoCATMAT"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    forecast_codes = set(
        pd.to_numeric(
            forecast["CodigoCATMAT"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    material_codes = set(
        pd.to_numeric(
            materials["CodigoCATMAT"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    forecast_meter = forecast.loc[forecast["CodigoCATMAT"].isin(NEW_METER_CODES)].copy()

    pivot = forecast_meter.pivot_table(
        index=[
            "CodigoCATMAT",
            "Data",
        ],
        columns="CenarioKey",
        values="QuantidadePrevista",
        aggfunc="sum",
    ).reset_index()

    scenario_order_valid = True

    required_scenarios = {
        "INFERIOR",
        "BASE",
        "SUPERIOR",
    }

    if not required_scenarios.issubset(pivot.columns):
        scenario_order_valid = False
    else:
        scenario_order_valid = bool(
            (pivot["INFERIOR"] <= pivot["BASE"]).all()
            and (pivot["BASE"] <= pivot["SUPERIOR"]).all()
        )

    fact_state_filled = int(purchases["Estado"].notna().sum())
    dim_state_filled = int(uasgs["Estado"].notna().sum())
    states_represented = int(purchases["Estado"].dropna().nunique())

    validations = [
        {
            "validacao": "Estado preenchido na FatoCompras",
            "esperado": len(purchases),
            "encontrado": fact_state_filled,
            "resultado": fact_state_filled == len(purchases),
        },
        {
            "validacao": "Estado preenchido na DimUASG",
            "esperado": len(uasgs),
            "encontrado": dim_state_filled,
            "resultado": dim_state_filled == len(uasgs),
        },
        {
            "validacao": "UFs representadas",
            "esperado": 27,
            "encontrado": states_represented,
            "resultado": states_represented == 27,
        },
        {
            "validacao": ("Sete CATMATs no histórico"),
            "esperado": 7,
            "encontrado": len(NEW_METER_CODES & purchase_codes),
            "resultado": (len(NEW_METER_CODES & purchase_codes) == 7),
        },
        {
            "validacao": ("Sete CATMATs na previsão"),
            "esperado": 7,
            "encontrado": len(NEW_METER_CODES & forecast_codes),
            "resultado": (len(NEW_METER_CODES & forecast_codes) == 7),
        },
        {
            "validacao": ("Sete CATMATs na dimensão"),
            "esperado": 7,
            "encontrado": len(NEW_METER_CODES & material_codes),
            "resultado": (len(NEW_METER_CODES & material_codes) == 7),
        },
        {
            "validacao": ("Medidores antigos removidos"),
            "esperado": 0,
            "encontrado": len(
                old_meter_codes & (purchase_codes | forecast_codes) - NEW_METER_CODES
            ),
            "resultado": (
                len(old_meter_codes & (purchase_codes | forecast_codes) - NEW_METER_CODES) == 0
            ),
        },
        {
            "validacao": ("252 linhas previstas"),
            "esperado": 252,
            "encontrado": len(forecast_meter),
            "resultado": (len(forecast_meter) == 252),
        },
        {
            "validacao": ("Previsões não negativas"),
            "esperado": 0,
            "encontrado": int((forecast_meter["QuantidadePrevista"] < 0).sum()),
            "resultado": bool((forecast_meter["QuantidadePrevista"] >= 0).all()),
        },
        {
            "validacao": ("Ordem dos cenários"),
            "esperado": True,
            "encontrado": (scenario_order_valid),
            "resultado": (scenario_order_valid),
        },
        {
            "validacao": ("DimMaterial sem duplicidade"),
            "esperado": 0,
            "encontrado": int(materials["CodigoCATMAT"].duplicated().sum()),
            "resultado": bool(not materials["CodigoCATMAT"].duplicated().any()),
        },
        {
            "validacao": ("CATMATs da compra na dimensão"),
            "esperado": 0,
            "encontrado": len(purchase_codes - material_codes),
            "resultado": (purchase_codes <= material_codes),
        },
        {
            "validacao": ("CATMATs da previsão na dimensão"),
            "esperado": 0,
            "encontrado": len(forecast_codes - material_codes),
            "resultado": (forecast_codes <= material_codes),
        },
        {
            "validacao": ("Fornecedores cobertos"),
            "esperado": 0,
            "encontrado": len(set(purchases["FornecedorKey"]) - set(suppliers["FornecedorKey"])),
            "resultado": (set(purchases["FornecedorKey"]) <= set(suppliers["FornecedorKey"])),
        },
        {
            "validacao": ("UASGs cobertas"),
            "esperado": 0,
            "encontrado": len(set(purchases["UASGKey"]) - set(uasgs["UASGKey"])),
            "resultado": (set(purchases["UASGKey"]) <= set(uasgs["UASGKey"])),
        },
        {
            "validacao": ("Unidades cobertas"),
            "esperado": 0,
            "encontrado": len(set(purchases["UnidadeKey"]) - set(units["UnidadeKey"])),
            "resultado": (set(purchases["UnidadeKey"]) <= set(units["UnidadeKey"])),
        },
        {
            "validacao": ("Cenários cobertos"),
            "esperado": 0,
            "encontrado": len(set(forecast["CenarioKey"]) - set(scenarios["CenarioKey"])),
            "resultado": (set(forecast["CenarioKey"]) <= set(scenarios["CenarioKey"])),
        },
        {
            "validacao": ("Datas das compras cobertas"),
            "esperado": 0,
            "encontrado": len(set(purchases["Data"]) - set(calendar["Data"])),
            "resultado": (set(purchases["Data"]) <= set(calendar["Data"])),
        },
        {
            "validacao": ("Datas da previsão cobertas"),
            "esperado": 0,
            "encontrado": len(set(forecast["Data"]) - set(calendar["Data"])),
            "resultado": (set(forecast["Data"]) <= set(calendar["Data"])),
        },
    ]

    if (production_dir / "FatoCompras.parquet").exists():
        production_purchases = pd.read_parquet(production_dir / "FatoCompras.parquet")

        # A produção pode já conter a versão nova dos medidores. Para validar
        # somente as demais famílias, excluímos tanto os códigos antigos quanto
        # os novos nas duas bases antes da comparação.
        all_meter_codes = old_meter_codes | NEW_METER_CODES

        production_non_meter = production_purchases.loc[
            ~production_purchases["CodigoCATMAT"].isin(all_meter_codes)
        ]
        staging_non_meter = purchases.loc[~purchases["CodigoCATMAT"].isin(all_meter_codes)]

        validations.append(
            {
                "validacao": ("Linhas de outras famílias preservadas"),
                "esperado": len(production_non_meter),
                "encontrado": len(staging_non_meter),
                "resultado": (len(production_non_meter) == len(staging_non_meter)),
            }
        )

    return pd.DataFrame(validations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Integra os novos medidores em staging e recria o modelo Power BI sem publicar."
        )
    )
    parser.add_argument(
        "--history-current",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--forecast-current",
        type=Path,
        default=Path("data/gold/previsao_catmat_v2/previsao_catmat_v2.parquet"),
    )
    parser.add_argument(
        "--summary-current",
        type=Path,
        default=Path("data/gold/previsao_catmat_v2/resumo_previsao_catmat_v2.parquet"),
    )
    parser.add_argument(
        "--pilot-purchases",
        type=Path,
        default=Path("data/pilot_medidores/silver/compras_medidores_piloto.parquet"),
    )
    parser.add_argument(
        "--pilot-forecast",
        type=Path,
        default=Path("data/pilot_medidores/previsao/previsao_catmat_hierarquica.parquet"),
    )
    parser.add_argument(
        "--pilot-summary",
        type=Path,
        default=Path("data/pilot_medidores/previsao/resumo_previsao_catmat.parquet"),
    )
    parser.add_argument(
        "--unit-summary",
        type=Path,
        default=Path("data/pilot_medidores/previsao/resumo_unidade_modelagem.parquet"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/pilot_medidores/modelagem/pesos_rateio_catmat.parquet"),
    )
    parser.add_argument(
        "--powerbi-builder",
        type=Path,
        default=Path("scripts/preparar_modelo_powerbi.py"),
    )
    parser.add_argument(
        "--production-powerbi",
        type=Path,
        default=Path("data/powerbi"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/staging_powerbi/integracao_medidores"),
    )
    args = parser.parse_args()

    required_paths = [
        args.history_current,
        args.forecast_current,
        args.summary_current,
        args.pilot_purchases,
        args.pilot_forecast,
        args.pilot_summary,
        args.unit_summary,
        args.weights,
        args.powerbi_builder,
    ]

    missing = [path.resolve() for path in required_paths if not path.exists()]

    if missing:
        raise SystemExit("Arquivos ausentes:\n" + "\n".join(str(path) for path in missing))

    history_current = pd.read_parquet(args.history_current)
    forecast_current = pd.read_parquet(args.forecast_current)
    summary_current = pd.read_parquet(args.summary_current)

    pilot_purchases = pd.read_parquet(args.pilot_purchases)
    pilot_forecast = pd.read_parquet(args.pilot_forecast)
    pilot_summary = pd.read_parquet(args.pilot_summary)
    unit_summary = pd.read_parquet(args.unit_summary)
    weights = pd.read_parquet(args.weights)

    old_meter_codes = identify_meter_codes(
        history_current,
        forecast_current,
        summary_current,
    )
    family_label = resolve_family_label(
        history_current,
        forecast_current,
        old_meter_codes,
    )

    print(
        "Família preservada:",
        family_label,
    )
    print(
        "CATMATs antigos identificados:",
        sorted(old_meter_codes),
    )

    history_codes = pd.to_numeric(
        history_current["codigo_catmat"],
        errors="coerce",
    )
    forecast_codes = pd.to_numeric(
        forecast_current["codigo_catmat"],
        errors="coerce",
    )
    summary_codes = pd.to_numeric(
        summary_current["codigo_catmat"],
        errors="coerce",
    )

    history_base = history_current.loc[~history_codes.isin(old_meter_codes)].copy()
    forecast_base = forecast_current.loc[~forecast_codes.isin(old_meter_codes)].copy()
    summary_base = summary_current.loc[~summary_codes.isin(old_meter_codes)].copy()

    descriptions = build_description_lookup(pilot_purchases)

    new_history = build_pilot_history(
        pilot_purchases,
        history_current,
        family_label,
    )
    new_forecast = build_pilot_forecast(
        pilot_forecast,
        forecast_current,
        family_label,
        descriptions,
    )
    new_summary = build_pilot_summary(
        pilot_summary,
        unit_summary,
        weights,
        summary_current,
        family_label,
        descriptions,
    )

    combined_history = pd.concat(
        [
            history_base,
            new_history,
        ],
        ignore_index=True,
    )
    combined_forecast = pd.concat(
        [
            forecast_base,
            new_forecast,
        ],
        ignore_index=True,
    )
    combined_summary = pd.concat(
        [
            summary_base,
            new_summary,
        ],
        ignore_index=True,
    )

    combined_history = align_to_schema(
        combined_history,
        history_current,
    )
    combined_forecast = align_to_schema(
        combined_forecast,
        forecast_current,
    )
    combined_summary = align_to_schema(
        combined_summary,
        summary_current,
    )

    input_dir = args.output_root / "inputs_gold"
    powerbi_dir = args.output_root / "powerbi"
    report_dir = args.output_root / "auditoria"

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = input_dir / "fact_compras_integrada.parquet"
    forecast_path = input_dir / "previsao_catmat_integrada.parquet"
    summary_path = input_dir / "resumo_previsao_catmat_integrado.parquet"

    combined_history.to_parquet(
        history_path,
        index=False,
    )
    combined_forecast.to_parquet(
        forecast_path,
        index=False,
    )
    combined_summary.to_parquet(
        summary_path,
        index=False,
    )

    save_psv(
        new_history,
        report_dir / "novas_compras_medidores.psv",
    )
    save_psv(
        new_forecast,
        report_dir / "nova_previsao_medidores.psv",
    )
    save_psv(
        new_summary,
        report_dir / "novo_resumo_medidores.psv",
    )

    if powerbi_dir.exists():
        for path in powerbi_dir.glob("*"):
            if path.is_file():
                path.unlink()

    powerbi_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(args.powerbi_builder),
        "--history",
        str(history_path),
        "--forecast",
        str(forecast_path),
        "--forecast-summary",
        str(summary_path),
        "--output-dir",
        str(powerbi_dir),
    ]

    print()
    print("Recriando modelo em staging...")
    subprocess.run(
        command,
        check=True,
    )

    schema_comparison = compare_schemas(
        args.production_powerbi,
        powerbi_dir,
    )
    validations = validate_staging(
        args.production_powerbi,
        powerbi_dir,
        old_meter_codes,
    )

    save_psv(
        schema_comparison,
        report_dir / "comparacao_esquemas.psv",
    )
    save_psv(
        validations,
        report_dir / "validacoes_integracao.psv",
    )

    manifest = {
        "familia_material": family_label,
        "catmats_antigos_removidos": sorted(old_meter_codes),
        "catmats_novos": sorted(NEW_METER_CODES),
        "linhas_novas_historico": len(new_history),
        "linhas_novas_previsao_gold": len(new_forecast),
        "linhas_novas_resumo": len(new_summary),
        "staging_powerbi": str(powerbi_dir.resolve()),
        "aprovado": bool(
            validations["resultado"].all()
            and schema_comparison["ordem_colunas_igual"].fillna(True).all()
        ),
    }

    manifest_path = args.output_root / "manifesto_integracao_medidores.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Integração em staging concluída.")
    print(
        "Novas linhas de compras:",
        len(new_history),
    )
    print(
        "Novas linhas de previsão Gold:",
        len(new_forecast),
    )
    print(
        "CATMATs no novo resumo:",
        len(new_summary),
    )
    print(
        "Validações aprovadas:",
        int(validations["resultado"].sum()),
        "/",
        len(validations),
    )
    print(
        "Staging aprovado:",
        manifest["aprovado"],
    )
    print(
        "Diretório:",
        args.output_root.resolve(),
    )

    if not manifest["aprovado"]:
        raise SystemExit(
            "\nStaging gerado, mas existem validações pendentes. Nenhum arquivo foi publicado."
        )


if __name__ == "__main__":
    main()
