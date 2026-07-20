from __future__ import annotations

import argparse
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORECAST_HORIZON = 12
MIN_TRAINING_MONTHS = 24


def normalize_header(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def save_psv(dataframe: pd.DataFrame, path: Path) -> None:
    output = dataframe.copy()

    for column in output.select_dtypes(include=["object", "string"]).columns:
        output[column] = (
            output[column]
            .astype("string")
            .str.replace(r"[\r\n\t]+", " ", regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.strip()
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def safe_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())

    if denominator <= 0:
        return np.nan

    return float(np.abs(actual - predicted).sum() / denominator)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def normalized_profile(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(values, dtype=float),
        0,
        None,
    )

    total = float(clipped.sum())

    if total <= 0:
        return np.repeat(
            1.0 / len(clipped),
            len(clipped),
        )

    return clipped / total


def annual_level_recent_mean(
    history: np.ndarray,
    years: int,
) -> float:
    if history.size == 0:
        return 0.0

    annual_totals: list[float] = []

    complete_years = history.size // 12
    if complete_years <= 0:
        return float(history.mean() * 12)

    usable = history[-complete_years * 12 :]
    matrix = usable.reshape(
        complete_years,
        12,
    )
    annual_totals = [float(value) for value in matrix.sum(axis=1)]

    selected = annual_totals[-years:]
    return float(np.mean(selected))


def model_uniform_12m(history: np.ndarray) -> np.ndarray:
    annual_total = annual_level_recent_mean(
        history,
        years=2,
    )
    return np.repeat(
        annual_total / 12,
        FORECAST_HORIZON,
    )


def model_uniform_36m(history: np.ndarray) -> np.ndarray:
    window = history[-36:]
    monthly_level = float(window.mean()) if window.size else 0.0
    return np.repeat(
        monthly_level,
        FORECAST_HORIZON,
    )


def model_seasonal_naive(history: np.ndarray) -> np.ndarray:
    if history.size < 12:
        return model_uniform_36m(history)

    return np.clip(
        history[-12:].astype(float),
        0,
        None,
    )


def model_profile_full(history: np.ndarray) -> np.ndarray:
    if history.size < 24:
        return model_uniform_36m(history)

    complete_years = history.size // 12
    usable = history[-complete_years * 12 :]
    matrix = usable.reshape(
        complete_years,
        12,
    )

    annual_total = annual_level_recent_mean(
        history,
        years=min(2, complete_years),
    )

    shares_by_year = np.vstack([normalized_profile(row) for row in matrix if row.sum() > 0])

    if shares_by_year.size == 0:
        profile = np.repeat(
            1.0 / 12,
            12,
        )
    else:
        profile = normalized_profile(shares_by_year.mean(axis=0) + 0.01)

    return annual_total * profile


def model_profile_recent_36m(history: np.ndarray) -> np.ndarray:
    if history.size < 24:
        return model_uniform_36m(history)

    recent = history[-36:]
    month_positions = np.arange(recent.size) % 12

    monthly_totals = np.zeros(12)
    monthly_counts = np.zeros(12)

    for position, value in zip(
        month_positions,
        recent,
        strict=False,
    ):
        monthly_totals[position] += max(
            float(value),
            0.0,
        )
        monthly_counts[position] += 1

    monthly_average = np.divide(
        monthly_totals,
        monthly_counts,
        out=np.zeros_like(
            monthly_totals,
            dtype=float,
        ),
        where=monthly_counts > 0,
    )

    profile = normalized_profile(monthly_average + 0.01)
    annual_total = annual_level_recent_mean(
        history,
        years=2,
    )

    return annual_total * profile


def croston_sba_level(
    history: np.ndarray,
    alpha: float = 0.20,
) -> float:
    values = np.asarray(
        history,
        dtype=float,
    )
    positive_indices = np.flatnonzero(values > 0)

    if positive_indices.size == 0:
        return 0.0

    first_index = int(positive_indices[0])
    demand_estimate = float(values[first_index])
    interval_estimate = float(first_index + 1)
    last_positive_index = first_index

    for index in positive_indices[1:]:
        demand = float(values[index])
        interval = float(index - last_positive_index)

        demand_estimate = alpha * demand + (1 - alpha) * demand_estimate
        interval_estimate = alpha * interval + (1 - alpha) * interval_estimate
        last_positive_index = int(index)

    if interval_estimate <= 0:
        return 0.0

    return max(
        (1 - alpha / 2) * demand_estimate / interval_estimate,
        0.0,
    )


def model_croston_sba(history: np.ndarray) -> np.ndarray:
    level = croston_sba_level(history)
    return np.repeat(
        level,
        FORECAST_HORIZON,
    )


MODEL_FUNCTIONS: dict[
    str,
    Callable[[np.ndarray], np.ndarray],
] = {
    "MEDIA_2_ANOS_UNIFORME": model_uniform_12m,
    "MEDIA_36_MESES_UNIFORME": model_uniform_36m,
    "SAZONAL_NAIVE_12": model_seasonal_naive,
    "PERFIL_SAZONAL_HISTORICO": model_profile_full,
    "PERFIL_SAZONAL_36_MESES": model_profile_recent_36m,
    "CROSTON_SBA": model_croston_sba,
}


@dataclass
class ValidationResult:
    model_name: str
    windows: int
    wape: float
    mae: float
    annual_bias: float
    zero_forecast_penalty: int
    score: float


def validation_cutoffs(
    series_length: int,
) -> list[int]:
    latest_cutoff = series_length - FORECAST_HORIZON

    if latest_cutoff < MIN_TRAINING_MONTHS:
        return []

    cutoffs = list(
        range(
            MIN_TRAINING_MONTHS,
            latest_cutoff + 1,
            12,
        )
    )

    if latest_cutoff not in cutoffs:
        cutoffs.append(latest_cutoff)

    return sorted(set(cutoffs))


def validate_model(
    series: np.ndarray,
    model_name: str,
    model_function: Callable[
        [np.ndarray],
        np.ndarray,
    ],
) -> ValidationResult:
    cutoffs = validation_cutoffs(len(series))

    actual_all: list[float] = []
    predicted_all: list[float] = []
    annual_biases: list[float] = []
    zero_penalty = 0

    for cutoff in cutoffs:
        train = series[:cutoff]
        actual = series[cutoff : cutoff + FORECAST_HORIZON]

        predicted = np.asarray(
            model_function(train),
            dtype=float,
        )
        predicted = np.clip(
            predicted,
            0,
            None,
        )

        actual_all.extend(actual.tolist())
        predicted_all.extend(predicted.tolist())

        actual_total = float(actual.sum())
        predicted_total = float(predicted.sum())

        if actual_total > 0 and predicted_total <= 0:
            zero_penalty += 1

        if actual_total > 0:
            annual_biases.append(abs(predicted_total - actual_total) / actual_total)

    if not cutoffs:
        return ValidationResult(
            model_name=model_name,
            windows=0,
            wape=np.nan,
            mae=np.nan,
            annual_bias=np.nan,
            zero_forecast_penalty=0,
            score=np.inf,
        )

    actual_array = np.asarray(
        actual_all,
        dtype=float,
    )
    predicted_array = np.asarray(
        predicted_all,
        dtype=float,
    )

    wape_value = safe_wape(
        actual_array,
        predicted_array,
    )
    mae_value = mae(
        actual_array,
        predicted_array,
    )
    annual_bias = float(np.mean(annual_biases)) if annual_biases else np.nan

    normalized_mae = mae_value / max(
        float(np.mean(np.abs(actual_array))),
        1.0,
    )

    score = (
        (wape_value if not np.isnan(wape_value) else normalized_mae)
        + 0.25 * (annual_bias if not np.isnan(annual_bias) else normalized_mae)
        + 0.50 * zero_penalty
    )

    return ValidationResult(
        model_name=model_name,
        windows=len(cutoffs),
        wape=wape_value,
        mae=mae_value,
        annual_bias=annual_bias,
        zero_forecast_penalty=zero_penalty,
        score=float(score),
    )


def choose_model(
    series: np.ndarray,
) -> tuple[
    ValidationResult,
    pd.DataFrame,
]:
    results = [
        validate_model(
            series,
            model_name,
            model_function,
        )
        for model_name, model_function in MODEL_FUNCTIONS.items()
    ]

    validation = pd.DataFrame(
        [
            {
                "modelo": result.model_name,
                "janelas_validacao": result.windows,
                "wape_validacao": result.wape,
                "mae_validacao": result.mae,
                "vies_anual_medio": result.annual_bias,
                "penalidade_previsao_zero": (result.zero_forecast_penalty),
                "score_selecao": result.score,
            }
            for result in results
        ]
    ).sort_values(
        [
            "score_selecao",
            "modelo",
        ]
    )

    finite_results = [result for result in results if math.isfinite(result.score)]

    if not finite_results:
        fallback = ValidationResult(
            model_name="CROSTON_SBA",
            windows=0,
            wape=np.nan,
            mae=np.nan,
            annual_bias=np.nan,
            zero_forecast_penalty=0,
            score=np.inf,
        )
        return fallback, validation

    best_score = min(result.score for result in finite_results)

    tolerance = best_score * 1.05

    simplicity_order = [
        "CROSTON_SBA",
        "MEDIA_2_ANOS_UNIFORME",
        "MEDIA_36_MESES_UNIFORME",
        "SAZONAL_NAIVE_12",
        "PERFIL_SAZONAL_HISTORICO",
        "PERFIL_SAZONAL_36_MESES",
    ]

    candidates = {
        result.model_name: result for result in finite_results if result.score <= tolerance
    }

    for model_name in simplicity_order:
        if model_name in candidates:
            return candidates[model_name], validation

    best = min(
        finite_results,
        key=lambda result: result.score,
    )
    return best, validation


def confidence_level(
    active_months: int,
    active_years: int,
    validation_windows: int,
    wape_value: float,
    level: str,
) -> str:
    if (
        active_months >= 24
        and active_years >= 4
        and validation_windows >= 3
        and not np.isnan(wape_value)
        and wape_value <= 0.75
    ):
        base = "MEDIA"
    else:
        base = "BAIXA"

    if level.upper() == "SEGMENTO" and base == "MEDIA":
        return "BAIXA"

    return base


def uncertainty_factor(
    confidence: str,
    wape_value: float,
    active_months: int,
) -> float:
    base = wape_value if not np.isnan(wape_value) else 0.75

    sparsity_penalty = max(
        0.0,
        1.0 - active_months / 36,
    )

    factor = base + 0.35 * sparsity_penalty

    minimum = 0.35 if confidence == "MEDIA" else 0.60

    return float(
        np.clip(
            max(factor, minimum),
            0.25,
            1.25,
        )
    )


def forecast_units(
    modeling_series: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    forecast_months = pd.date_range(
        forecast_start,
        periods=FORECAST_HORIZON,
        freq="MS",
    )

    forecast_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []

    for unit, group in modeling_series.groupby(
        "unidade_modelagem",
        dropna=False,
    ):
        ordered = group.sort_values("inicio_mes")
        series = (
            pd.to_numeric(
                ordered["quantidade"],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
        )

        selected, validation = choose_model(series)
        validation.insert(
            0,
            "unidade_modelagem",
            unit,
        )
        validation_rows.append(validation)

        base_forecast = np.asarray(
            MODEL_FUNCTIONS[selected.model_name](series),
            dtype=float,
        )
        base_forecast = np.clip(
            base_forecast,
            0,
            None,
        )

        if series.sum() > 0 and base_forecast.sum() <= 0:
            base_forecast = model_croston_sba(series)

        active_months = int((series > 0).sum())
        active_years = int(
            ordered.loc[
                ordered["quantidade"] > 0,
                "inicio_mes",
            ].dt.year.nunique()
        )
        level = str(ordered["nivel_modelagem"].iloc[0])

        confidence = confidence_level(
            active_months=active_months,
            active_years=active_years,
            validation_windows=selected.windows,
            wape_value=selected.wape,
            level=level,
        )

        uncertainty = uncertainty_factor(
            confidence=confidence,
            wape_value=selected.wape,
            active_months=active_months,
        )

        lower_forecast = base_forecast * max(
            0.0,
            1.0 - uncertainty,
        )
        upper_forecast = base_forecast * (1.0 + uncertainty)

        for month, lower, base, upper in zip(
            forecast_months,
            lower_forecast,
            base_forecast,
            upper_forecast,
            strict=False,
        ):
            forecast_rows.append(
                {
                    "unidade_modelagem": unit,
                    "nivel_modelagem": level,
                    "segmento_modelagem": (ordered["segmento_modelagem"].iloc[0]),
                    "data": month,
                    "quantidade_cenario_inferior": float(lower),
                    "quantidade_prevista_base": float(base),
                    "quantidade_cenario_superior": float(upper),
                    "modelo_selecionado": selected.model_name,
                    "confianca_previsao": confidence,
                    "fator_incerteza": uncertainty,
                }
            )

        summary_rows.append(
            {
                "unidade_modelagem": unit,
                "nivel_modelagem": level,
                "segmento_modelagem": (ordered["segmento_modelagem"].iloc[0]),
                "modelo_selecionado": selected.model_name,
                "confianca_previsao": confidence,
                "janelas_validacao": selected.windows,
                "wape_validacao": selected.wape,
                "mae_validacao": selected.mae,
                "vies_anual_medio": selected.annual_bias,
                "score_selecao": selected.score,
                "meses_historicos": len(series),
                "meses_com_demanda": active_months,
                "anos_ativos": active_years,
                "percentual_meses_sem_demanda": float((series == 0).mean()),
                "quantidade_historica_total": float(series.sum()),
                "previsao_12m_inferior": float(lower_forecast.sum()),
                "previsao_12m_base": float(base_forecast.sum()),
                "previsao_12m_superior": float(upper_forecast.sum()),
                "valores_mensais_distintos_base": int(pd.Series(base_forecast).round(8).nunique()),
                "fator_incerteza": uncertainty,
            }
        )

    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(summary_rows),
        pd.concat(
            validation_rows,
            ignore_index=True,
        ),
    )


def allocate_to_catmat(
    unit_forecast: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    allocated = unit_forecast.merge(
        weights[
            [
                "unidade_modelagem",
                "codigo_catmat",
                "peso_rateio",
                "metodo_rateio",
            ]
        ],
        on="unidade_modelagem",
        how="left",
        validate="many_to_many",
    )

    if allocated["peso_rateio"].isna().any():
        missing = (
            allocated.loc[
                allocated["peso_rateio"].isna(),
                "unidade_modelagem",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Unidades sem pesos de rateio: " + ", ".join(str(value) for value in missing)
        )

    for source, target in [
        (
            "quantidade_cenario_inferior",
            "quantidade_cenario_inferior",
        ),
        (
            "quantidade_prevista_base",
            "quantidade_prevista_base",
        ),
        (
            "quantidade_cenario_superior",
            "quantidade_cenario_superior",
        ),
    ]:
        allocated[target] = allocated[source] * allocated["peso_rateio"]

    columns = [
        "codigo_catmat",
        "data",
        "unidade_modelagem",
        "nivel_modelagem",
        "segmento_modelagem",
        "modelo_selecionado",
        "confianca_previsao",
        "metodo_rateio",
        "peso_rateio",
        "quantidade_cenario_inferior",
        "quantidade_prevista_base",
        "quantidade_cenario_superior",
        "fator_incerteza",
    ]

    allocated = allocated[columns].sort_values(
        [
            "codigo_catmat",
            "data",
        ]
    )

    validate_allocation(
        unit_forecast,
        allocated,
    )

    return allocated


def validate_allocation(
    unit_forecast: pd.DataFrame,
    allocated: pd.DataFrame,
) -> None:
    unit_totals = unit_forecast.groupby(
        [
            "unidade_modelagem",
            "data",
        ],
        as_index=False,
    )[
        [
            "quantidade_cenario_inferior",
            "quantidade_prevista_base",
            "quantidade_cenario_superior",
        ]
    ].sum()

    allocated_totals = allocated.groupby(
        [
            "unidade_modelagem",
            "data",
        ],
        as_index=False,
    )[
        [
            "quantidade_cenario_inferior",
            "quantidade_prevista_base",
            "quantidade_cenario_superior",
        ]
    ].sum()

    merged = unit_totals.merge(
        allocated_totals,
        on=[
            "unidade_modelagem",
            "data",
        ],
        suffixes=(
            "_unidade",
            "_catmat",
        ),
        validate="one_to_one",
    )

    for column in [
        "quantidade_cenario_inferior",
        "quantidade_prevista_base",
        "quantidade_cenario_superior",
    ]:
        if not np.allclose(
            merged[f"{column}_unidade"],
            merged[f"{column}_catmat"],
            atol=1e-8,
        ):
            raise ValueError(f"Falha de preservação do total no rateio: {column}")


def build_long_forecast(
    catmat_forecast: pd.DataFrame,
) -> pd.DataFrame:
    scenarios = [
        (
            "INFERIOR",
            1,
            "quantidade_cenario_inferior",
        ),
        (
            "BASE",
            2,
            "quantidade_prevista_base",
        ),
        (
            "SUPERIOR",
            3,
            "quantidade_cenario_superior",
        ),
    ]

    rows: list[pd.DataFrame] = []

    for scenario, order, column in scenarios:
        part = catmat_forecast[
            [
                "codigo_catmat",
                "data",
                "unidade_modelagem",
                "nivel_modelagem",
                "segmento_modelagem",
                "modelo_selecionado",
                "confianca_previsao",
                "metodo_rateio",
                "peso_rateio",
            ]
        ].copy()

        part["cenario"] = scenario
        part["ordem_cenario"] = order
        part["quantidade_prevista"] = catmat_forecast[column]

        rows.append(part)

    return pd.concat(
        rows,
        ignore_index=True,
    ).sort_values(
        [
            "codigo_catmat",
            "data",
            "ordem_cenario",
        ]
    )


def build_catmat_summary(
    catmat_forecast: pd.DataFrame,
) -> pd.DataFrame:
    return (
        catmat_forecast.groupby(
            [
                "codigo_catmat",
                "unidade_modelagem",
                "nivel_modelagem",
                "segmento_modelagem",
                "modelo_selecionado",
                "confianca_previsao",
                "metodo_rateio",
                "peso_rateio",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            previsao_12m_inferior=(
                "quantidade_cenario_inferior",
                "sum",
            ),
            previsao_12m_base=(
                "quantidade_prevista_base",
                "sum",
            ),
            previsao_12m_superior=(
                "quantidade_cenario_superior",
                "sum",
            ),
            media_mensal_prevista_base=(
                "quantidade_prevista_base",
                "mean",
            ),
            valores_mensais_distintos_base=(
                "quantidade_prevista_base",
                lambda values: int(pd.Series(values).round(8).nunique()),
            ),
        )
        .sort_values(
            [
                "unidade_modelagem",
                "codigo_catmat",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera previsão hierárquica piloto para medidores, "
            "selecionando modelos por backtesting e rateando "
            "segmentos aos CATMATs."
        )
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=Path("data/pilot_medidores/modelagem/serie_unidade_modelagem.parquet"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/pilot_medidores/modelagem/pesos_rateio_catmat.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pilot_medidores/previsao"),
    )
    parser.add_argument(
        "--forecast-start",
        default="2026-01-01",
    )
    args = parser.parse_args()

    for path in [
        args.series,
        args.weights,
    ]:
        if not path.exists():
            raise SystemExit(f"Arquivo não encontrado: {path.resolve()}")

    modeling_series = pd.read_parquet(args.series)
    modeling_series.columns = [normalize_header(column) for column in modeling_series.columns]

    weights = pd.read_parquet(args.weights)
    weights.columns = [normalize_header(column) for column in weights.columns]

    modeling_series["inicio_mes"] = pd.to_datetime(
        modeling_series["inicio_mes"],
        errors="coerce",
    )

    forecast_start = pd.Timestamp(args.forecast_start).to_period("M").to_timestamp()

    (
        unit_forecast,
        unit_summary,
        validation,
    ) = forecast_units(
        modeling_series,
        forecast_start,
    )

    catmat_forecast = allocate_to_catmat(
        unit_forecast,
        weights,
    )
    catmat_long = build_long_forecast(catmat_forecast)
    catmat_summary = build_catmat_summary(catmat_forecast)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "previsao_unidade_modelagem": (unit_forecast),
        "resumo_unidade_modelagem": (unit_summary),
        "validacao_modelos": validation,
        "previsao_catmat_hierarquica": (catmat_forecast),
        "previsao_catmat_long": (catmat_long),
        "resumo_previsao_catmat": (catmat_summary),
    }

    for name, dataframe in outputs.items():
        dataframe.to_parquet(
            args.output_dir / f"{name}.parquet",
            index=False,
        )
        save_psv(
            dataframe,
            args.output_dir / f"{name}.psv",
        )

    print("Previsão hierárquica concluída.")
    print(
        "Unidades modeladas:",
        unit_summary["unidade_modelagem"].nunique(),
    )
    print(
        "CATMATs previstos:",
        catmat_summary["codigo_catmat"].nunique(),
    )
    print(
        "Meses previstos:",
        catmat_forecast["data"].nunique(),
    )
    print()
    print("Resumo por unidade:")
    print(
        unit_summary[
            [
                "unidade_modelagem",
                "modelo_selecionado",
                "confianca_previsao",
                "meses_com_demanda",
                "wape_validacao",
                "previsao_12m_base",
                "valores_mensais_distintos_base",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Diretório:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
