from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

save_psv = import_module("src.reporting").save_psv


REQUIRED_COLUMNS = {
    "data_publicacao",
    "codigo_catmat",
    "familia_material",
    "quantidade",
}


@dataclass(frozen=True)
class ModelEvaluation:
    model: str
    mae: float
    rmse: float
    wape_percentual: float
    bias: float
    absolute_error_quantile_80: float
    absolute_error_quantile_90: float
    validation_points: int
    folds: int


def validate_schema(dataframe: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError("Colunas obrigatórias ausentes na fato: " + ", ".join(missing))


def prepare_fact(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    result["data_publicacao"] = pd.to_datetime(
        result["data_publicacao"],
        errors="coerce",
        utc=True,
    )
    result["codigo_catmat"] = pd.to_numeric(
        result["codigo_catmat"],
        errors="coerce",
    ).astype("Int64")
    result["quantidade"] = pd.to_numeric(
        result["quantidade"],
        errors="coerce",
    )

    valid = (
        result["data_publicacao"].notna()
        & result["codigo_catmat"].notna()
        & result["familia_material"].notna()
        & result["quantidade"].notna()
        & (result["quantidade"] >= 0)
    )

    if "dq_duplicado" in result.columns:
        valid &= ~result["dq_duplicado"].fillna(False).astype(bool)

    result = result.loc[valid].copy()
    result["mes"] = (
        result["data_publicacao"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    )

    description_candidates = [
        "descricao_material",
        "descricao_item",
        "descricao",
    ]
    description_column = next(
        (column for column in description_candidates if column in result.columns),
        None,
    )

    if description_column is None:
        result["descricao_catmat"] = pd.NA
    else:
        result["descricao_catmat"] = (
            result[description_column]
            .astype("string")
            .str.replace(r"[\r\n\t]+", " ", regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.strip()
        )

    return result


def build_monthly_series(dataframe: pd.DataFrame) -> pd.DataFrame:
    grouped = dataframe.groupby(
        [
            "codigo_catmat",
            "familia_material",
            "descricao_catmat",
            "mes",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        quantidade_total=("quantidade", "sum"),
        numero_observacoes=("quantidade", "size"),
    )

    if grouped.empty:
        return pd.DataFrame()

    global_start = grouped["mes"].min()
    global_end = grouped["mes"].max()
    complete_index = pd.date_range(
        start=global_start,
        end=global_end,
        freq="MS",
    )

    rows: list[pd.DataFrame] = []

    group_columns = [
        "codigo_catmat",
        "familia_material",
        "descricao_catmat",
    ]

    for keys, group in grouped.groupby(
        group_columns,
        dropna=False,
    ):
        codigo_catmat, familia_material, descricao_catmat = keys

        complete = group.set_index("mes").reindex(complete_index).rename_axis("mes").reset_index()

        complete["codigo_catmat"] = codigo_catmat
        complete["familia_material"] = familia_material
        complete["descricao_catmat"] = descricao_catmat
        complete["quantidade_total"] = complete["quantidade_total"].fillna(0.0).astype(float)
        complete["numero_observacoes"] = complete["numero_observacoes"].fillna(0).astype(int)

        rows.append(complete)

    return pd.concat(rows, ignore_index=True)


def forecast_naive(
    train: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if len(train) == 0:
        return np.zeros(horizon)
    return np.repeat(float(train[-1]), horizon)


def forecast_mean(
    train: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if len(train) == 0:
        return np.zeros(horizon)
    return np.repeat(float(np.mean(train)), horizon)


def forecast_moving_average(
    train: np.ndarray,
    horizon: int,
    window: int,
) -> np.ndarray:
    if len(train) == 0:
        return np.zeros(horizon)

    effective_window = min(window, len(train))
    level = float(np.mean(train[-effective_window:]))
    return np.repeat(level, horizon)


def forecast_seasonal_naive(
    train: np.ndarray,
    horizon: int,
    season_length: int = 12,
) -> np.ndarray:
    if len(train) < season_length:
        return forecast_mean(train, horizon)

    seasonal = train[-season_length:]
    return np.resize(seasonal, horizon).astype(float)


def forecast_linear_trend(
    train: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if len(train) < 6:
        return forecast_mean(train, horizon)

    x = np.arange(len(train), dtype=float)
    slope, intercept = np.polyfit(x, train.astype(float), 1)
    future_x = np.arange(
        len(train),
        len(train) + horizon,
        dtype=float,
    )

    forecast = intercept + slope * future_x
    return np.maximum(forecast, 0.0)


def croston_sba_level(
    train: np.ndarray,
    alpha: float = 0.1,
) -> float:
    values = np.asarray(train, dtype=float)
    nonzero_positions = np.flatnonzero(values > 0)

    if len(nonzero_positions) == 0:
        return 0.0

    first_position = int(nonzero_positions[0])
    demand_estimate = float(values[first_position])
    interval_estimate = float(first_position + 1)
    last_position = first_position

    for position_value in nonzero_positions[1:]:
        position = int(position_value)
        interval = float(position - last_position)

        demand_estimate += alpha * (float(values[position]) - demand_estimate)
        interval_estimate += alpha * (interval - interval_estimate)
        last_position = position

    if interval_estimate <= 0:
        return 0.0

    level = (1.0 - alpha / 2.0) * demand_estimate / interval_estimate
    return max(level, 0.0)


def forecast_croston_sba(
    train: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return np.repeat(
        croston_sba_level(train),
        horizon,
    )


def forecast_positive_mean_rate(
    train: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = np.asarray(train, dtype=float)
    positive = values[values > 0]

    if len(positive) == 0 or len(values) == 0:
        return np.zeros(horizon)

    occurrence_rate = len(positive) / len(values)
    level = float(np.mean(positive) * occurrence_rate)
    return np.repeat(max(level, 0.0), horizon)


MODEL_FUNCTIONS: dict[
    str,
    Callable[[np.ndarray, int], np.ndarray],
] = {
    "INGENUO_ULTIMO_MES": forecast_naive,
    "MEDIA_HISTORICA": forecast_mean,
    "MEDIA_MOVEL_3M": lambda train, horizon: forecast_moving_average(train, horizon, 3),
    "MEDIA_MOVEL_6M": lambda train, horizon: forecast_moving_average(train, horizon, 6),
    "SAZONAL_12M": forecast_seasonal_naive,
    "TENDENCIA_LINEAR": forecast_linear_trend,
    "CROSTON_SBA": forecast_croston_sba,
    "MEDIA_POSITIVA_X_FREQUENCIA": forecast_positive_mean_rate,
}


INTERMITTENT_MODELS = {
    "MEDIA_HISTORICA",
    "MEDIA_MOVEL_6M",
    "CROSTON_SBA",
    "MEDIA_POSITIVA_X_FREQUENCIA",
}


def intermittency_metrics(
    values: np.ndarray,
) -> dict[str, float | str | int]:
    values = np.asarray(values, dtype=float)
    active_values = values[values > 0]
    active_positions = np.flatnonzero(values > 0)

    total_months = len(values)
    active_months = len(active_values)
    zero_months = total_months - active_months

    if active_months == 0:
        adi = math.inf
        cv2 = math.nan
        demand_type = "SEM_DEMANDA"
    else:
        adi = (
            float(np.mean(np.diff(active_positions))) if active_months > 1 else float(total_months)
        )

        mean_active = float(np.mean(active_values))
        cv2 = (
            float((np.std(active_values, ddof=1) / mean_active) ** 2)
            if active_months > 1 and mean_active > 0
            else 0.0
        )

        if adi < 1.32 and cv2 < 0.49:
            demand_type = "REGULAR"
        elif adi < 1.32 and cv2 >= 0.49:
            demand_type = "ERRATICA"
        elif adi >= 1.32 and cv2 < 0.49:
            demand_type = "INTERMITENTE"
        else:
            demand_type = "IRREGULAR_INTERMITENTE"

    return {
        "meses_total": total_months,
        "meses_com_demanda": active_months,
        "meses_sem_demanda": zero_months,
        "percentual_meses_sem_demanda": (
            round(zero_months / total_months * 100, 2) if total_months else 0.0
        ),
        "adi": adi,
        "cv2_demanda_positiva": cv2,
        "tipo_demanda": demand_type,
    }


def rolling_windows(
    n_months: int,
) -> list[tuple[int, int]]:
    if n_months >= 36:
        horizon = 3
        folds = 4
    elif n_months >= 24:
        horizon = 3
        folds = 3
    elif n_months >= 15:
        horizon = 3
        folds = 2
    else:
        return []

    windows: list[tuple[int, int]] = []

    for fold_index in range(folds, 0, -1):
        validation_end = n_months - horizon * (fold_index - 1)
        validation_start = validation_end - horizon

        if validation_start < 12:
            continue

        windows.append(
            (
                validation_start,
                validation_end,
            )
        )

    return windows


def evaluate_model(
    values: np.ndarray,
    model_name: str,
) -> ModelEvaluation | None:
    windows = rolling_windows(len(values))

    if not windows:
        return None

    model_function = MODEL_FUNCTIONS[model_name]

    all_actual: list[float] = []
    all_forecast: list[float] = []
    absolute_errors: list[float] = []

    for validation_start, validation_end in windows:
        train = values[:validation_start]
        actual = values[validation_start:validation_end]

        forecast = model_function(
            train,
            len(actual),
        )
        forecast = np.maximum(
            np.asarray(forecast, dtype=float),
            0.0,
        )

        all_actual.extend(actual.tolist())
        all_forecast.extend(forecast.tolist())
        absolute_errors.extend(np.abs(actual - forecast).tolist())

    actual_array = np.asarray(all_actual, dtype=float)
    forecast_array = np.asarray(
        all_forecast,
        dtype=float,
    )
    error = actual_array - forecast_array
    absolute_error = np.abs(error)

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum(np.abs(actual_array)))
    wape = float(np.sum(absolute_error) / denominator * 100) if denominator > 0 else math.nan
    bias = float(np.mean(forecast_array - actual_array))

    return ModelEvaluation(
        model=model_name,
        mae=mae,
        rmse=rmse,
        wape_percentual=wape,
        bias=bias,
        absolute_error_quantile_80=float(np.quantile(absolute_errors, 0.80)),
        absolute_error_quantile_90=float(np.quantile(absolute_errors, 0.90)),
        validation_points=len(actual_array),
        folds=len(windows),
    )


def select_models(
    demand_type: str,
) -> list[str]:
    if demand_type in {
        "INTERMITENTE",
        "IRREGULAR_INTERMITENTE",
        "SEM_DEMANDA",
    }:
        return sorted(INTERMITTENT_MODELS)

    return sorted(MODEL_FUNCTIONS)


def evaluate_series(
    values: np.ndarray,
    demand_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name in select_models(demand_type):
        evaluation = evaluate_model(
            values,
            model_name,
        )

        if evaluation is None:
            continue

        rows.append(
            {
                "modelo": evaluation.model,
                "mae": evaluation.mae,
                "rmse": evaluation.rmse,
                "wape_percentual": (evaluation.wape_percentual),
                "vies_medio": evaluation.bias,
                "erro_absoluto_q80": (evaluation.absolute_error_quantile_80),
                "erro_absoluto_q90": (evaluation.absolute_error_quantile_90),
                "pontos_validacao": (evaluation.validation_points),
                "janelas_validacao": evaluation.folds,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["wape_ordenacao"] = result["wape_percentual"].fillna(np.inf)

    return (
        result.sort_values(
            [
                "wape_ordenacao",
                "mae",
                "rmse",
                "modelo",
            ]
        )
        .drop(columns=["wape_ordenacao"])
        .reset_index(drop=True)
    )


def fallback_nonzero_model(
    performance: pd.DataFrame,
) -> str:
    preferred = performance.loc[
        performance["modelo"].isin(
            [
                "CROSTON_SBA",
                "MEDIA_POSITIVA_X_FREQUENCIA",
                "MEDIA_HISTORICA",
            ]
        )
    ]

    if not preferred.empty:
        return str(preferred.iloc[0]["modelo"])

    return str(performance.iloc[0]["modelo"])


def confidence_classification(
    diagnostics: dict[str, float | str | int],
    selected_metrics: dict[str, Any],
) -> tuple[str, str]:
    months = int(diagnostics["meses_total"])
    active_months = int(diagnostics["meses_com_demanda"])
    zero_percentage = float(diagnostics["percentual_meses_sem_demanda"])
    wape = float(
        selected_metrics.get(
            "wape_percentual",
            math.nan,
        )
    )
    folds = int(
        selected_metrics.get(
            "janelas_validacao",
            0,
        )
    )

    if (
        months >= 36
        and active_months >= 18
        and zero_percentage <= 50
        and folds >= 4
        and np.isfinite(wape)
        and wape <= 35
    ):
        return (
            "ALTA",
            "Histórico amplo, demanda frequente e erro baixo em validação temporal múltipla.",
        )

    if months >= 24 and active_months >= 8 and folds >= 3 and np.isfinite(wape) and wape <= 80:
        return (
            "MEDIA",
            "Histórico utilizável, com volatilidade ou intermitência ainda relevante.",
        )

    return (
        "BAIXA",
        "Série curta, esparsa ou com erro elevado; resultado apenas indicativo.",
    )


def planning_error_band(
    selected_metrics: dict[str, Any],
    confidence: str,
) -> float:
    if confidence == "BAIXA":
        return float(
            selected_metrics.get(
                "erro_absoluto_q90",
                0.0,
            )
        )

    return float(
        selected_metrics.get(
            "erro_absoluto_q80",
            0.0,
        )
    )


def forecast_all_catmats(
    monthly: pd.DataFrame,
    horizon: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    forecast_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    group_columns = [
        "codigo_catmat",
        "familia_material",
        "descricao_catmat",
    ]

    for keys, group in monthly.groupby(
        group_columns,
        dropna=False,
    ):
        codigo_catmat, familia_material, descricao_catmat = keys
        group = group.sort_values("mes")
        values = group["quantidade_total"].to_numpy(dtype=float)

        diagnostics = intermittency_metrics(values)
        demand_type = str(diagnostics["tipo_demanda"])
        performance = evaluate_series(
            values,
            demand_type,
        )

        if performance.empty:
            selected_model = "CROSTON_SBA"
            selected_metrics: dict[str, Any] = {
                "mae": math.nan,
                "rmse": math.nan,
                "wape_percentual": math.nan,
                "vies_medio": math.nan,
                "erro_absoluto_q80": 0.0,
                "erro_absoluto_q90": 0.0,
                "pontos_validacao": 0,
                "janelas_validacao": 0,
            }
        else:
            selected_model = str(performance.iloc[0]["modelo"])
            selected_metrics = performance.iloc[0].to_dict()

        point_forecast = MODEL_FUNCTIONS[selected_model](
            values,
            horizon,
        )
        point_forecast = np.maximum(
            np.asarray(
                point_forecast,
                dtype=float,
            ),
            0.0,
        )

        if np.sum(values) > 0 and np.sum(point_forecast) == 0 and not performance.empty:
            selected_model = fallback_nonzero_model(performance)
            selected_metrics = (
                performance.loc[performance["modelo"] == selected_model].iloc[0].to_dict()
            )
            point_forecast = MODEL_FUNCTIONS[selected_model](
                values,
                horizon,
            )
            point_forecast = np.maximum(
                np.asarray(
                    point_forecast,
                    dtype=float,
                ),
                0.0,
            )

        confidence, confidence_reason = confidence_classification(
            diagnostics,
            selected_metrics,
        )
        error_band = planning_error_band(
            selected_metrics,
            confidence,
        )

        lower_forecast = np.maximum(
            point_forecast - error_band,
            0.0,
        )
        upper_forecast = point_forecast + error_band

        last_month = group["mes"].max()
        future_months = pd.date_range(
            start=last_month + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )

        for (
            month,
            lower_value,
            point_value,
            upper_value,
        ) in zip(
            future_months,
            lower_forecast,
            point_forecast,
            upper_forecast,
            strict=True,
        ):
            forecast_rows.append(
                {
                    "mes": month,
                    "codigo_catmat": codigo_catmat,
                    "familia_material": familia_material,
                    "descricao_catmat": descricao_catmat,
                    "modelo_selecionado": (selected_model),
                    "confianca_previsao": confidence,
                    "tipo_demanda": demand_type,
                    "quantidade_cenario_inferior": (float(lower_value)),
                    "quantidade_prevista_base": (float(point_value)),
                    "quantidade_cenario_superior": (float(upper_value)),
                    "erro_referencia_mensal": (error_band),
                    "observacao": (
                        "Faixa de planejamento baseada nos "
                        "erros históricos de validação; não é "
                        "intervalo estatístico formal."
                    ),
                }
            )

        if not performance.empty:
            for _, row in performance.iterrows():
                performance_rows.append(
                    {
                        "codigo_catmat": codigo_catmat,
                        "familia_material": (familia_material),
                        "descricao_catmat": (descricao_catmat),
                        **row.to_dict(),
                        "modelo_selecionado": (row["modelo"] == selected_model),
                    }
                )

        next_step = (
            "Agregar por família e incorporar consumo interno, estoque, criticidade e plano de obras."
            if str(familia_material).strip().lower() == "medidores energia"
            else ("Incorporar dados internos de consumo, estoque e planejamento para recalibração.")
        )

        diagnostics_rows.append(
            {
                "codigo_catmat": codigo_catmat,
                "familia_material": familia_material,
                "descricao_catmat": descricao_catmat,
                **diagnostics,
                "mes_inicial": group["mes"].min(),
                "mes_final": group["mes"].max(),
                "modelo_selecionado": (selected_model),
                "confianca_previsao": confidence,
                "motivo_confianca": (confidence_reason),
                "wape_validacao_percentual": (
                    selected_metrics.get(
                        "wape_percentual",
                        math.nan,
                    )
                ),
                "mae_validacao": (
                    selected_metrics.get(
                        "mae",
                        math.nan,
                    )
                ),
                "janelas_validacao": (
                    selected_metrics.get(
                        "janelas_validacao",
                        0,
                    )
                ),
                "proximos_passos": next_step,
            }
        )

        summary_rows.append(
            {
                "codigo_catmat": codigo_catmat,
                "familia_material": familia_material,
                "descricao_catmat": descricao_catmat,
                "modelo_selecionado": (selected_model),
                "confianca_previsao": confidence,
                "tipo_demanda": demand_type,
                "quantidade_historica_total": float(np.sum(values)),
                "previsao_12m_cenario_inferior": float(np.sum(lower_forecast)),
                "previsao_12m_base": float(np.sum(point_forecast)),
                "previsao_12m_cenario_superior": float(np.sum(upper_forecast)),
                "media_mensal_prevista_base": float(np.mean(point_forecast)),
                "wape_validacao_percentual": (
                    selected_metrics.get(
                        "wape_percentual",
                        math.nan,
                    )
                ),
                "mae_validacao": (
                    selected_metrics.get(
                        "mae",
                        math.nan,
                    )
                ),
                "meses_com_demanda": (diagnostics["meses_com_demanda"]),
                "percentual_meses_sem_demanda": (diagnostics["percentual_meses_sem_demanda"]),
                "janelas_validacao": (
                    selected_metrics.get(
                        "janelas_validacao",
                        0,
                    )
                ),
                "proximos_passos": next_step,
            }
        )

    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(performance_rows),
        pd.DataFrame(diagnostics_rows),
        pd.DataFrame(summary_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Previsão robusta por CATMAT com validação "
            "temporal múltipla e cenários de planejamento."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gold/previsao_catmat_v2"),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    if args.horizon <= 0:
        raise SystemExit("O horizonte deve ser maior que zero.")

    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input.resolve()}")

    fact = pd.read_parquet(args.input)
    validate_schema(fact)
    prepared = prepare_fact(fact)
    monthly = build_monthly_series(prepared)

    if monthly.empty:
        raise SystemExit("Nenhuma série mensal foi construída.")

    (
        forecast,
        performance,
        diagnostics,
        summary,
    ) = forecast_all_catmats(
        monthly,
        args.horizon,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_parquet(
        args.output_dir / "serie_mensal_catmat_v2.parquet",
        index=False,
    )
    forecast.to_parquet(
        args.output_dir / "previsao_catmat_v2.parquet",
        index=False,
    )
    summary.to_parquet(
        args.output_dir / "resumo_previsao_catmat_v2.parquet",
        index=False,
    )

    save_psv(
        monthly,
        args.output_dir / "serie_mensal_catmat_v2.psv",
    )
    save_psv(
        forecast,
        args.output_dir / "previsao_catmat_v2.psv",
    )
    save_psv(
        performance,
        args.output_dir / "desempenho_modelos_catmat_v2.psv",
    )
    save_psv(
        diagnostics,
        args.output_dir / "diagnostico_series_catmat_v2.psv",
    )
    save_psv(
        summary,
        args.output_dir / "resumo_previsao_catmat_v2.psv",
    )

    print("Previsão CATMAT V2 concluída.")
    print(
        "CATMATs:",
        summary["codigo_catmat"].nunique(),
    )
    print(
        "Horizonte:",
        args.horizon,
        "meses",
    )
    print(
        "Período previsto:",
        forecast["mes"].min().date(),
        "a",
        forecast["mes"].max().date(),
    )
    print()
    print("Confiança:")
    print(summary["confianca_previsao"].value_counts().to_string())
    print()
    print("Modelos selecionados:")
    print(summary["modelo_selecionado"].value_counts().to_string())
    print()
    print(
        "Cenário inferior total:",
        round(
            summary["previsao_12m_cenario_inferior"].sum(),
            2,
        ),
    )
    print(
        "Cenário base total:",
        round(
            summary["previsao_12m_base"].sum(),
            2,
        ),
    )
    print(
        "Cenário superior total:",
        round(
            summary["previsao_12m_cenario_superior"].sum(),
            2,
        ),
    )
    print()
    print(
        "Diretório:",
        args.output_dir.resolve(),
    )
    print()
    print(
        "Nota: previsão de aquisições públicas observadas; não representa consumo físico nem necessidade interna de uma distribuidora."
    )


if __name__ == "__main__":
    main()
