from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def first_existing(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in dataframe.columns), None)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
    )


def summarize_layer(
    layer: str,
    dataframe: pd.DataFrame,
    state_candidates: list[str],
    uasg_candidates: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    state_column = first_existing(dataframe, state_candidates)
    uasg_column = first_existing(dataframe, uasg_candidates)

    if state_column is None:
        states = pd.Series(pd.NA, index=dataframe.index, dtype="string")
    else:
        states = normalize_text(dataframe[state_column])

    if uasg_column is None:
        uasgs = pd.Series(pd.NA, index=dataframe.index, dtype="string")
    else:
        uasgs = normalize_text(dataframe[uasg_column])

    filled = int(states.notna().sum())
    total = int(len(dataframe))

    summary = {
        "camada": layer,
        "linhas": total,
        "coluna_estado": state_column,
        "estado_preenchido": filled,
        "estado_ausente": total - filled,
        "cobertura_percentual": round((filled / total * 100) if total else 0.0, 2),
        "ufs_distintas": int(states.nunique(dropna=True)),
        "coluna_uasg": uasg_column,
        "uasgs_distintas": int(uasgs.nunique(dropna=True)),
    }

    distribution = (
        pd.DataFrame({"UF": states}).dropna().value_counts("UF").rename("registros").reset_index()
    )
    distribution.insert(0, "camada", layer)

    pairs = pd.DataFrame({"UASG": uasgs, "UF": states}).dropna()
    if pairs.empty:
        conflicts = pd.DataFrame(columns=["camada", "UASG", "ufs_distintas", "ufs"])
    else:
        grouped = pairs.groupby("UASG")["UF"].agg(
            ufs_distintas="nunique",
            ufs=lambda values: ", ".join(sorted(set(values))),
        )
        conflicts = grouped.loc[grouped["ufs_distintas"] > 1].reset_index()
        conflicts.insert(0, "camada", layer)

    return summary, distribution, conflicts


def save_psv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, sep="|", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita a propagação de Estado/UF até o modelo do Power BI."
    )
    parser.add_argument(
        "--gold-current",
        type=Path,
        default=Path("data/gold/fact_compras.parquet"),
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("data/pilot_medidores/silver/compras_medidores_piloto.parquet"),
    )
    parser.add_argument(
        "--powerbi-fact",
        type=Path,
        default=Path("data/staging_powerbi/integracao_medidores/powerbi/FatoCompras.parquet"),
    )
    parser.add_argument(
        "--powerbi-dim-uasg",
        type=Path,
        default=Path("data/staging_powerbi/integracao_medidores/powerbi/DimUASG.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/staging_powerbi/auditoria_estado_uf"),
    )
    args = parser.parse_args()

    inputs = {
        "GOLD_ATUAL": (
            args.gold_current,
            ["uf_uasg", "estado", "uf", "sigla_uf"],
            ["uasg", "codigo_uasg"],
        ),
        "PILOTO_MEDIDORES": (
            args.pilot,
            ["estado", "uf_uasg", "uf", "sigla_uf"],
            ["codigo_uasg", "uasg"],
        ),
        "POWERBI_FATO_STAGING": (
            args.powerbi_fact,
            ["Estado", "estado", "uf_uasg", "uf"],
            ["UASGKey", "uasg", "codigo_uasg"],
        ),
        "POWERBI_DIMUASG_STAGING": (
            args.powerbi_dim_uasg,
            ["Estado", "estado", "uf_uasg", "uf"],
            ["UASGKey", "uasg", "codigo_uasg"],
        ),
    }

    missing = [str(path.resolve()) for path, _, _ in inputs.values() if not path.exists()]
    if missing:
        raise SystemExit("Arquivos ausentes:\n" + "\n".join(missing))

    summaries: list[dict[str, Any]] = []
    distributions: list[pd.DataFrame] = []
    conflicts: list[pd.DataFrame] = []

    for layer, (path, state_candidates, uasg_candidates) in inputs.items():
        dataframe = pd.read_parquet(path)
        summary, distribution, layer_conflicts = summarize_layer(
            layer,
            dataframe,
            state_candidates,
            uasg_candidates,
        )
        summaries.append(summary)
        distributions.append(distribution)
        conflicts.append(layer_conflicts)

    summary_frame = pd.DataFrame(summaries)
    distribution_frame = pd.concat(distributions, ignore_index=True)
    conflict_frame = pd.concat(conflicts, ignore_index=True)

    save_psv(summary_frame, args.output_dir / "cobertura_estado_uf.psv")
    save_psv(distribution_frame, args.output_dir / "distribuicao_uf.psv")
    save_psv(conflict_frame, args.output_dir / "conflitos_uasg_estado.psv")

    expected = {
        "GOLD_ATUAL": (2230, 2230, 27),
        "PILOTO_MEDIDORES": (103, 103, 19),
        "POWERBI_FATO_STAGING": (2322, 2322, 27),
        "POWERBI_DIMUASG_STAGING": (719, 719, 27),
    }

    checks: list[dict[str, Any]] = []
    for row in summaries:
        layer = str(row["camada"])
        expected_rows, expected_filled, expected_states = expected[layer]
        checks.append(
            {
                "camada": layer,
                "linhas_esperadas": expected_rows,
                "linhas_encontradas": row["linhas"],
                "preenchidos_esperados": expected_filled,
                "preenchidos_encontrados": row["estado_preenchido"],
                "ufs_esperadas": expected_states,
                "ufs_encontradas": row["ufs_distintas"],
                "resultado": (
                    row["linhas"] == expected_rows
                    and row["estado_preenchido"] == expected_filled
                    and row["ufs_distintas"] == expected_states
                ),
            }
        )

    checks_frame = pd.DataFrame(checks)
    save_psv(checks_frame, args.output_dir / "validacao_estado_uf.psv")

    print("Auditoria de Estado/UF concluída.")
    print(summary_frame.to_string(index=False))
    print()
    print("Conflitos UASG x UF:", len(conflict_frame))
    print("Validações aprovadas:", int(checks_frame["resultado"].sum()), "/", len(checks_frame))
    print("Diretório:", args.output_dir.resolve())

    if not checks_frame["resultado"].all() or not conflict_frame.empty:
        raise SystemExit("Existem validações pendentes em Estado/UF.")


if __name__ == "__main__":
    main()
