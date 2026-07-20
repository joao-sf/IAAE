from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


def sanitize_report_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepara uma tabela para relatórios PSV de leitura humana.

    - remove quebras de linha e tabulações internas;
    - preserva ponto e vírgula como parte do texto;
    - preserva pipes internos por meio do quoting CSV;
    - não altera o DataFrame original.
    """

    report = dataframe.copy()

    text_columns = report.select_dtypes(
        include=["object", "string"],
    ).columns

    for column in text_columns:
        series = report[column].astype("string")
        series = series.str.replace(r"[\r\n\t]+", " ", regex=True)
        series = series.str.replace(r"\s{2,}", " ", regex=True)
        report[column] = series.str.strip()

    return report


def save_psv(
    dataframe: pd.DataFrame,
    path: Path,
) -> Path:
    """Salva relatório separado por pipe com uma linha física por registro."""

    output_path = path.with_suffix(".psv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = sanitize_report_dataframe(dataframe)
    report.to_csv(
        output_path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        lineterminator="\n",
    )

    return output_path


def read_psv(path: Path) -> pd.DataFrame:
    """Lê um relatório PSV produzido pelo projeto."""

    return pd.read_csv(
        path,
        sep="|",
        encoding="utf-8-sig",
    )


def safe_report_title(path: Path) -> str:
    """Cria um título curto para relatórios e logs."""

    name = re.sub(r"[_-]+", " ", path.stem).strip()
    return name.title()
