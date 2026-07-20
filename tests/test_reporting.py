from pathlib import Path

import pandas as pd

from src.reporting import read_psv, save_psv


def test_save_psv_keeps_semicolon_inside_field(tmp_path: Path) -> None:
    source = pd.DataFrame([{"objeto": "Secretaria de Obras; Secretaria de Saúde", "valor": 1}])

    output = save_psv(source, tmp_path / "relatorio.psv")
    loaded = read_psv(output)

    assert len(loaded) == 1
    assert loaded.loc[0, "objeto"] == ("Secretaria de Obras; Secretaria de Saúde")


def test_save_psv_removes_internal_line_breaks(tmp_path: Path) -> None:
    source = pd.DataFrame([{"objeto": "Linha 1\nLinha 2\r\nLinha 3", "valor": 1}])

    output = save_psv(source, tmp_path / "relatorio.psv")
    physical_lines = output.read_text(encoding="utf-8-sig").splitlines()
    loaded = read_psv(output)

    assert len(physical_lines) == 2
    assert loaded.loc[0, "objeto"] == "Linha 1 Linha 2 Linha 3"


def test_save_psv_preserves_pipe_inside_quoted_field(tmp_path: Path) -> None:
    source = pd.DataFrame([{"unidade_comparavel": "RO|CAP=100", "valor": 1}])

    output = save_psv(source, tmp_path / "relatorio.psv")
    loaded = read_psv(output)

    assert len(loaded) == 1
    assert loaded.loc[0, "unidade_comparavel"] == "RO|CAP=100"
