from pathlib import Path

import pandas as pd

from src.classifier import MaterialClassifier

CONFIG = Path(__file__).resolve().parents[1] / "config" / "material_categories.yml"


def test_classifies_electrical_cable() -> None:
    classifier = MaterialClassifier(CONFIG)
    result = classifier.classify("Cabo elétrico de potência, cobre, 0,6/1 kV")
    assert result.family_code == "cabos_condutores"


def test_excludes_network_cable() -> None:
    classifier = MaterialClassifier(CONFIG)
    result = classifier.classify("Cabo de rede categoria 6 com conector RJ45")
    assert result.family_code is None


def test_classifies_energy_meter() -> None:
    classifier = MaterialClassifier(CONFIG)
    result = classifier.classify("Medidor eletrônico de energia elétrica trifásico")
    assert result.family_code == "medidores_energia"


def test_dataframe_adds_audit_columns() -> None:
    classifier = MaterialClassifier(CONFIG)
    df = pd.DataFrame({"descricao": ["Disjuntor termomagnético", "Medidor de água"]})
    result = classifier.classify_dataframe(df)
    assert result.loc[0, "familia_codigo"] == "disjuntores_chaves"
    assert pd.isna(result.loc[1, "familia_codigo"])
    assert "classificacao_motivo" in result.columns
