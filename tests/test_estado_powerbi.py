from __future__ import annotations

import pandas as pd

from scripts import integrar_medidores_staging_powerbi as integrator
from scripts import preparar_modelo_powerbi as powerbi


def history_row() -> dict[str, object]:
    return {
        "purchase_id": "compra-1",
        "data_publicacao": "2025-01-10",
        "codigo_catmat": 408494,
        "familia_material": "Cabos Condutores",
        "quantidade": 10.0,
        "valor_unitario": 5.0,
        "valor_total": 50.0,
        "unidade_comparavel": "M",
        "cnpj_fornecedor": "00000000000100",
        "fornecedor_nome": "Fornecedor Teste",
        "uasg": "123456",
        "nome_orgao": "Órgão Teste",
        "uf_uasg": "SP",
        "municipio": "CAMPINAS",
        "poder": "E",
        "esfera": "F",
    }


def test_prepare_history_recognizes_uf_uasg() -> None:
    result = powerbi.prepare_history(pd.DataFrame([history_row()]))
    assert result.loc[0, "Estado"] == "SP"


def test_build_uasg_dimension_recognizes_uf_uasg() -> None:
    result = powerbi.build_uasg_dimension(pd.DataFrame([history_row()]))
    assert result.loc[0, "UASGKey"] == "123456"
    assert result.loc[0, "Estado"] == "SP"


def test_pilot_integration_preserves_state_in_gold_schema() -> None:
    pilot = pd.DataFrame(
        [
            {
                "codigo_catmat": 422736,
                "data_referencia": "2025-06-01",
                "quantidade": 2.0,
                "preco_unitario": 100.0,
                "valor_total_calculado": 200.0,
                "ni_fornecedor": "00000000000100",
                "nome_fornecedor": "Fornecedor Teste",
                "codigo_uasg": "123456",
                "nome_uasg": "Órgão Teste",
                "estado": "SP",
                "municipio": "CAMPINAS",
                "poder": "E",
                "esfera": "F",
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "capacidade_unidade_fornecimento": 1.0,
                "descricao_catalogo": "MEDIDOR TESTE",
            }
        ]
    )

    reference = pd.DataFrame(
        columns=[
            "purchase_id",
            "data_publicacao",
            "codigo_catmat",
            "familia_material",
            "quantidade",
            "valor_unitario",
            "valor_total",
            "unidade_comparavel",
            "uf_uasg",
            "municipio",
            "poder",
            "esfera",
        ]
    )

    result = integrator.build_pilot_history(
        pilot,
        reference,
        "Medidores Energia",
    )

    assert len(result) == 1
    assert result.loc[result.index[0], "uf_uasg"] == "SP"
