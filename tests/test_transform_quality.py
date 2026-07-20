import pandas as pd

from src.quality import apply_quality_rules, flag_price_outliers
from src.transform import (
    build_comparable_unit,
    build_practiced_price_row,
    build_purchase_row,
)


def test_build_purchase_row_calculates_unit_price() -> None:
    tender = {
        "identificador": "ABC",
        "data_publicacao": "2025-01-10",
        "uasg": 123,
    }
    item = {
        "numero_item_licitacao": 1,
        "codigo_item_material": 999,
        "quantidade": "10",
        "valor_total": "1.000,00",
        "unidade": "unidade",
    }
    lookup = {
        999: {
            "descricao": "Disjuntor",
            "familia_codigo": "x",
            "familia_material": "X",
        }
    }
    row = build_purchase_row(tender, item, lookup, "item_licitacao")
    assert row["valor_unitario"] == 100.0
    assert row["unidade_padronizada"] == "UN"


def test_comparable_unit_uses_supply_capacity() -> None:
    assert build_comparable_unit("RO", 100) == "RO|CAP=100"
    assert build_comparable_unit("UN", 1) == "UN"
    assert build_comparable_unit("M", 0) == "M"


def test_quality_flags_invalid_quantity() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "purchase_id": "1",
                "id_compra_item": "item-1",
                "data_publicacao": "2025-01-01",
                "codigo_catmat": 10,
                "quantidade": 0,
                "valor_unitario": 5,
                "valor_total": 0,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            }
        ]
    )
    result, report = apply_quality_rules(dataframe)
    assert bool(result.loc[0, "dq_quantidade_invalida"])
    assert (
        report.loc[
            report["regra"] == "dq_quantidade_invalida",
            "quantidade",
        ].iloc[0]
        == 1
    )


def test_same_purchase_with_different_items_is_not_duplicate() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "purchase_id": "hash-1",
                "id_compra": "compra-1",
                "id_compra_item": "compra-1-item-1",
                "id_item_compra": "1",
                "data_publicacao": "2025-01-01",
                "codigo_catmat": 10,
                "quantidade": 2,
                "valor_unitario": 5,
                "valor_total": 10,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            },
            {
                "purchase_id": "hash-2",
                "id_compra": "compra-1",
                "id_compra_item": "compra-1-item-2",
                "id_item_compra": "2",
                "data_publicacao": "2025-01-01",
                "codigo_catmat": 10,
                "quantidade": 2,
                "valor_unitario": 5,
                "valor_total": 10,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            },
        ]
    )
    result, _ = apply_quality_rules(dataframe)
    assert int(result["dq_duplicado"].sum()) == 0


def test_repeated_item_key_is_duplicate() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "purchase_id": "hash-1",
                "id_compra_item": "item-1",
                "data_publicacao": "2025-01-01",
                "codigo_catmat": 10,
                "quantidade": 2,
                "valor_unitario": 5,
                "valor_total": 10,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            },
            {
                "purchase_id": "hash-2",
                "id_compra_item": "item-1",
                "data_publicacao": "2025-01-01",
                "codigo_catmat": 10,
                "quantidade": 2,
                "valor_unitario": 5,
                "valor_total": 10,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            },
        ]
    )
    result, _ = apply_quality_rules(dataframe)
    assert int(result["dq_duplicado"].sum()) == 2


def test_build_practiced_price_row_preserves_supply_unit() -> None:
    record = {
        "idCompra": "98600105900052025",
        "idItemCompra": 7749581,
        "idCompraItem": "9860010590005202500026",
        "numeroItemCompra": 26,
        "codigoItemCatalogo": 610539,
        "descricaoItem": "TINTA ESMALTE",
        "quantidade": 5.0,
        "precoUnitario": 139.76,
        "siglaUnidadeFornecimento": "GL",
        "nomeUnidadeFornecimento": "GALÃO 3,6 L",
        "capacidadeUnidadeFornecimento": 3.6,
        "siglaUnidadeMedida": "L",
        "dataCompra": "2025-03-21",
        "dataResultado": "2025-03-21",
        "niFornecedor": "40932123000185",
        "nomeFornecedor": "Fornecedor teste",
        "codigoUasg": 986001,
        "estado": "RJ",
    }
    lookup = {
        610539: {
            "descricao": "TINTA ESMALTE",
            "familia_codigo": "teste_api",
            "familia_material": "Teste da API",
        }
    }

    row = build_practiced_price_row(record, lookup)

    assert row["codigo_catmat"] == 610539
    assert row["valor_total"] == 698.80
    assert row["sigla_unidade_fornecimento"] == "GL"
    assert row["nome_unidade_fornecimento"] == "GALÃO 3,6 L"
    assert row["capacidade_unidade_fornecimento"] == 3.6
    assert row["unidade_comparavel"] == "GL|CAP=3.6"


def test_outlier_is_calculated_by_unit_and_year() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "purchase_id": f"a-{index}",
                "id_compra_item": f"a-{index}",
                "data_publicacao": "2025-01-01",
                "codigo_catmat": 10,
                "quantidade": 1,
                "valor_unitario": 10 + index % 2,
                "valor_total": 10 + index % 2,
                "sigla_unidade_fornecimento": "M",
                "nome_unidade_fornecimento": "METRO",
                "unidade_comparavel": "M",
            }
        )
    rows.append(
        {
            "purchase_id": "a-outlier",
            "id_compra_item": "a-outlier",
            "data_publicacao": "2025-01-01",
            "codigo_catmat": 10,
            "quantidade": 1,
            "valor_unitario": 100,
            "valor_total": 100,
            "sigla_unidade_fornecimento": "M",
            "nome_unidade_fornecimento": "METRO",
            "unidade_comparavel": "M",
        }
    )
    rows.append(
        {
            "purchase_id": "b-1",
            "id_compra_item": "b-1",
            "data_publicacao": "2025-01-01",
            "codigo_catmat": 10,
            "quantidade": 1,
            "valor_unitario": 100,
            "valor_total": 100,
            "sigla_unidade_fornecimento": "RO",
            "nome_unidade_fornecimento": "ROLO",
            "unidade_comparavel": "RO|CAP=100",
        }
    )

    quality, _ = apply_quality_rules(pd.DataFrame(rows))
    result = flag_price_outliers(quality)

    flagged = result.loc[result["is_price_outlier"], "purchase_id"].tolist()
    assert flagged == ["a-outlier"]


def test_same_item_with_different_date_or_supplier_is_history() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "purchase_id": "obs-1",
                "id_compra": "compra-1",
                "id_compra_item": "item-1",
                "id_item_compra": "100",
                "numero_item": 1,
                "data_publicacao": "2024-01-10",
                "codigo_catmat": 10,
                "quantidade": 2,
                "valor_unitario": 5,
                "valor_total": 10,
                "cnpj_fornecedor": "11111111000111",
                "uasg": 1,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            },
            {
                "purchase_id": "obs-2",
                "id_compra": "compra-1",
                "id_compra_item": "item-1",
                "id_item_compra": "100",
                "numero_item": 1,
                "data_publicacao": "2025-02-20",
                "codigo_catmat": 10,
                "quantidade": 3,
                "valor_unitario": 6,
                "valor_total": 18,
                "cnpj_fornecedor": "22222222000122",
                "uasg": 1,
                "sigla_unidade_fornecimento": "UN",
                "nome_unidade_fornecimento": "UNIDADE",
                "unidade_comparavel": "UN",
            },
        ]
    )

    result, _ = apply_quality_rules(dataframe)

    assert int(result["dq_duplicado"].sum()) == 0
    assert int(result["hist_item_repetido"].sum()) == 2


def test_exact_observation_is_duplicate() -> None:
    row = {
        "purchase_id": "obs-1",
        "id_compra": "compra-1",
        "id_compra_item": "item-1",
        "id_item_compra": "100",
        "numero_item": 1,
        "data_publicacao": "2025-02-20",
        "codigo_catmat": 10,
        "quantidade": 3,
        "valor_unitario": 6,
        "valor_total": 18,
        "cnpj_fornecedor": "22222222000122",
        "uasg": 1,
        "sigla_unidade_fornecimento": "UN",
        "nome_unidade_fornecimento": "UNIDADE",
        "unidade_comparavel": "UN",
    }

    result, _ = apply_quality_rules(pd.DataFrame([row, row.copy()]))

    assert int(result["dq_duplicado"].sum()) == 2
