from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .text_utils import normalize_text, parse_decimal

FIELD_ALIASES = {
    "codigo": (
        "codigo",
        "codigoItem",
        "codigo_item",
        "codigo_item_material",
    ),
    "descricao": (
        "descricao",
        "descricaoItem",
        "descricao_item",
        "nome",
    ),
    "status": (
        "status",
        "statusItem",
        "situacaoItem",
    ),
    "sustentavel": (
        "sustentavel",
        "indicadorSustentavel",
    ),
    "id_classe": (
        "id_classe",
        "classe",
        "codigoClasse",
    ),
    "id_grupo": (
        "id_grupo",
        "grupo",
        "codigoGrupo",
    ),
    "id_pdm": (
        "id_pdm",
        "pdm",
        "codigoPdm",
        "codigoPDM",
    ),
}

UNIT_MAP = {
    "unidade": "UN",
    "un": "UN",
    "peca": "PC",
    "pecas": "PC",
    "pc": "PC",
    "metro": "M",
    "metros": "M",
    "m": "M",
    "quilograma": "KG",
    "kg": "KG",
    "rolo": "RO",
    "ro": "RO",
    "bobina": "BOBINA",
    "conjunto": "CJ",
    "caixa": "CX",
    "litro": "L",
    "galao": "GL",
}


def first_present(record: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return None


def material_records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in records:
        row = {target: first_present(record, aliases) for target, aliases in FIELD_ALIASES.items()}
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(FIELD_ALIASES))

    dataframe = pd.DataFrame(rows)
    dataframe["codigo"] = pd.to_numeric(
        dataframe["codigo"],
        errors="coerce",
    ).astype("Int64")
    dataframe["descricao"] = dataframe["descricao"].fillna("").astype(str).str.strip()

    return dataframe.drop_duplicates(subset=["codigo", "descricao"]).reset_index(drop=True)


def standardize_unit(value: Any) -> str | None:
    normalized = normalize_text(value)

    if not normalized:
        return None

    return UNIT_MAP.get(normalized, str(value).strip().upper())


def format_capacity(value: Any) -> str | None:
    capacity = parse_decimal(value)

    if capacity is None or capacity <= 0:
        return None

    if float(capacity).is_integer():
        return str(int(capacity))

    return f"{capacity:.6f}".rstrip("0").rstrip(".")


def build_comparable_unit(
    supply_unit: Any,
    supply_capacity: Any,
) -> str | None:
    standardized_unit = standardize_unit(supply_unit)

    if standardized_unit is None:
        return None

    capacity = format_capacity(supply_capacity)

    if capacity is None or capacity == "1":
        return standardized_unit

    return f"{standardized_unit}|CAP={capacity}"


def build_purchase_row(
    tender: dict[str, Any],
    item: dict[str, Any],
    material_lookup: dict[int, dict[str, Any]],
    source_type: str,
) -> dict[str, Any]:
    """Mantido apenas para compatibilidade com testes legados."""

    material_code = first_present(
        item,
        ("codigo_item_material", "codigoItemMaterial", "codigo_material"),
    )

    try:
        material_code_int = int(material_code) if material_code is not None else None
    except TypeError, ValueError:
        material_code_int = None

    material = material_lookup.get(material_code_int or -1, {})
    quantity = parse_decimal(first_present(item, ("quantidade", "quantidade_item")))
    unit_price = parse_decimal(first_present(item, ("valor_unitario", "valorUnitario")))
    total_value = parse_decimal(
        first_present(item, ("valor_total", "valorTotal", "valor_estimado"))
    )

    if unit_price is None and quantity and total_value is not None and quantity != 0:
        unit_price = total_value / quantity

    if total_value is None and quantity is not None and unit_price is not None:
        total_value = quantity * unit_price

    tender_id = first_present(
        tender,
        ("identificador", "id_licitacao", "numero_licitacao"),
    )
    item_number = first_present(
        item,
        ("numero_item_licitacao", "numero_item", "item"),
    )
    raw_key = f"{tender_id}|{item_number}|{material_code_int}|{source_type}"
    purchase_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    original_unit = first_present(
        item,
        ("unidade", "unidade_fornecimento"),
    )

    return {
        "purchase_id": purchase_id,
        "id_licitacao": tender_id,
        "numero_item": item_number,
        "data_publicacao": first_present(
            tender,
            ("data_publicacao", "data_abertura_proposta"),
        ),
        "codigo_catmat": material_code_int,
        "descricao_item": first_present(
            item,
            ("descricao_item", "descricao"),
        )
        or material.get("descricao"),
        "familia_codigo": material.get("familia_codigo"),
        "familia_material": material.get("familia_material"),
        "quantidade": quantity,
        "unidade_original": original_unit,
        "unidade_padronizada": standardize_unit(original_unit),
        "unidade_comparavel": standardize_unit(original_unit),
        "valor_unitario": unit_price,
        "valor_total": total_value,
        "cnpj_fornecedor": first_present(
            item,
            ("cnpj_fornecedor", "cnpjVencedor", "cnpj_vencedor"),
        ),
        "uasg": first_present(item, ("uasg",)) or first_present(tender, ("uasg",)),
        "uf_uasg": first_present(tender, ("uf_uasg", "uf")),
        "modalidade": first_present(item, ("modalidade",))
        or first_present(tender, ("modalidade",)),
        "objeto": first_present(tender, ("objeto",)),
        "fonte_item": source_type,
    }


def build_practiced_price_row(
    record: dict[str, Any],
    material_lookup: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Converte um preço praticado da API atual para o modelo analítico."""

    material_code = first_present(
        record,
        ("codigoItemCatalogo", "codigo_item_catalogo"),
    )

    try:
        material_code_int = int(material_code) if material_code is not None else None
    except TypeError, ValueError:
        material_code_int = None

    material = material_lookup.get(material_code_int or -1, {})

    quantity = parse_decimal(first_present(record, ("quantidade",)))
    unit_price = parse_decimal(first_present(record, ("precoUnitario", "valorUnitario")))
    total_value = quantity * unit_price if quantity is not None and unit_price is not None else None

    purchase_raw_id = first_present(record, ("idCompra",))
    id_item_compra = first_present(record, ("idItemCompra",))
    id_compra_item = first_present(record, ("idCompraItem",))
    item_number = first_present(record, ("numeroItemCompra",))

    result_date = first_present(record, ("dataResultado",))
    purchase_date = first_present(record, ("dataCompra",))
    supplier_id = first_present(record, ("niFornecedor",))
    uasg_code = first_present(record, ("codigoUasg",))

    raw_key = "|".join(
        str(value)
        for value in [
            id_compra_item,
            id_item_compra,
            purchase_raw_id,
            item_number,
            material_code_int,
            result_date or purchase_date,
            supplier_id,
            uasg_code,
            quantity,
            unit_price,
            first_present(record, ("siglaUnidadeFornecimento",)),
            first_present(record, ("capacidadeUnidadeFornecimento",)),
            "PRECO_PRATICADO",
        ]
    )
    purchase_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    supply_unit_abbreviation = first_present(
        record,
        ("siglaUnidadeFornecimento",),
    )
    supply_unit_name = first_present(
        record,
        ("nomeUnidadeFornecimento",),
    )
    supply_capacity = parse_decimal(first_present(record, ("capacidadeUnidadeFornecimento",)))
    original_unit = supply_unit_abbreviation or supply_unit_name
    standardized_unit = standardize_unit(original_unit)
    comparable_unit = build_comparable_unit(
        original_unit,
        supply_capacity,
    )

    return {
        "purchase_id": purchase_id,
        "id_compra": purchase_raw_id,
        "id_item_compra": id_item_compra,
        "id_compra_item": id_compra_item,
        "numero_item": item_number,
        "data_publicacao": result_date or purchase_date,
        "data_compra": purchase_date,
        "data_resultado": result_date,
        "codigo_catmat": material_code_int,
        "descricao_item": first_present(record, ("descricaoItem",)) or material.get("descricao"),
        "descricao_detalhada": first_present(
            record,
            ("descricaoDetalhadaItem",),
        ),
        "familia_codigo": material.get("familia_codigo"),
        "familia_material": material.get("familia_material"),
        "quantidade": quantity,
        "unidade_original": original_unit,
        "unidade_padronizada": standardized_unit,
        "sigla_unidade_fornecimento": supply_unit_abbreviation,
        "nome_unidade_fornecimento": supply_unit_name,
        "capacidade_unidade_fornecimento": supply_capacity,
        "sigla_unidade_medida": first_present(
            record,
            ("siglaUnidadeMedida",),
        ),
        "nome_unidade_medida": first_present(
            record,
            ("nomeUnidadeMedida",),
        ),
        "unidade_comparavel": comparable_unit,
        "valor_unitario": unit_price,
        "valor_total": total_value,
        "tipo_valor": "PRATICADO",
        "cnpj_fornecedor": supplier_id,
        "nome_fornecedor": first_present(record, ("nomeFornecedor",)),
        "marca": first_present(record, ("marca",)),
        "uasg": uasg_code,
        "nome_uasg": first_present(record, ("nomeUasg",)),
        "codigo_orgao": first_present(record, ("codigoOrgao",)),
        "nome_orgao": first_present(record, ("nomeOrgao",)),
        "codigo_municipio": first_present(record, ("codigoMunicipio",)),
        "municipio": first_present(record, ("municipio",)),
        "uf_uasg": first_present(record, ("estado",)),
        "poder": first_present(record, ("poder",)),
        "esfera": first_present(record, ("esfera",)),
        "forma": first_present(record, ("forma",)),
        "modalidade": first_present(record, ("modalidade",)),
        "criterio_julgamento": first_present(
            record,
            ("criterioJulgamento",),
        ),
        "percentual_maior_desconto": parse_decimal(
            first_present(record, ("percentualMaiorDesconto",))
        ),
        "codigo_classe": first_present(record, ("codigoClasse",)),
        "nome_classe": first_present(record, ("nomeClasse",)),
        "codigo_pdm": first_present(record, ("codigoPdm",)),
        "nome_pdm": first_present(record, ("nomePdm",)),
        "objeto": first_present(record, ("objetoCompra",)),
        "data_atualizacao_fato": first_present(
            record,
            ("dataAtualizacaoFato",),
        ),
        "fonte_item": "API_PRECO_PRATICADO",
    }
