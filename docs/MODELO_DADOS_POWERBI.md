# Modelo de dados Power BI

## Estrutura

```text
DimCalendario ───────┬──── FatoCompras
                     └──── FatoPrevisao

DimMaterial ─────────┬──── FatoCompras
                     ├──── FatoPrevisao
                     └──── ResumoPrevisao

DimFornecedor ──────────── FatoCompras
DimUASG ────────────────── FatoCompras
DimUnidade ─────────────── FatoCompras
DimCenario ─────────────── FatoPrevisao
```

Não existe relacionamento direto entre `FatoCompras` e `FatoPrevisao`.

## Dimensões e fatos finais

| Tabela | Grão | Linhas no snapshot |
|---|---|---:|
| FatoCompras | uma observação de aquisição pública | 2.322 |
| FatoPrevisao | CATMAT × mês × cenário | 540 |
| ResumoPrevisao | um CATMAT | 15 |
| DimCalendario | uma data | 1.827 |
| DimMaterial | um CATMAT | 15 |
| DimFornecedor | um fornecedor | 768 |
| DimUASG | uma UASG | 719 |
| DimUnidade | CATMAT × unidade comparável | 33 |
| DimCenario | um cenário | 3 |

## Relacionamentos

Todos os relacionamentos são ativos, muitos-para-um e com filtro em direção única da dimensão para a fato.

## Cenários

A `DimCenario` possui:

- Inferior;
- Base;
- Superior.

Para 15 CATMATs, 12 meses e três cenários:

```text
15 × 12 × 3 = 540 linhas
```

## Páginas do dashboard

### Visão Executiva

Cobertura, valor, volume, fornecedores, UASGs, qualidade e evolução histórica.

### Preços e Mercado

Preço mediano comparável, média ponderada contextual, dispersão, HHI, fornecedores e tabela detalhada.

### Planejamento

Histórico, cenários, modelo selecionado, confiança, tipo de demanda, WAPE e próximos passos.

## Regra de comparabilidade

Preços não devem ser combinados quando CATMAT, unidade ou capacidade são incompatíveis. A `DimUnidade` e as medidas contextuais protegem essa regra.

## Estado/UF

`DimUASG[Estado]` filtra a `FatoCompras`. A previsão não possui dimensão geográfica, portanto Estado não deve ser sincronizado com Planejamento.
