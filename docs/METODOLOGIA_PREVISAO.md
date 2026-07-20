# Metodologia de previsão

## Objetivo

Projetar aquisições públicas observadas para os 12 meses seguintes, com modelos explicáveis e cenários de planejamento.

A previsão não representa consumo físico, estoque necessário ou plano de compra de uma empresa.

## Série temporal

As compras são agregadas mensalmente por CATMAT. Meses sem compra permanecem na série com quantidade zero para preservar a intermitência real.

## Modelos avaliados

Conforme a unidade de modelagem, são avaliados modelos como:

- média histórica;
- médias móveis;
- sazonal ingênuo de 12 meses;
- perfil sazonal histórico;
- tendência linear;
- Croston SBA;
- média positiva ponderada pela frequência.

## Validação temporal

A seleção utiliza janelas de backtesting, evitando escolher um modelo apenas pelo ajuste ao histórico completo. Entre as métricas analisadas estão erro absoluto, WAPE, estabilidade e disponibilidade de histórico.

## Medidores de energia

O universo final contém sete CATMATs de medidores. A modelagem utiliza três unidades:

- CATMAT 422736 individualmente;
- segmento Medição de Massa;
- segmento Legado Eletromecânico.

As previsões dos segmentos são rateadas aos CATMATs originais com pesos baseados em quantidade histórica, compras distintas e meses ativos. A soma alocada é validada contra o total previsto da unidade de modelagem.

## Cenários

Cada CATMAT possui três cenários:

- Inferior;
- Base;
- Superior.

As faixas refletem erro histórico e incerteza operacional. Não são intervalos estatísticos formais.

## Limitações

- compras públicas são influenciadas por licitação, orçamento e política de contratação;
- baixa frequência reduz a confiança de algumas séries;
- dados externos não incorporam estoque, criticidade, consumo interno ou plano de obras;
- previsão agregada não possui granularidade por UF.
