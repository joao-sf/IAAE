# Dashboard Power BI

O arquivo `IAAE_Dashboard_Final.pbix` contém o snapshot final do IAAE e pode ser aberto no Power BI Desktop sem executar novamente a API.

## Páginas

- Visão Executiva;
- Preços e Mercado;
- Planejamento.

## Atualização local

As consultas esperam as tabelas Parquet geradas em `data/powerbi`:

- FatoCompras;
- FatoPrevisao;
- ResumoPrevisao;
- DimCalendario;
- DimMaterial;
- DimFornecedor;
- DimUASG;
- DimUnidade;
- DimCenario.

Depois de clonar o repositório e gerar as tabelas, abra **Transformar dados → Configurações da fonte de dados** e aponte as consultas para a pasta local `data/powerbi`.

A previsão não possui granularidade geográfica. Portanto, filtros de Estado/UF afetam o histórico de compras, mas não devem ser sincronizados com a página Planejamento.

## Visualização pelo GitHub

O GitHub não renderiza o conteúdo do PBIX. Para uma leitura rápida do projeto, o repositório deve conter imagens das três páginas em `docs/images`. O PBIX permanece disponível para download e inspeção técnica.
