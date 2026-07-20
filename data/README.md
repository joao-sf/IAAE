# Dados

Os dados operacionais não são versionados neste repositório.

A estrutura é criada e preenchida durante a execução:

```text
data/
├── bronze/     respostas JSON brutas da API
├── silver/     dados tipados, padronizados e validados
├── gold/       fatos, dimensões, análises e previsões
├── powerbi/    tabelas Parquet consumidas pelo dashboard
└── sample/     pequena amostra versionada para demonstração
```

Motivos para não versionar as camadas completas:

- são artefatos reproduzíveis a partir de dados públicos;
- podem crescer significativamente;
- geram ruído no histórico do Git;
- contêm nomes e identificadores públicos de fornecedores que não são necessários para revisar o código.

A amostra em `sample/amostra_materiais.csv` é fictícia e serve apenas para demonstrar a lógica de classificação.
