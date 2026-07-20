# Reprodução do projeto

## 1. Preparar o ambiente

```powershell
.\scripts\setup_vscode.ps1
.\scripts\diagnostico_ambiente.ps1
```

## 2. Validar a API

```powershell
python .\main.py smoke-test --code 610532
python .\main.py price-smoke-test --code 610539 --uasg 986001 --page-size 10
```

## 3. Executar a extração principal

```powershell
python .\main.py run `
  --start-date 2021-01-01 `
  --end-date 2025-12-31 `
  --catalog-file .\config\catmat_eletricos.csv
```

A execução completa depende da disponibilidade e do desempenho da API pública.

## 4. Reconstruir a Gold sem consultar a API

```powershell
python .\main.py rebuild `
  --catalog-file .\config\catmat_eletricos.csv `
  --silver-file .\data\silver\precos_praticados.parquet
```

## 5. Gerar análises

```powershell
python .\scripts\gerar_analises_iaae.py
```

## 6. Gerar previsão geral

```powershell
python .\scripts\gerar_previsao_catmat_v2.py
```

## 7. Modelar e prever medidores

```powershell
python .\scripts\preparar_modelagem_medidores.py
python .\scripts\gerar_previsao_hierarquica_medidores.py
```

## 8. Preparar modelo do Power BI

```powershell
python .\scripts\preparar_modelo_powerbi.py `
  --history .\data\gold\fact_compras.parquet `
  --forecast .\data\gold\previsao_catmat_v2\previsao_catmat_v2.parquet `
  --forecast-summary .\data\gold\previsao_catmat_v2\resumo_previsao_catmat_v2.parquet `
  --output-dir .\data\powerbi
```

## 9. Qualidade

```powershell
.\scripts\qa.ps1
python .\scripts\validar_repositorio_publico.py
```

Algumas rotinas especializadas possuem parâmetros adicionais. Consulte `--help` antes da execução.
