# Portabilidade do Power BI

O dashboard usa um parâmetro de texto chamado `pBasePath` para localizar os arquivos Parquet gerados pelo pipeline.

## Configuração

No Power BI Desktop, abra **Transformar dados → Gerenciar parâmetros → Novo parâmetro** e configure:

- Nome: `pBasePath`
- Tipo: Texto
- Valor atual: caminho completo da pasta local `data/powerbi`

Para obter o caminho correto no Windows, execute o comando abaixo na raiz do repositório:

```powershell
(Resolve-Path .\data\powerbi).Path
```

Copie o resultado para o valor atual do parâmetro `pBasePath`.

## Consultas

A etapa de origem de cada consulta deve usar o parâmetro:

```powerquery
Fonte = Parquet.Document(
    File.Contents(pBasePath & "\\FatoCompras.parquet")
)
```

Aplique o mesmo padrão aos arquivos:

- `FatoCompras.parquet`
- `FatoPrevisao.parquet`
- `ResumoPrevisao.parquet`
- `DimCalendario.parquet`
- `DimMaterial.parquet`
- `DimFornecedor.parquet`
- `DimUASG.parquet`
- `DimUnidade.parquet`
- `DimCenario.parquet`

## Atualização em outro computador

1. Execute o pipeline para gerar `data/powerbi`.
2. Na raiz do repositório, obtenha o caminho com `(Resolve-Path .\data\powerbi).Path`.
3. Abra o PBIX.
4. Altere somente o valor de `pBasePath` para o caminho retornado.
5. Selecione **Fechar e Aplicar**.
6. Atualize o relatório.

Os arquivos Parquet de execução são ignorados pelo Git e não devem ser publicados no repositório.
