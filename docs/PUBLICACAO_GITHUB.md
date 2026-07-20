# Publicação no GitHub

## Conteúdo público

O repositório deve conter código, testes, configurações, documentação, amostra fictícia e o PBIX final.

Não publicar:

- `.env`;
- `.venv`;
- caches;
- logs de execução;
- Bronze, Silver, Gold ou staging completos;
- backups;
- arquivos temporários;
- versões antigas do dashboard;
- pacotes ZIP intermediários.

## Validação antes do commit

```powershell
python .\scripts\validar_repositorio_publico.py
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=src --cov-report=term-missing
```

## Visualização

Antes da publicação final, exporte imagens das três páginas do Power BI e salve em:

```text
docs/images/visao-executiva.png
docs/images/precos-mercado.png
docs/images/planejamento.png
```

Depois, inclua essas imagens no README.
