# Dicionário de dados

## FatoCompras

| Campo | Descrição |
|---|---|
| CompraID | chave técnica da observação |
| Data | data da aquisição observada |
| CodigoCATMAT | código do material |
| FornecedorKey | identificador normalizado do fornecedor |
| UASGKey | identificador da unidade compradora |
| UnidadeKey | CATMAT combinado com unidade comparável |
| Quantidade | quantidade observada |
| ValorUnitario | preço unitário informado ou calculado |
| ValorTotal | valor total observado |
| TipoPreco | natureza do preço disponível |
| Estado | UF da UASG |
| Municipio | município da UASG |
| DQDuplicado | indicador de duplicidade confirmada |
| DQPossuiErro | indicador consolidado de qualidade |
| AlertaPreco | sinalização para investigação de preço |
| GravidadeAlertaPreco | classificação do alerta |
| Fonte | origem pública do registro |

## FatoPrevisao

| Campo | Descrição |
|---|---|
| Data | mês previsto |
| CodigoCATMAT | material previsto |
| CenarioKey | Inferior, Base ou Superior |
| QuantidadePrevista | quantidade mensal projetada |
| ModeloSelecionado | modelo escolhido na validação |
| ConfiancaPrevisao | classificação da confiança |
| TipoDemanda | perfil observado da série |
| ErroReferenciaMensal | referência de erro usada nos cenários |

## DimMaterial

Contém CATMAT, família, descrição e informações resumidas da previsão.

## DimUASG

Contém UASG, órgão, Estado, município, Poder e esfera. O snapshot final possui 719 UASGs e 27 UFs.

## DimUnidade

Representa a combinação entre CATMAT, unidade e capacidade. É usada para impedir comparações de preço incompatíveis.
