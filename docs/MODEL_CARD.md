# Model Card — LSTM Price Forecaster

> Documento de transparência do modelo, seguindo o template de Mitchell et al. (2019) — *Model Cards for Model Reporting*.

## Identificação

| Campo | Valor |
|---|---|
| Nome | `lstm-price-forecaster` |
| Versão | `0.1.0` |
| Tipo | Regressão de série temporal (forecasting recursivo) |
| Framework | TensorFlow 2.16 / Keras 3 |
| Owner | grupo-XX |
| Risk level | medium |
| Fase | Datathon Fase 05 |
| Data de registro | 2026-05-XX |
| Commit SHA | rastreável via tag `git_sha` no MLflow Registry |

## Uso pretendido

Prever preços de fechamento dos próximos 10 dias úteis a partir de 60 dias de histórico de um único ativo. Ferramenta exploratória para fins acadêmicos — **não deve ser usada como sinal único para decisão de investimento**.

### Casos de uso fora do escopo

- Decisão automatizada de compra ou venda em produção real
- Previsão em horizontes acima de 10 dias (erros se acumulam exponencialmente)
- Modelagem de ativos com comportamento radicalmente distinto do ticker treinado (ex: criptomoedas, commodities)
- Análise de risco regulatório (Basileia, VaR)

## Dados de treino

| Campo | Valor |
|---|---|
| Fonte | `yfinance` (Yahoo Finance), API pública |
| Ticker default | `DIS` (configurável via `configs/model_config.yaml`) |
| Janela | 2014-01-01 a 2024-12-31 (10 anos, ~2.500 dias úteis) |
| Variável-alvo | `Close` ajustado |
| Versão dos dados | hash SHA-256 (12 chars) registrado como tag `training_data_version` no MLflow |
| Split | 50% mais recente da série como holdout temporal (sem shuffle) |
| Pré-processamento | `MinMaxScaler` ajustado apenas no treino, persistido como artefato |

### Validação de schema

Antes do treino, os dados passam por validação com `pandera`:

- Coluna `Close` deve ser `float` positivo
- Índice deve ser `DatetimeIndex` ordenado
- Sem nulls em colunas críticas
- Pelo menos 60 observações disponíveis (mínimo para gerar uma sequência)

## Arquitetura

```
Input (60, 1)
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(64)
  → Dropout(0.2)
  → Dense(1)
```

| Componente | Configuração |
|---|---|
| Otimizador | `adam` (lr inicial 1e-3) |
| Loss | `mse` |
| Batch size | 32 |
| Épocas | 50 (com early stopping) |
| Callbacks | `EarlyStopping(patience=10)`, `ReduceLROnPlateau(factor=0.5, patience=5)` |
| Seed | 42 (fixada em numpy, tensorflow, random e PYTHONHASHSEED) |

## Métricas

Calculadas em validação holdout (50% mais recente da série), no espaço original de preço (não normalizado):

| Métrica | Valor (run de referência) | Meta operacional |
|---|---|---|
| **MAE** | 2.70 USD | < 5.00 USD |
| **RMSE** | 3.73 USD | < 6.00 USD |
| **MAPE** | 2.31% | < 5% |
| **Latência p95 (inferência)** | ~750 ms | < 1000 ms |

> Métricas exatas variam entre runs por causa de randomicidade residual. Valores acima são do run `0e85f90e21c241699e1a7d7e658ccc9b`. Para métricas atualizadas, consultar o MLflow UI em `http://localhost:5000`.

## Limitações conhecidas

### Técnicas

- **Forecasting recursivo:** erros se acumulam ao longo dos 10 dias. A previsão do dia 10 depende das previsões dos dias 1-9.
- **Univariado:** usa apenas `Close`. Não incorpora volume, indicadores técnicos, sentimento ou fatores macroeconômicos.
- **Sem tratamento de eventos atípicos:** splits, fusões e dividendos especiais podem distorcer a previsão.
- **Estacionariedade não verificada:** o modelo assume implicitamente que a dinâmica recente se mantém.
- **Single ticker:** modelo treinado em um único ativo não generaliza para outros sem retreino.

### Operacionais

- **Cold start no rollback:** após repontar o alias, a API só recarrega o modelo no próximo restart do container (`docker compose restart api`). Para evitar restart, seria necessário implementar hot reload no `app.py`.
- **Sem A/B testing nativo:** comparação champion/challenger é manual, via comparação de runs no MLflow UI.

## Considerações éticas

Modelo de uso interno educacional. **Não é recomendação de investimento.** Resultados não constituem promessa de retorno. Não foram realizadas auditorias de fairness pois o domínio (previsão de preço de ativo único) não envolve grupos demográficos protegidos pela LGPD ou regulamentações de mercado.

## Versionamento e rollback

Cada treino registra no MLflow Registry com tags obrigatórias:

| Tag | Propósito |
|---|---|
| `model_name` | Nome para registry |
| `model_type` | `regression-timeseries-lstm` |
| `framework` | `tensorflow-keras` |
| `owner` | Responsável pelo modelo |
| `risk_level` | `low` / `medium` / `high` / `critical` |
| `training_data_version` | SHA-256 dos dados de treino (lineage) |
| `git_sha` | Commit do código (rastreabilidade) |
| `phase` | Contexto do projeto |

### Rollback via alias

A API serve o modelo via alias `@production` do Model Registry:

```bash
# Repontar alias para versão anterior (rollback)
mlflow models alias --name lstm-price-forecaster --alias production --version <N_ANTERIOR>

# Recarregar a API
docker compose restart api
```

### Reprodução exata de uma versão

Como cada versão tem `git_sha` e `training_data_version` registrados, reprodução é determinística:

```bash
git checkout <git_sha>
dvc checkout                # restaura versão dos dados
docker exec -it datathon-api python -m src.models.train
```

## Monitoramento em produção

| Dimensão | Ferramenta | Onde |
|---|---|---|
| Drift de input | PSI sobre janela de preços | `src/monitoring/drift.py`, threshold 0.10 (warning) / 0.20 (retrain) |
| Latência e throughput | Prometheus + Grafana | endpoint `/metrics`, dashboard em `http://localhost:3000` |
| Distribuição das predições | Histograma Prometheus | métrica `lstm_prediction_value_bucket` |
| Reports detalhados | Evidently sob demanda | `make drift` ou endpoint `/monitoring/drift_csv` |

## Métricas técnicas vs métricas de negócio

| Métrica técnica | Tradução para negócio | Meta operacional |
|---|---|---|
| MAE = $2.70 | Erro médio em $2.70 em previsão de fechamento → impacto direto em decisão de trade de N ações | MAE < $5 |
| MAPE = 2.31% | Margem de erro percentual usada por traders pra dimensionar posição | MAPE < 5% |
| Latência p95 | Janela máxima pra rodar predição em pipeline de trade automatizado | < 1000 ms |
| Drift PSI > 0.20 | Mudança estrutural no ativo (split, fusão, regime macro) → modelo precisa retreinar | Trigger automático |

## Referências

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. Proceedings of FAT* '19.
- Microsoft. *MLOps Maturity Model*. https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model
