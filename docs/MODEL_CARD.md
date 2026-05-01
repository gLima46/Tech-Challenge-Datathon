# Model Card — LSTM Price Forecaster

## Identificação

| Campo | Valor |
|---|---|
| Nome | `lstm-price-forecaster` |
| Versão | `0.1.0` |
| Tipo | Regressão de série temporal (forecasting recursivo) |
| Framework | TensorFlow / Keras |
| Owner | grupo-XX |
| Risk level | medium |
| Fase | Datathon Fase 05 |

## Uso pretendido

Prever preços de fechamento dos próximos 10 dias úteis a partir de 60 dias de histórico de um único ativo. Ferramenta exploratória — não deve ser usada como sinal único para decisão de investimento.

## Dados de treino

- Fonte: `yfinance` (Yahoo Finance), API pública.
- Ticker default: `DIS` (configurável via `configs/model_config.yaml`).
- Janela: 2014-01-01 a 2024-12-31.
- Variável-alvo: `Close` ajustado.
- Versão dos dados: hash SHA-256 dos primeiros 12 chars do conteúdo, registrado como tag `training_data_version` no MLflow.

## Arquitetura

---
Input (60, 1)
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(64)
  → Dropout(0.2)
  → Dense(1)
---

Otimizador `adam`, loss `mse`, callbacks `EarlyStopping(patience=10)` e `ReduceLROnPlateau`.

## Métricas

Calculadas em validação holdout (50% mais recente da série):

| Métrica | Valor de referência |
|---|---|
| MAE | preencher após primeiro treino |
| RMSE | preencher após primeiro treino |
| MAPE | preencher após primeiro treino |

## Limitações conhecidas

- Forecasting recursivo: erros se acumulam ao longo dos 10 dias. A previsão do dia 10 depende das previsões dos dias 1-9.
- Univariado: usa apenas `Close`. Não incorpora volume, indicadores técnicos, sentimento ou fatores macroeconômicos.
- Sem tratamento de eventos atípicos: splits, fusões e dividendos especiais podem distorcer a previsão.
- Estacionariedade não verificada: o modelo assume implicitamente que a dinâmica recente se mantém.

## Considerações éticas

Modelo de uso interno educacional. Não é recomendação de investimento. Resultados não constituem promessa de retorno.

## Versionamento e rollback

Cada treino registra no MLflow Registry com tags obrigatórias (`git_sha`, `training_data_version`, `owner`, `risk_level`). Para reverter uma versão problemática:

---
bash
mlflow models update --name lstm-price-forecaster --version <N> --stage Archived
---

A API recarrega no próximo restart e passa a servir a versão imediatamente anterior em `Production`.

## Monitoramento em produção

- Drift de input: PSI sobre janela de preços em `src/monitoring/drift.py`. Threshold 0.10 (warning), 0.20 (retrain).
- Métricas de serving: Prometheus expõe latência, throughput e distribuição das predições em `/metrics`.
- Reports detalhados: Evidently rodado sob demanda via `make drift`.

## Métricas técnicas vs métricas de negócio

| Métrica técnica | Tradução para negócio | Meta operacional |
|---|---|---|
| MAE = $X | Erro médio em $X em previsão de fechamento → impacto direto em decisão de trade de N ações | MAE < $5 |
| MAPE = X% | Margem de erro percentual usada por traders pra dimensionar posição | MAPE < 5% |
| Latência p95 | Janela máxima pra rodar predição em pipeline de trade automatizado | < 500ms |
| Drift PSI > 0.20 | Mudança estrutural no ativo (split, fusão, regime macro) → modelo precisa retreinar | Trigger automático |