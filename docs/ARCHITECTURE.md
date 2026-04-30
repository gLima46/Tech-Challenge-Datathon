# Arquitetura - Datathon Fase 05

## Visão geral

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   yfinance   │───▶│  ingest (DVC)   │───▶│ data/raw/    │
└──────────────┘    └─────────────────┘    │  prices.pq   │
                                            └──────┬───────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │ feature_eng.py   │
                                          │ + Pandera schema │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │   train.py       │
                                          │   (LSTM Keras)   │
                                          └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  MLflow Registry │
                                          │  + tags + git_sha│
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │  FastAPI app.py  │
                                          │  + Prometheus    │
                                          └────────┬─────────┘
                                                   │
                            ┌──────────────────────┼──────────────────────┐
                            ▼                      ▼                      ▼
                  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
                  │  Cliente HTTP    │  │   Prometheus     │  │  drift.py (PSI)  │
                  └──────────────────┘  │  scrape /metrics │  │  evidently report│
                                        └────────┬─────────┘  └─────────┬────────┘
                                                 ▼                      │
                                        ┌──────────────────┐            │
                                        │   Grafana        │◀───────────┘
                                        └──────────────────┘
```

## Mapeamento Gap → Componente

| Gap (PDF) | Componente | Onde | Como |
|---|---|---|---|
| 01 - Sem monitoramento | Prometheus + Grafana | `src/serving/app.py` `docker-compose.yml` | `Instrumentator` expõe `/metrics`, Grafana consome |
| 02 - SPOF em notebook | Pipeline DVC | `dvc.yaml` | Stages isolados, sem notebook como trigger |
| 03 - Feature store destrutivo | Decisão arquitetural | `docs/ARCHITECTURE.md` §"GAP 03 em detalhe" | Não aplicável: modelo univariado, features chegam no request, custo > benefício. Cenários de evolução documentados. |
| 04 - Sem testes | pytest + cov 60% | `tests/` | `test_features`, `test_drift`, `test_api` |
| 05 - Sem governança de modelo | MLflow Registry | `src/models/train.py` | Tags obrigatórias + Model Registry |
| 06 - Sem drift detection | PSI + Evidently | `src/monitoring/drift.py` | Threshold 0.10 / 0.20 do PDF |
| 07 - Sem retraining auto | GitHub Actions | `.github/workflows/retrain.yml` `.github/workflows/drift-check.yml` | Retreino agendado semanal + retreino event-driven via drift |
| 08 - Dev sem dados | DVC | `dvc.yaml` + `data/raw/` | `dvc repro` reproduz tudo |
| 09 - Skill gap eng. soft | Type hints + ruff + mypy | `pyproject.toml` `.pre-commit` | Lint obrigatório no CI |

## GAP 03 em detalhe — por que não há feature store nesta arquitetura

O PDF do Datathon (p. 5) descreve o anti-padrão de **feature store destrutivo**:
caches Redis ou tabelas que sofrem `FLUSHALL` periódico, criando janelas em que
o serving fica sem features e o modelo retorna predições degradadas. A
recomendação canônica é usar `upsert` incremental com Change Data Feed.

**Decisão arquitetural: não usamos feature store nesta entrega.** A justificativa
parte da natureza do problema:

1. **Modelo univariado.** O LSTM aqui consome apenas a série temporal de `Close`.
   Não há features derivadas (médias móveis, volume normalizado, indicadores
   técnicos, dados macro) que precisem ser pré-computadas e compartilhadas entre
   modelos. Não existe "feature compartilhada" para hospedar.

2. **Janela curta e local ao request.** A inferência precisa apenas dos últimos
   60 preços. Esses dados chegam **no próprio request HTTP** (`POST /predict`
   com array de 60 floats), não vêm de um cache externo. Não há lookup remoto,
   logo não há janela vulnerável a flush.

3. **Custo > benefício para um único consumidor.** Feature stores se justificam
   quando múltiplos modelos compartilham as mesmas features (ex: detecção de
   fraude consome `customer_velocity_30d` que também alimenta o modelo de
   risco). Aqui temos um modelo, uma feature primária, um endpoint. Adicionar
   Redis introduz operação, monitoramento e ponto de falha sem ganho real.

**Como evoluiríamos se o escopo crescesse:**

- **Cenário A — adicionar features técnicas (RSI, MACD, Bollinger):**
  manteríamos cálculo on-the-fly dentro de `src/features/feature_engineering.py`,
  rodando no `predict_next_days`. Continua sem feature store.

- **Cenário B — múltiplos ativos servidos pela mesma API:** introduziríamos
  cache de features pré-computadas em Redis com:
  - **Padrão upsert incremental** (`HSET ticker:DIS:close <new_value>` +
    TTL de 7 dias), nunca `FLUSHALL`
  - **Change Data Feed** consumindo updates do banco transacional como source
    of truth
  - **Estratégia shadow ou canary** para atualizações de schema, garantindo
    que a versão antiga continua respondendo enquanto a nova sobe

- **Cenário C — features compartilhadas entre múltiplos modelos:** aí sim
  Feast ou Tecton justificariam o overhead, com offline store em Parquet/Delta
  e online store em Redis, ambos seguindo o mesmo padrão upsert.

**Resumo:** o anti-padrão do GAP 03 é resolvido aqui pela **ausência da
necessidade**, não pela negligência. Documentamos o caminho de evolução para
deixar explícito que a decisão é consciente.

## Decisões e trade-offs

### Por que LSTM e não algo mais simples?

Reaproveitamos o modelo da Fase 4 para focar a entrega da Fase 5 na plataforma — que é o critério avaliado pela banca, não a métrica do modelo.

### Por que MLflow e não Weights & Biases / Neptune?

MLflow é local-first, open source, sem dependência de SaaS, e tem Model Registry nativo. Permite rodar a stack inteira em `docker compose` sem credencial externa.

### Por que PSI manual em vez de só Evidently?

PSI implementado na mão dá controle total sobre os bins e thresholds, e é o índice padrão do mercado financeiro (que é o cenário do Datathon). Evidently fica disponível como plug-in para reports HTML.

### Por que não Airflow?

Para o escopo do Datathon, GitHub Actions com `cron` cobre orquestração de retreino sem overhead de subir scheduler dedicado. Se o caso evoluísse, migraríamos para Prefect (já está nas dependências).

## Como reagir a um drift em produção

Ver `docs/RUNBOOK.md`.
