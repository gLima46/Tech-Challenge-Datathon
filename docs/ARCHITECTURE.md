# Arquitetura — Datathon Fase 05

> Documento de decisões arquiteturais e mapeamento dos componentes da plataforma.

## Sumário

- [Visão geral](#visão-geral)
- [Diagrama da stack](#diagrama-da-stack)
- [Mapeamento Gap → Componente](#mapeamento-gap--componente)
- [GAP 03 em detalhe — por que não há feature store](#gap-03-em-detalhe--por-que-não-há-feature-store-nesta-arquitetura)
- [Decisões e trade-offs](#decisões-e-trade-offs)
- [Como reagir a um drift em produção](#como-reagir-a-um-drift-em-produção)

---

## Visão geral

A plataforma é composta por quatro serviços containerizados, cada um com responsabilidade única e falha isolada (blast radius zero). A comunicação entre serviços é feita por rede interna do Docker Compose; o cliente externo só acessa os endpoints expostos.

| Serviço | Responsabilidade | Porta exposta |
|---|---|---|
| `datathon-api` | Serving FastAPI + endpoint `/metrics` | `8000` |
| `datathon-mlflow` | Tracking + Model Registry | `5000` |
| `datathon-prometheus` | Coleta de métricas via scrape | `9090` |
| `datathon-grafana` | Visualização e dashboards | `3000` |

---

## Diagrama da stack

```
┌──────────────┐    ┌─────────────────┐    ┌────────────────┐
│   yfinance   │───▶│  ingest (DVC)   │───▶│ data/raw/      │
└──────────────┘    └─────────────────┘    │ prices.parquet │
                                            └────────┬───────┘
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
                                          │  + alias prod    │
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
                  └──────────────────┘  │  scrape /metrics │  │  + reference set │
                                        └────────┬─────────┘  └─────────┬────────┘
                                                 ▼                      │
                                        ┌──────────────────┐            │
                                        │   Grafana        │◀───────────┘
                                        └──────────────────┘
```

Para um diagrama renderizado, ver `docs/diagrams/arquitetura.png`.

---

## Mapeamento Gap → Componente

| Gap (PDF) | Componente | Onde | Como |
|---|---|---|---|
| **01** — Sem monitoramento | Prometheus + Grafana | `src/serving/app.py`, `docker-compose.yml` | `prometheus-fastapi-instrumentator` expõe `/metrics`; Grafana consome com dashboards de latência, throughput e distribuição de predições |
| **02** — SPOF em notebook | Pipeline modular | `src/models/train.py`, `dvc.yaml` | Pipeline executável via CLI (`python -m src.models.train`), sem notebook como gatilho de produção |
| **03** — Feature store destrutivo | Decisão arquitetural | Ver seção dedicada abaixo | Não aplicável: modelo univariado, features chegam no request; cenários de evolução documentados |
| **04** — Sem testes | pytest + cov 60% | `tests/`, `pyproject.toml` | `test_features`, `test_drift`, `test_api`; CI bloqueia merge se cobertura < 60% |
| **05** — Sem governança de modelo | MLflow Registry | `src/models/train.py` | Schema obrigatório de tags + Model Registry com aliases para rollback sem downtime |
| **06** — Sem drift detection | PSI | `src/monitoring/drift.py` | Threshold 0.10 (warning) / 0.20 (retrain) conforme PDF p. 7 |
| **07** — Sem retraining auto | GitHub Actions | `.github/workflows/retrain.yml` | Retreino agendado semanal + retreino event-driven via drift |
| **08** — Dev sem dados | DVC + fixtures | `dvc.yaml`, `tests/conftest.py` | `dvc repro` reproduz pipeline; fixtures pytest geram dados sintéticos |
| **09** — Skill gap eng. soft | Type hints + ruff + mypy + bandit | `pyproject.toml`, `.pre-commit-config.yaml` | Lint, tipos e segurança obrigatórios no CI |

---

## GAP 03 em detalhe — por que não há feature store nesta arquitetura

O PDF do Datathon (p. 5) descreve o anti-padrão de feature store destrutivo: caches Redis ou tabelas que sofrem `FLUSHALL` periódico, criando janelas em que o serving fica sem features e o modelo retorna predições degradadas. A recomendação canônica é usar `upsert` incremental com Change Data Feed.

**Decisão arquitetural:** não usamos feature store nesta entrega. A justificativa parte da natureza do problema:

### 1. Modelo univariado

O LSTM aqui consome apenas a série temporal de `Close`. Não há features derivadas (médias móveis, volume normalizado, indicadores técnicos, dados macro) que precisem ser pré-computadas e compartilhadas entre modelos. Não existe "feature compartilhada" para hospedar.

### 2. Janela curta e local ao request

A inferência precisa apenas dos últimos 60 preços. Esses dados chegam no próprio request HTTP (`POST /predict` com array de 60 floats), não vêm de um cache externo. Não há lookup remoto, logo não há janela vulnerável a flush.

### 3. Custo > benefício para um único consumidor

Feature stores se justificam quando múltiplos modelos compartilham as mesmas features (ex: detecção de fraude consome `customer_velocity_30d` que também alimenta o modelo de risco). Aqui temos um modelo, uma feature primária, um endpoint. Adicionar Redis introduz operação, monitoramento e ponto de falha sem ganho real.

### Como evoluiríamos se o escopo crescesse

**Cenário A — adicionar features técnicas (RSI, MACD, Bollinger):** manteríamos cálculo on-the-fly dentro de `src/features/feature_engineering.py`, rodando no `predict_next_days`. Continua sem feature store.

**Cenário B — múltiplos ativos servidos pela mesma API:** introduziríamos cache de features pré-computadas em Redis com:

- Padrão upsert incremental (`HSET ticker:DIS:close <new_value>` + TTL de 7 dias), nunca `FLUSHALL`
- Change Data Feed consumindo updates do banco transacional como source of truth
- Estratégia shadow ou canary para atualizações de schema, garantindo que a versão antiga continua respondendo enquanto a nova sobe

**Cenário C — features compartilhadas entre múltiplos modelos:** aí sim Feast ou Tecton justificariam o overhead, com offline store em Parquet/Delta e online store em Redis, ambos seguindo o mesmo padrão upsert.

**Resumo:** o anti-padrão do GAP 03 é resolvido aqui pela ausência da necessidade, não pela negligência. Documentamos o caminho de evolução para deixar explícito que a decisão é consciente.

---

## Decisões e trade-offs

### Por que LSTM e não algo mais simples?

Reaproveitamos o modelo da Fase 4 para focar a entrega da Fase 5 na plataforma — que é o critério avaliado pela banca, não a métrica do modelo. O professor foi explícito na apresentação do Datathon: "pouco me importo com seu modelo, eu quero saber se vocês conseguem montar uma plataforma de engenharia de ML".

### Por que MLflow e não Weights & Biases / Neptune?

MLflow é local-first, open source, sem dependência de SaaS, e tem Model Registry nativo. Permite rodar a stack inteira em `docker compose` sem credencial externa. Além disso, é exigência explícita do PDF do Datathon (única biblioteca obrigatória).

### Por que MLflow 2.14.1 fixado e não a versão mais recente?

O servidor MLflow oficial (`ghcr.io/mlflow/mlflow:v2.14.1`) tem que ser compatível com o cliente. Versões mais novas do cliente chamam endpoints (`/api/2.0/mlflow/logged-models`) que não existem no servidor 2.14.1, gerando 404. Fixar `mlflow==2.14.1` no `pyproject.toml` evita drift de compatibilidade.

### Por que aliases do Model Registry e não stages?

Stages (`Production`, `Staging`, `Archived`) foram **deprecados** no MLflow 2.9+. A API moderna usa **aliases mutáveis** (`@production`, `@challenger`), que oferecem o mesmo comportamento sem o limite rígido de 4 estágios. A API lê `models:/lstm-price-forecaster@production` — para rollback, basta repontar o alias para uma versão anterior, sem deploy nem restart.

### Por que PSI manual em vez de só Evidently?

PSI implementado na mão dá controle total sobre os bins e thresholds, e é o índice padrão do mercado financeiro (que é o cenário do Datathon). Implementação direta tem três vantagens: controle total sobre o cálculo, código auditável em 30 linhas, e remove uma dependência pesada com 50+ pacotes transitivos. Evidently fica disponível como plug-in para reports HTML sob demanda.

### Por que não Airflow?

Para o escopo do Datathon, GitHub Actions com `cron` cobre orquestração de retreino sem overhead de subir scheduler dedicado. Se o caso evoluísse, migraríamos para Prefect (mais leve que Airflow para pipelines Python).

### Por que volume `mlflow-data` compartilhado entre `api` e `mlflow`?

O cliente MLflow (rodando no `api`) precisa escrever artefatos no mesmo storage que o servidor (`mlflow`) lê. Sem o volume compartilhado, o cliente tentaria gravar em `/mlflow/artifacts` localmente, falhando porque esse path só existe no container do servidor. Compartilhar o volume nomeado garante consistência sem precisar do modo `--serve-artifacts` (que adiciona overhead HTTP).

---

## Como reagir a um drift em produção

Ver [`docs/RUNBOOK.md`](./RUNBOOK.md).
