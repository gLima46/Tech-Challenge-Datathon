# Datathon Fase 05 — Plataforma MLOps de Previsão de Série Temporal Financeira

Plataforma completa de engenharia de machine learning construída para o Datathon da FIAP — Pós-Tech MLE, Fase 05. O projeto demonstra maturidade Nível 2 do [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model) nas dimensões críticas de Experiment Management, Model Management, CI/CD e Monitoring.

> **Foco do projeto:** plataforma operacional, não otimização de modelo. O LSTM é veículo para demonstrar rastreabilidade, isolamento, observabilidade e governança em escala de produção.

---

## 📋 Índice

- [Stack Tecnológico](#-stack-tecnológico)
- [Arquitetura](#-arquitetura)
- [Pré-requisitos](#-pré-requisitos)
- [Quick Start](#-quick-start)
- [Treinando o Modelo](#-treinando-o-modelo)
- [Acessando os Serviços](#-acessando-os-serviços)
- [Testando a API](#-testando-a-api)
- [Observabilidade](#-observabilidade)
- [Detecção de Drift](#-detecção-de-drift)
- [Testes e CI/CD](#-testes-e-cicd)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Mapeamento dos Gaps MLOps](#-mapeamento-dos-gaps-mlops)
- [Documentação Adicional](#-documentação-adicional)
- [Troubleshooting](#-troubleshooting)
- [Equipe](#-equipe)

---

## 🛠️ Stack Tecnológico

| Camada | Tecnologia |
|---|---|
| **Modelo** | TensorFlow 2.16 / Keras 3 (LSTM) |
| **Tracking & Registry** | MLflow 2.14.1 |
| **API Serving** | FastAPI + Uvicorn |
| **Validação de Schema** | Pandera |
| **Observabilidade** | Prometheus + Grafana |
| **Orquestração** | Docker Compose |
| **CI/CD** | GitHub Actions (ruff, mypy, bandit, pytest) |
| **Linguagem** | Python 3.11 |

---

## 🏗️ Arquitetura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   datathon-api  │────▶│ datathon-mlflow  │     │ datathon-promet │
│   (FastAPI)     │     │  (Tracking +     │◀────│    heus         │
│   :8000         │     │   Registry)      │     │   :9090         │
└────────┬────────┘     │   :5000          │     └────────┬────────┘
         │              └──────────────────┘              │
         │                                                │
         │ /metrics                                       │ scrape
         └────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ datathon-grafana│
                                                 │   (Dashboards)  │
                                                 │     :3000       │
                                                 └─────────────────┘
```

**Princípios arquiteturais:**

- **Isolamento de containers:** cada serviço é independente. Falha de um não afeta os outros (blast radius zero).
- **Volume compartilhado `mlflow-data`:** garante consistência de artefatos entre o cliente (api) e o servidor (mlflow).
- **Configuração externalizada:** toda configuração via variáveis de ambiente e YAMLs (`configs/`).
- **Modelo agnóstico:** a API lê o modelo via alias do Model Registry, não por path. Rollback sem deploy.

Para decisões arquiteturais detalhadas (incluindo a justificativa do GAP 03), ver [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

---

## ✅ Pré-requisitos

- Docker Desktop 20.10+ ou Docker Engine + Compose Plugin
- 4 GB de RAM disponíveis para os containers
- Portas livres: `5000`, `8000`, `9090`, `3000`
- Conexão com internet (download de dados via `yfinance`)

---

## 🚀 Quick Start

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/Tech-Challenge-Datathon.git
cd Tech-Challenge-Datathon

# 2. Suba toda a stack (build + start)
docker compose up -d --build

# 3. Aguarde ~30 segundos para os serviços ficarem prontos
docker compose ps

# 4. Treine o primeiro modelo
docker exec -it datathon-api python -m src.models.train

# 5. Pronto! Acesse os serviços nos endereços abaixo
```

**Para resetar tudo do zero** (apaga banco do MLflow, métricas, dashboards):

```bash
docker compose down -v
docker compose up -d --build
```

> ⚠️ O `-v` remove os volumes persistentes. Use apenas para reset completo.

---

## 🧠 Treinando o Modelo

O pipeline de treino baixa dados via `yfinance`, valida schema com pandera, treina o LSTM, registra tudo no MLflow e promove o modelo para o Model Registry.

```bash
docker exec -it datathon-api python -m src.models.train
```

**O que acontece em cada execução:**

1. Carrega configuração de `configs/model_config.yaml`
2. Fixa seeds para reprodutibilidade
3. Baixa dados históricos do ativo configurado (default: DIS)
4. Valida schema com pandera (quality gate)
5. Calcula `data_hash` (SHA256 do DataFrame) e `git_sha` (commit atual)
6. Treina o modelo LSTM (50 épocas com early stopping)
7. Registra no MLflow:
   - Tags obrigatórias (model_name, model_type, framework, owner, risk_level, training_data_version, git_sha, phase)
   - Hiperparâmetros completos
   - Métricas no espaço original (MAE, RMSE, MAPE)
   - Artefatos: modelo `.keras`, scaler `.pkl`, janela de referência `.parquet`
8. Promove para o Model Registry com versão incremental

**Saída esperada:**

```
2026-05-XX XX:XX:XX,XXX - __main__ - INFO - Métricas finais: {'mae': 2.70, 'rmse': 3.73, 'mape': 2.31}
2026-05-XX XX:XX:XX,XXX - __main__ - INFO - Modelo registrado no Model Registry
2026-05-XX XX:XX:XX,XXX - __main__ - INFO - Run concluído: <run_id>
```

Para procedimentos operacionais (promoção, rollback, retreino), ver [`docs/RUNBOOK.md`](./docs/RUNBOOK.md).

---

## 🌐 Acessando os Serviços

| Serviço | URL | Credenciais |
|---|---|---|
| **API FastAPI** (Swagger) | http://localhost:8000/docs | API Key necessária (gerar via endpoint) |
| **MLflow UI** | http://localhost:5000 | — |
| **Prometheus** | http://localhost:9090 | — |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |

---

## 🔌 Testando a API

### 1. Gerar uma API Key

```bash
curl -X POST http://localhost:8000/generate_api_key \
  -H "Content-Type: application/json" \
  -d '{"initial_key": "dev-key-trocar-em-prod"}'
```

Resposta:
```json
{"api_key": "sk-XXXXXXXXXXXXXXXXXXXX"}
```

### 2. Fazer uma Predição

```bash
curl -X POST http://localhost:8000/predict_csv \
  -H "X-API-Key: sk-XXXXXXXXXXXXXXXXXXXX" \
  -H "Content-Type: application/json" \
  -d '{"steps": 10}'
```

### 3. Popular o Grafana com Dados (loop de testes)

Para ver os gráficos do Grafana ganharem vida, faça várias chamadas:

**Linux / Mac:**

```bash
for i in {1..20}; do
  curl -X POST http://localhost:8000/predict_csv \
    -H "X-API-Key: sk-XXXXXXXXXXXXXXXXXXXX" \
    -H "Content-Type: application/json" \
    -d '{"steps": 5}'
  sleep 1
done
```

**Windows PowerShell:**

```powershell
1..20 | ForEach-Object {
  curl -X POST http://localhost:8000/predict_csv `
    -H "X-API-Key: sk-XXXXXXXXXXXXXXXXXXXX" `
    -H "Content-Type: application/json" `
    -d '{"steps": 5}'
  Start-Sleep -Seconds 1
}
```

Após ~30 segundos, o Prometheus vai ter coletado as métricas e o Grafana vai exibir os dashboards populados.

---

## 📊 Observabilidade

### Prometheus (`http://localhost:9090`)

O Prometheus coleta métricas da API a cada 15 segundos via scrape no endpoint `/metrics`. Acesse a aba **Status → Targets** para confirmar que ambos os targets (`datathon-api` e `prometheus`) estão **UP**.

**Queries úteis** (aba **Graph**):

```promql
# Todas as métricas da API de uma vez
{job="datathon-api"}

# Total de requisições por endpoint
http_requests_total

# Contador de duração das requisições
http_request_duration_seconds_count

# Taxa de requisições por segundo (últimos 5 min)
rate(http_requests_total[5m])

# Latência p95 dos últimos 5 min
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Histograma da distribuição de valores preditos (drift de saída)
lstm_prediction_value_bucket

# Total acumulado de predições servidas
lstm_predictions_total

# Apenas requisições com erro
http_requests_total{status=~"5.."}
```

### Grafana (`http://localhost:3000`)

Login: `admin` / `admin`

Dashboard principal exibe:

- **Latência das predições (p50 e p95)** — SLA da API em tempo real
- **Throughput (req/s por endpoint)** — padrão de uso
- **Error rate (últimos 5 min)** — alarme de falhas
- **Distribuição das predições** — detector precoce de drift de saída
- **Total de predições / Total de erros** — contadores acumulados

Se o dashboard não aparecer automaticamente, vá em **Connections → Data Sources → Add → Prometheus** e configure URL como `http://prometheus:9090`.

---

## 🌊 Detecção de Drift

O endpoint `/monitoring/drift_csv` calcula PSI (Population Stability Index) entre as predições recentes e a janela de referência de 252 dias salva como artefato no MLflow durante o treino.

**Thresholds** (configuráveis em `configs/monitoring_config.yaml`):

- `PSI < 0.1`: Estável (sem drift)
- `0.1 ≤ PSI < 0.2`: Warning (monitorar)
- `PSI ≥ 0.2`: Trigger de retreinamento

**Exemplo de chamada:**

```bash
curl -X POST http://localhost:8000/monitoring/drift_csv \
  -H "X-API-Key: sk-XXXXXXXXXXXXXXXXXXXX" \
  -F "file=@dados_recentes.csv"
```

Para resposta a alertas de drift, ver [`docs/RUNBOOK.md`](./docs/RUNBOOK.md#4-reagir-a-alerta-de-drift).

---

## 🧪 Testes e CI/CD

### Rodando os testes localmente

```bash
docker exec -it datathon-api pytest tests/ --cov=src -v
```

Cobertura mínima configurada em **60%** (falha o build se não atingir).

**O que está coberto:**

- Validação de schema com pandera
- Feature engineering (shapes, nulls, range)
- Endpoints da API (FastAPI TestClient)
- Pipeline de treino (mocked)

### CI/CD no GitHub Actions

Pipeline executado automaticamente em cada push e pull request:

1. **Lint** — `ruff check src tests`
2. **Type check** — `mypy src --ignore-missing-imports`
3. **Security scan** — `bandit -r src -c pyproject.toml`
4. **Unit tests** — `pytest tests/ --cov=src --cov-fail-under=60`
5. **Docker build** — valida que a imagem builda corretamente

Se qualquer etapa falhar, o build não passa e o PR fica bloqueado.

---

## 📁 Estrutura do Repositório

```
Tech-Challenge-Datathon/
├── .github/
│   └── workflows/
│       └── ci.yml              # Pipeline de CI
├── configs/
│   ├── model_config.yaml       # Hiperparâmetros do modelo
│   ├── monitoring_config.yaml  # Thresholds de drift
│   ├── prometheus/             # Config do Prometheus
│   └── grafana/                # Dashboards do Grafana
├── docs/
│   ├── README.md               # Índice da documentação
│   ├── ARCHITECTURE.md         # Decisões arquiteturais
│   ├── MODEL_CARD.md           # Model Card (Mitchell et al. 2019)
│   ├── RUNBOOK.md              # Procedimentos operacionais
│   ├── GAPS_MAPPING.md         # Mapeamento dos 9 gaps do PDF
│   ├── diagrams/               # Diagramas de arquitetura
│   └── screenshots/            # Evidências visuais
├── src/
│   ├── features/
│   │   ├── ingestion.py        # Download de dados (yfinance)
│   │   └── feature_engineering.py  # Schema + transformações
│   ├── models/
│   │   └── train.py            # Pipeline de treino com MLflow
│   ├── serving/
│   │   ├── app.py              # Endpoints FastAPI
│   │   └── Dockerfile          # Container de serving
│   └── monitoring/
│       ├── drift.py            # Cálculo de PSI
│       └── metrics.py          # Métricas customizadas Prometheus
├── tests/
│   ├── conftest.py             # Fixtures
│   ├── test_features.py        # Testes de schema
│   ├── test_models.py          # Testes de treino
│   └── test_api.py             # Testes de endpoints
├── docker-compose.yml          # Orquestração da stack
├── pyproject.toml              # Dependências e configuração
└── README.md                   # Este arquivo
```

---

## 🎯 Mapeamento dos Gaps MLOps

Cada gap do PDF do Datathon é atacado com mecanismo concreto:

| Gap | Anti-padrão | Nossa Solução |
|---|---|---|
| **GAP 01** — Sem monitoramento | Modelo deployado e esquecido | Prometheus + Grafana com latência, throughput, distribuição de predições |
| **GAP 02** — Notebook como SPOF | Edição quebra tudo | Pipeline modular em `src/models/train.py`, executável via CLI |
| **GAP 03** — Feature store destrutivo | Cache flush deixa vazio | Não aplicável: modelo univariado, features chegam no request. Justificativa em [`ARCHITECTURE.md`](./docs/ARCHITECTURE.md#gap-03-em-detalhe--por-que-não-há-feature-store-nesta-arquitetura) |
| **GAP 04** — Sem testes | `--cov 1%` | `--cov-fail-under=60` no CI, testes de schema, features e API |
| **GAP 05** — Sem governança de modelo | Tags inconsistentes | Schema obrigatório de tags + Model Registry com aliases |
| **GAP 06** — Sem drift detection | Degradação silenciosa | Endpoint `/monitoring/drift_csv` com PSI configurável |
| **GAP 07** — Sem retraining automatizado | Manual ad-hoc | Trigger via PSI > 0.2 (arquitetura preparada) |
| **GAP 08** — Dev sem dados | Testes em produção | Fixtures pytest + dados sintéticos |
| **GAP 09** — Skills gap | Sem type hints, sem docstrings | Type hints em 100% das funções, docstrings, logging estruturado |

Para detalhamento de cada gap (anti-padrão, evidência real, solução, arquivos, comando de verificação), ver [`docs/GAPS_MAPPING.md`](./docs/GAPS_MAPPING.md).

---

## 📚 Documentação Adicional

A documentação completa está em [`docs/`](./docs/):

| Documento | Conteúdo |
|---|---|
| [`docs/README.md`](./docs/README.md) | Índice navegável da documentação com caminhos rápidos por persona |
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | Decisões arquiteturais e justificativas técnicas |
| [`docs/MODEL_CARD.md`](./docs/MODEL_CARD.md) | Especificação do modelo: dados, métricas, limitações |
| [`docs/RUNBOOK.md`](./docs/RUNBOOK.md) | Procedimentos operacionais (treinar, promover, rollback, drift) |
| [`docs/GAPS_MAPPING.md`](./docs/GAPS_MAPPING.md) | Mapeamento explícito dos 9 gaps do PDF |

---

## 🔧 Troubleshooting

### Os containers não sobem

```bash
# Verifica logs do serviço problemático
docker compose logs api
docker compose logs mlflow

# Reseta tudo
docker compose down -v
docker compose up -d --build
```

### `git_sha: unknown` no MLflow

O container não tem acesso ao `.git`. Adicione no `docker-compose.yml`, no serviço `api`:

```yaml
    volumes:
      - ./.git:/app/.git:ro
```

Depois: `docker compose up -d`

### MLflow retorna 404 em `/api/2.0/mlflow/logged-models`

Versão do cliente MLflow incompatível com servidor 2.14.1. Confirme que `pyproject.toml` tem `mlflow==2.14.1` (não `>=2.14`).

### Prometheus mostra target `datathon-api` como DOWN

A API ainda não terminou de subir. Aguarde ~30 segundos e atualize a página de Targets.

### Grafana abre mas não tem dashboards

Vá em **Connections → Data Sources → Add → Prometheus**, configure URL `http://prometheus:9090`, salve. Depois importe o dashboard de `configs/grafana/dashboards/`.

### Erro de DNS no container ao baixar dados

Adicione DNS público no `docker-compose.yml`:

```yaml
    dns:
      - 8.8.8.8
      - 8.8.4.4
```

### Erro `OSError: Read-only file system` ao salvar modelo

O `model.save()` está tentando escrever em diretório sem permissão. A solução é salvar via `tempfile` e logar como artefato MLflow:

```python
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = os.path.join(tmp, "modelo.keras")
    model.save(tmp_path)
    mlflow.log_artifact(tmp_path, artifact_path="model")
```

### Bandit `B404` reclamando do `import subprocess`

Adicione `# nosec B404` na linha do import com justificativa documentada:

```python
import subprocess  # nosec B404 - usado apenas para git rev-parse, comando hardcoded
```

Para troubleshooting expandido, ver [`docs/RUNBOOK.md`](./docs/RUNBOOK.md#6-diagnóstico-de-problemas-comuns).

---

## 📚 Referências

- [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- [Google — Practitioners Guide to MLOps](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf)
- [Mitchell et al. (2019) — Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)

---

## 📄 Licença

Projeto acadêmico — uso educacional.