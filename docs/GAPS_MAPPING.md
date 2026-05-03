# Mapeamento dos Gaps MLOps — Datathon Fase 05

> Documento canônico que mapeia cada gap descrito no PDF do Datathon para o componente concreto que o resolve no projeto. Use este documento na defesa para apontar evidência direta.

---

## Como ler este documento

Cada gap segue a estrutura:

- **Anti-padrão** — comportamento problemático descrito no PDF
- **Evidência real** — caso citado pelo professor (quando aplicável)
- **Nossa solução** — mecanismo concreto que ataca o gap
- **Arquivos** — onde está implementado
- **Como verificar** — comando ou tela para validar funcionamento

---

## GAP 01 — Ausência de Monitoramento de Modelos

**Anti-padrão (PDF p. 4):** modelo é deployado e nunca mais verificado. Ninguém sabe se está performando bem ou mal.

**Por que importa:** entregar modelo sem monitoramento é o equivalente a entregar carro sem painel.

**Nossa solução:**

- Prometheus coleta métricas da API a cada 15s via scrape em `/metrics`
- Grafana exibe dashboards em tempo real:
  - Latência p50 e p95
  - Throughput por endpoint
  - Distribuição de valores preditos (detector precoce de drift de saída)
  - Total de predições e erros acumulados
- Alertas automáticos por degradação (configuráveis em `configs/grafana/alerts/`)

**Arquivos:**

- `src/serving/app.py` — instrumentação com `prometheus-fastapi-instrumentator`
- `docker-compose.yml` — serviços `prometheus` e `grafana`
- `configs/prometheus/prometheus.yml` — configuração de scrape
- `configs/grafana/dashboards/lstm-overview.json` — dashboard principal

**Como verificar:**

```bash
# Confirma que /metrics está exposto
curl http://localhost:8000/metrics | head

# Vê targets do Prometheus
open http://localhost:9090/targets

# Acessa dashboard
open http://localhost:3000
```

---

## GAP 02 — Notebook Compartilhado como SPOF

**Anti-padrão (PDF p. 4):** um único notebook é o trigger de produção para múltiplos modelos. Uma edição quebra tudo.

**Evidência real (PDF):** Em instituição financeira, um cientista editou o notebook compartilhado. O modelo dele rodou. Todos os outros pararam. O MLE recebeu 30+ alertas no final de semana.

**Nossa solução:**

- **Zero notebooks em produção.** Pipeline de treino é módulo Python executável (`python -m src.models.train`)
- Configuração externalizada em YAML (`configs/model_config.yaml`) — não há valores hardcoded
- Cada serviço em container isolado — falha de um não derruba os outros
- CI bloqueia merge sem aprovação — cientista não tem acesso direto à main
- Versionamento Git completo, com rastreabilidade via `git_sha`

**Arquivos:**

- `src/models/train.py` — pipeline modular
- `configs/model_config.yaml` — configuração externalizada
- `docker-compose.yml` — isolamento de serviços
- `.github/workflows/ci.yml` — gates de qualidade

**Como verificar:**

```bash
# Pipeline reproduzível por linha de comando
docker exec -it datathon-api python -m src.models.train

# Cada serviço falha isoladamente (teste:
# derruba o MLflow e a API continua respondendo predições do alias atual)
docker compose stop mlflow
curl http://localhost:8000/health   # ainda responde 200
```

---

## GAP 03 — Feature Store com Padrão Destrutivo

**Anti-padrão (PDF p. 5):** Feature store com estratégia "deleta tudo, carrega tudo, sempre". Durante a janela de flush, o store está vazio.

**Evidência real (PDF):** Em sistema de detecção de fraude, a cada 10-55 minutos todo o cache era deletado e recarregado. Na janela vazia, transações eram processadas sem features — decisões erradas sistematicamente.

**Nossa solução:** **decisão arquitetural de não usar feature store**, justificada por:

1. Modelo univariado (uma feature: `Close`)
2. Janela curta e local ao request (60 preços vêm no payload HTTP)
3. Custo > benefício para um único consumidor

Cenários de evolução documentados em [`ARCHITECTURE.md`](./ARCHITECTURE.md#gap-03-em-detalhe--por-que-não-há-feature-store-nesta-arquitetura).

**Arquivos:**

- `docs/ARCHITECTURE.md` — seção dedicada com 3 cenários de evolução
- `src/features/feature_engineering.py` — features calculadas on-the-fly

---

## GAP 04 — Cobertura de Testes Próxima a Zero

**Anti-padrão (PDF p. 5):** Quality gate configurado em thresholds triviais (1% coverage). Equipe burla SonarQube excluindo pastas.

**Nossa solução:**

- `pytest` com `--cov-fail-under=60` (CI bloqueia merge se cobertura cair)
- Testes de schema com `pandera`
- Testes de feature engineering: shapes, nulls, ranges
- Testes de inferência: determinismo, range de predições
- Testes de integração com `FastAPI TestClient`

**Arquivos:**

- `tests/test_features.py` — schema e transformações
- `tests/test_drift.py` — cálculo de PSI
- `tests/test_api.py` — endpoints
- `pyproject.toml` — `[tool.pytest.ini_options]` com `--cov-fail-under=60`

**Como verificar:**

```bash
docker exec -it datathon-api pytest tests/ --cov=src -v
```

---

## GAP 05 — Sem Governança de Versionamento de Modelos

**Anti-padrão (PDF p. 6):** Cada cientista loga informações diferentes no MLflow — ou nada. Sem tags padronizadas, sem lineage, sem approval workflow.

**Evidência real (PDF):** Em uma plataforma com 20+ modelos, era impossível responder: "Qual versão está em produção? Com quais dados foi treinada? Quem aprovou?"

**Nossa solução:** schema obrigatório de tags em todo run, conforme página 6 do PDF:

```python
mlflow.set_tags({
    "model_name": ...,
    "model_type": "regression-timeseries-lstm",
    "framework": "tensorflow-keras",
    "owner": ...,
    "risk_level": "medium",
    "training_data_version": data_hash,   # SHA-256 do DataFrame de treino
    "git_sha": get_git_sha(),             # commit do código
    "phase": "datathon-fase05",
})
```

Mais Model Registry com aliases para rollback sem downtime.

**Arquivos:**

- `src/models/train.py` — bloco `mlflow.set_tags`
- `docs/MODEL_CARD.md` — schema documentado

**Como verificar:**

```bash
# Vê todas as tags no MLflow UI
open http://localhost:5000/#/experiments/1
# Clica em qualquer run → seção Tags
```

---

## GAP 06 — Sem Detecção de Drift

**Anti-padrão (PDF p. 7):** Modelo deployado e esquecido. Nenhuma verificação de data drift ou concept drift. Degradação acontece silenciosamente.

**Critério de aceite explícito (PDF p. 7):** "Detecção de drift implementada e documentada".

**Nossa solução:**

- Endpoint `/monitoring/drift_csv` calcula PSI entre janela atual e janela de referência (252 dias do treino)
- Thresholds configuráveis em `configs/monitoring_config.yaml`:
  - `PSI < 0.10`: estável
  - `0.10 ≤ PSI < 0.20`: warning
  - `PSI ≥ 0.20`: trigger de retreinamento
- Métricas de drift logadas no MLflow junto com métricas do modelo

**Arquivos:**

- `src/monitoring/drift.py` — implementação do PSI
- `src/serving/app.py` — endpoint `/monitoring/drift_csv`
- `configs/monitoring_config.yaml` — thresholds

**Como verificar:**

```bash
curl -X POST http://localhost:8000/monitoring/drift_csv \
  -H "X-API-Key: <KEY>" \
  -F "file=@dados.csv"
```

---

## GAP 07 — Ausência de Retraining Automatizado

**Anti-padrão (PDF p. 7):** Modelos retreinados manualmente, ad-hoc, sem trigger programado.

**Nossa solução:**

- Workflow GitHub Actions agendado semanal (`retrain.yml`)
- Workflow event-driven via drift (`drift-check.yml` chama `retrain.yml` se PSI > 0.20)
- Validação champion-challenger antes de promover (regra: `delta_mae <= -0.005`)
- Approval manual obrigatório no workflow (human-in-the-loop)

**Arquivos:**

- `.github/workflows/retrain.yml` — retreino agendado
- `.github/workflows/drift-check.yml` — retreino event-driven
- `docs/RUNBOOK.md` — procedimento manual de promoção

---

## GAP 08 — Ambiente de Desenvolvimento sem Dados

**Anti-padrão (PDF p. 8):** O ambiente de dev existe, mas não contém dados. Todo teste acontece em produção.

**Nossa solução:**

- DVC versiona dados (não commitar bytes brutos no Git)
- Fixtures pytest geram dados sintéticos para testes locais
- Script `make data` reproduz dataset completo localmente

**Arquivos:**

- `dvc.yaml` — pipeline DVC
- `tests/conftest.py` — fixtures
- `Makefile` — atalho `make data`

---

## GAP 09 — Skills Gap em Engenharia de Software

**Anti-padrão (PDF p. 8):** Cientistas sem fundamentos: sem testes, type hints, error handling, Git flow.

**Nossa solução:**

- Type hints em 100% das funções públicas
- Docstrings com Args e Returns (estilo Google)
- Logging estruturado via `logging` (zero `print` em produção)
- `pyproject.toml` com dependências fixadas
- `.pre-commit-config.yaml` com hooks de qualidade
- CI roda `ruff` (lint), `mypy` (tipos), `bandit` (segurança)

**Arquivos:**

- `pyproject.toml` — dependências e configuração de tooling
- `.pre-commit-config.yaml` — hooks locais
- `.github/workflows/ci.yml` — pipeline de qualidade

**Como verificar:**

```bash
# Lint
ruff check src tests

# Tipos
mypy src --ignore-missing-imports

# Segurança
bandit -r src -c pyproject.toml
```

---

## Resumo executivo

| Gap | Status | Localização principal |
|---|---|---|
| 01 — Monitoramento | ✅ Implementado | Prometheus + Grafana |
| 02 — SPOF | ✅ Implementado | Pipeline modular + containers isolados |
| 03 — Feature store | ✅ Documentado (não aplicável) | `ARCHITECTURE.md` |
| 04 — Testes | ✅ Implementado | `tests/` com cov 60% |
| 05 — Governança | ✅ Implementado | MLflow + tags obrigatórias |
| 06 — Drift | ✅ Implementado | PSI em `/monitoring/drift_csv` |
| 07 — Retraining | ✅ Implementado | GitHub Actions |
| 08 — Dev sem dados | ✅ Implementado | DVC + fixtures |
| 09 — Eng. soft | ✅ Implementado | Type hints + ruff + mypy + bandit |

**Maturidade MLOps atingida:** Nível 2 do Microsoft MLOps Maturity Model nas dimensões críticas de Experiment Management, Model Management, CI/CD e Monitoring.
