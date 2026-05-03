# Runbook Operacional

> Procedimentos para manutenção e resposta a incidentes da plataforma. Atualizado para MLflow 2.14+ (API de aliases).

## Sumário

1. [Treinar uma nova versão](#1-treinar-uma-nova-versão)
2. [Promover modelo para produção](#2-promover-modelo-para-produção)
3. [Rollback de versão problemática](#3-rollback-de-versão-problemática)
4. [Reagir a alerta de drift](#4-reagir-a-alerta-de-drift)
5. [Subir e derrubar a stack local](#5-subir-e-derrubar-a-stack-local)
6. [Diagnóstico de problemas comuns](#6-diagnóstico-de-problemas-comuns)

---

## 1. Treinar uma nova versão

```bash
docker exec -it datathon-api python -m src.models.train
```

O script:

1. Baixa preços via `yfinance`.
2. Valida schema com Pandera.
3. Cria sequências e treina LSTM.
4. Loga params, métricas, tags e artefatos no MLflow.
5. Registra a nova versão no Model Registry (sem alias atribuído).

Para customizar parâmetros, edite `configs/model_config.yaml` antes:

```yaml
data:
  symbol: DIS              # Ticker
  start_date: "2014-01-01"
  end_date: "2024-12-31"

model:
  lstm_units: [64, 64]
  dropout: 0.2

training:
  epochs: 50
  batch_size: 32
  early_stopping_patience: 10
```

---

## 2. Promover modelo para produção

A API serve o modelo cujo alias é `production`. Para promover uma versão:

### Via MLflow UI

1. Acesse `http://localhost:5000/#/models/lstm-price-forecaster`
2. Clique na versão desejada
3. Clique em **Aliases** → adicione alias `production`
4. Reinicie a API:

```bash
docker compose restart api
```

### Via Python

```bash
docker exec -it datathon-api python -c "
from mlflow import MlflowClient
client = MlflowClient(tracking_uri='http://mlflow:5000')
client.set_registered_model_alias(
    name='lstm-price-forecaster',
    alias='production',
    version='<N>'
)
print('Alias production apontando para versão <N>')
"

docker compose restart api
```

> ⚠️ **Não use `mlflow models update --stage Production`** — stages foram deprecados no MLflow 2.9+. A API moderna usa aliases mutáveis, que não têm o limite rígido de 4 estágios.

---

## 3. Rollback de versão problemática

Cenário: a versão `5` foi promovida e está degradando. Voltar para a `4`.

```bash
docker exec -it datathon-api python -c "
from mlflow import MlflowClient
client = MlflowClient(tracking_uri='http://mlflow:5000')

# Repontar alias production para versão anterior
client.set_registered_model_alias(
    name='lstm-price-forecaster',
    alias='production',
    version='4'
)

# (Opcional) marcar a versão ruim com alias para investigação
client.set_registered_model_alias(
    name='lstm-price-forecaster',
    alias='quarantine',
    version='5'
)
print('Rollback concluído: production -> v4, quarantine -> v5')
"

docker compose restart api
```

### Reprodução determinística da versão antiga

Como cada versão tem `git_sha` e `training_data_version` registrados, dá para reproduzir bit-a-bit:

```bash
# 1. Identifica o git_sha da versão a reproduzir
docker exec -it datathon-api python -c "
from mlflow import MlflowClient
client = MlflowClient(tracking_uri='http://mlflow:5000')
mv = client.get_model_version('lstm-price-forecaster', '4')
run = client.get_run(mv.run_id)
print('git_sha:', run.data.tags['git_sha'])
print('data_hash:', run.data.tags['training_data_version'])
"

# 2. Volta o código para o commit registrado
git checkout <git_sha>

# 3. Restaura versão dos dados (se usando DVC)
dvc checkout

# 4. Treina de novo
docker exec -it datathon-api python -m src.models.train
```

---

## 4. Reagir a alerta de drift

Quando o endpoint `/monitoring/drift_csv` retorna `status="retrain"` (PSI > 0.20):

### Passo 1 — Confirmar o sinal

Rode manualmente com a janela mais recente:

```bash
curl -X POST http://localhost:8000/monitoring/drift_csv \
  -H "X-API-Key: <SUA_KEY>" \
  -F "file=@dados_recentes.csv"
```

Se PSI confirmar acima de 0.20, prossiga.

### Passo 2 — Inspecionar a janela suspeita

Abra o report do Evidently para entender qual feature drittou e como:

```bash
docker exec -it datathon-api python -m src.monitoring.evidently_report \
  --reference data/reference.parquet \
  --current data/current.parquet \
  --output reports/drift.html
```

### Passo 3 — Decidir

| Causa | Ação |
|---|---|
| Evento conhecido (split, anúncio relevante) | Ignorar pontualmente, agendar revisão em 1 semana |
| Drift estrutural (regime macro, mudança de mercado) | Treinar nova versão com janela atualizada |
| Bug em ingestão de dados | Corrigir pipeline de ingestão antes de retreinar |

### Passo 4 — Treinar challenger

```bash
# Atualiza end_date no config
# vim configs/model_config.yaml

docker exec -it datathon-api python -m src.models.train
```

### Passo 5 — Comparar champion vs challenger

No MLflow UI, abra ambos os runs lado a lado. Promover apenas se:

- `delta_mae <= -0.005` (challenger melhor por margem material)
- Latência igual ou menor
- Sem regressão em outras métricas

### Passo 6 — Promover (se aprovado)

Ver seção [2. Promover modelo para produção](#2-promover-modelo-para-produção).

---

## 5. Subir e derrubar a stack local

### Subir tudo (build + start)

```bash
docker compose up -d --build
```

### Derrubar mantendo dados (recomendado)

```bash
docker compose down
```

Volumes (`mlflow-data`, `prometheus-data`, `grafana-data`) persistem.

### Derrubar e apagar tudo (reset total)

```bash
docker compose down -v
```

> ⚠️ Apaga MLflow runs, dashboards do Grafana e métricas históricas do Prometheus. Use apenas para reset completo.

### Verificar saúde dos serviços

```bash
docker compose ps
docker compose logs -f api      # logs do serving
docker compose logs -f mlflow   # logs do tracking
```

### Endpoints

| Serviço | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (`admin` / `admin`) |

---

## 6. Diagnóstico de problemas comuns

### A API não sobe

```bash
docker compose logs api
```

Causas comuns:
- MLflow ainda não está pronto → aguardar 30s e reiniciar
- Modelo não promovido com alias `production` → ver seção 2
- Volume `mlflow-data` corrompido → `docker compose down -v && up -d`

### MLflow retorna 404 em endpoints

Verifique compatibilidade de versão:

```bash
docker exec -it datathon-api pip show mlflow | grep Version
```

Deve ser `2.14.1`. Se for diferente, corrija no `pyproject.toml` e rebuilde.

### Prometheus mostra target DOWN

```bash
# Verifica se a API está expondo /metrics
curl http://localhost:8000/metrics | head -20

# Se vazio, reinicia a API
docker compose restart api
```

### `git_sha: unknown` no MLflow

O container não tem acesso ao `.git`. Adicione no `docker-compose.yml`:

```yaml
  api:
    volumes:
      - ./.git:/app/.git:ro
```

### Grafana sem dashboards

```bash
# Verifica se o datasource está configurado
curl -u admin:admin http://localhost:3000/api/datasources

# Reimporta dashboards
docker compose restart grafana
```

### Treino falha por DNS no container

```bash
# Adicione DNS público no docker-compose.yml
api:
  dns:
    - 8.8.8.8
    - 8.8.4.4
```

---

## Procedimento de incidente — template

Para qualquer incidente de produção, registrar:

```markdown
## Incidente <DATA>

**Sintoma:** descrição do que o usuário/sistema reportou
**Detecção:** como foi detectado (alerta, ticket, observação)
**Causa raiz:** análise técnica
**Resolução:** ação tomada
**Versão envolvida:** modelo e git_sha
**Tempo de resolução:** TTR
**Ação preventiva:** o que mudar para evitar recorrência
```

Salvar em `docs/incidents/<YYYY-MM-DD>-<slug>.md`.
