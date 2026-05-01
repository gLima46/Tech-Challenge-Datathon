# Runbook Operacional

Procedimentos para manutenção e resposta a incidentes da plataforma.

## Índice
1. Treinar uma nova versão
2. Promover modelo para produção
3. Rollback de versão problemática
4. Reagir a alerta de drift
5. Subir e derrubar a stack local

---

## 1. Treinar uma nova versão

---
bash
make train
# ou: python -m src.models.train
---

O script:
1. Baixa preços via yfinance.
2. Valida schema com Pandera.
3. Cria sequências e treina LSTM.
4. Loga params, métricas, tags e artefatos no MLflow.
5. Registra a nova versão no Model Registry em stage `None`.

Para customizar parâmetros, edite `configs/model_config.yaml` antes.

## 2. Promover modelo para produção

Via UI do MLflow (`http://localhost:5000`) ou via CLI:

---
bash
mlflow models update \
  --name lstm-price-forecaster \
  --version <N> \
  --stage Production
---

A API carrega do stage `Production` no startup, então é necessário reiniciar:

---
bash
docker compose restart api
---

## 3. Rollback de versão problemática

---
bash
# Move a versao ruim para Archived
mlflow models update --name lstm-price-forecaster --version <N_RUIM> --stage Archived

# Promove a anterior (ou outra escolhida) de volta para Production
mlflow models update --name lstm-price-forecaster --version <N_BOM> --stage Production

docker compose restart api
---

Como cada versão tem `git_sha` e `training_data_version` registrados, dá para reproduzir bit-a-bit:

---
bash
git checkout <git_sha>
dvc checkout
make train
---

## 4. Reagir a alerta de drift

Quando o `drift.py` retorna `status="retrain"`:

1. Confirmar: rode manualmente `make drift` para validar.
2. Inspecionar: abra o report do Evidently sobre a janela suspeita.
3. Decidir:
   - Se for evento conhecido (split, anúncio relevante) → ignorar pontualmente, agendar revisão.
   - Se for drift estrutural → treinar nova versão com janela atualizada.
4. Treinar challenger: `make train` com `end_date` movida.
5. Comparar champion vs challenger nas métricas registradas no MLflow.
6. Promover apenas se `delta_mae <= -0.005` (challenger melhor por margem material).

## 5. Subir e derrubar a stack local

---
bash
make docker-up      # API + MLflow + Prometheus + Grafana
make docker-down    # derruba mantendo volumes
docker compose down -v   # apaga volumes (perde MLflow runs!)
---

Endpoints:
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
