.PHONY: help install lint test test-all train serve docker-up docker-down drift clean

PYTHON := python
PIP := pip

help:
	@echo "Datathon Fase 05 - comandos disponiveis:"
	@echo "  make install       - instala dependencias (dev incluso)"
	@echo "  make lint          - roda ruff + mypy + bandit"
	@echo "  make test          - testes unitarios (sem integracao)"
	@echo "  make test-all      - todos os testes incluindo integracao"
	@echo "  make train         - treina modelo e loga no MLflow"
	@echo "  make serve         - sobe API local na porta 8000"
	@echo "  make drift         - roda check de drift sobre dados de exemplo"
	@echo "  make docker-up     - sobe stack completa (API+MLflow+Prom+Graf)"
	@echo "  make docker-down   - derruba stack"
	@echo "  make clean         - remove caches e artefatos"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install || true

lint:
	ruff check src tests evaluation
	ruff format --check src tests evaluation
	mypy src --ignore-missing-imports || true
	bandit -r src -c pyproject.toml || true

format:
	ruff format src tests evaluation
	ruff check --fix src tests evaluation

test:
	pytest tests --cov=src --cov-report=term --cov-fail-under=60 -m "not integration"

test-all:
	pytest tests --cov=src --cov-report=term --cov-fail-under=60

train:
	$(PYTHON) -m src.models.train

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

drift:
	$(PYTHON) -c "from src.monitoring.drift import run_drift_check; \
		print(run_drift_check('data/raw/prices.parquet', 'data/raw/prices.parquet').to_dict())"

docker-up:
	docker compose up -d --build
	@echo ""
	@echo "Servicos:"
	@echo "  API:        http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000  (admin/admin)"

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml test-results.xml htmlcov/ build/ dist/ *.egg-info/
