FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=tensorflow \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e ".[dev]" --no-build-isolation || pip install -e .

RUN mkdir -p ./artifacts

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]