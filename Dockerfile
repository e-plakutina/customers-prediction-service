# Сервис прогноза числа гостей: FastAPI (по умолчанию) и CLI predict.py.
# Двухэтапная сборка: зависимости ставятся uv из uv.lock в builder, в итоговый образ копируется только готовое окружение.
# Модель не обучается при сборке: используется обученная модель из models/.

# ---------- этап 1: окружение с зависимостями и пакетом проекта
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# сначала только зависимости: этот слой кэшируется, пока не меняется uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# затем пакет проекта (обычная, не editable установка в .venv)
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---------- этап 2: итоговый образ без uv, компиляторов и кэшей
FROM python:3.11-slim

# libgomp1 - OpenMP, без него LightGBM не загружается
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY models ./models
COPY data/processed ./data/processed
COPY predict.py ./

ENV PATH="/app/.venv/bin:$PATH" \
    PROJECT_ROOT=/app \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "customers_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
