# ===============================
# Builder stage
# ===============================
FROM python:3.13-slim AS build

# Environment settings
ENV POETRY_VERSION=2.3.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=/root/.local/bin:$PATH

# install dependencies
COPY pyproject.toml poetry.lock ./

# Install deps, then replace GPU torch with CPU-only version
RUN poetry install --only main --no-root \
    && .venv/bin/pip uninstall -y torch \
    && .venv/bin/pip install --no-cache-dir \
       torch==2.11.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf .venv/lib/python*/site-packages/nvidia* \
              .venv/lib/python*/site-packages/triton* \
              .venv/lib/python*/site-packages/cuda* \
    && rm -rf .venv/lib/python*/site-packages/networkx*


         # cleanup
RUN find .venv -name "__pycache__" -exec rm -rf {} + \
    && find .venv -name "*.pyc" -delete

# ===============================
# Runtime stage
# ===============================
FROM python:3.13-slim AS runner

WORKDIR /app

# curl for HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]
