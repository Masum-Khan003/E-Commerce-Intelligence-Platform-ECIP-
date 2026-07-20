# Dockerfile — E-CIP v3.0 API + Celery workers
# Single image, different commands per service (see docker-compose.yml) —
# api, celery_gpu, celery_cpu, and celery_beat all share this image since
# they need the same dependency set (torch/xgboost/lightgbm/shap are all
# in the [api] extra already).

FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime XGBoost's compiled extension requires — missing
# from python:3.12-slim's minimal package set, and its absence doesn't fail
# the pip install; it only surfaces at model-load time as a cryptic
# "libgomp.so.1: cannot open shared object file" error.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies BEFORE copying source. pip install -e needs the
# packages declared in pyproject.toml's [tool.setuptools.packages.find] to
# exist to discover them — empty stub directories are enough. This way the
# expensive step (torch/transformers/xgboost, several GB) only re-runs when
# pyproject.toml changes, not on every source edit. Getting this backwards
# (source copied before pip install) meant every application code change
# forced a full ~15min dependency reinstall from scratch.
COPY pyproject.toml ./
RUN mkdir -p api models mlops data observability && \
    pip install --no-cache-dir -e ".[api]"

COPY api/ ./api/
COPY models/ ./models/
COPY mlops/ ./mlops/
COPY data/ ./data/
COPY observability/ ./observability/
COPY db/ ./db/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
