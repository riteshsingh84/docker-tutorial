FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

WORKDIR /app

ENV HF_HOME=/models \
    HF_HUB_DISABLE_SSL_VERIFICATION=1 \
    HF_HUB_DISABLE_XET=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_CERTS=true \
    UV_INSECURE_HOST="pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org download-r2.pytorch.org" \
    PATH="/app/.venv/bin:$PATH"

COPY download_model.py ./

RUN uv pip install --system huggingface_hub && \
    python download_model.py

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
