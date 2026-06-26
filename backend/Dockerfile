FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependency layer (cached unless pyproject/src package metadata changes)
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system -e .

# Loaders + generated assets
COPY scripts ./scripts
COPY assets ./assets

EXPOSE 8000

CMD ["uvicorn", "financial_qa.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
