FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Dependencies first, so Docker can cache this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY predict.py model.bin ./

EXPOSE 9696

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]