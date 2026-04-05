FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY main.py ./

EXPOSE 8000

# Checkpoints and dataset are mounted at runtime, not baked into the image.
# Run with:
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     retinal-classifier
CMD ["uv", "run", "--no-dev", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
