FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    zlib1g-dev libbz2-dev liblzma-dev libboost-all-dev \
    libgl1 libglib2.0-0 libegl1 libgles2 \
    libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --frozen --no-cache
COPY inputs inputs
COPY src src

EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "src/interface.py", "--server.port=8080"]