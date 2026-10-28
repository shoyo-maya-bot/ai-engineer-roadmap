# 12-2 エージェントの API（roadmap_examples.p12_api）を動かすイメージ.
# examples/ で: docker build -t roadmap-agent .
FROM python:3.12-slim

# uv は版を固定して、公式のイメージから写す（https://docs.astral.sh/uv/guides/integration/docker/ ）
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1
WORKDIR /app

# 依存だけを先に入れる（コードだけを直したときは、この層を作り直さない）
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# コードと題材（設計書）を入れて、プロジェクトを入れる
COPY README.md ./
COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

# root では動かさない. 記録と承認待ちは /app/var（ボリュームをつなぐ）
RUN useradd --create-home agent && mkdir -p /app/var && chown agent /app/var
USER agent
ENV AGENT_DATA_DIR=/app/var
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["/app/.venv/bin/python", "-m", "roadmap_examples.p12_api", "--host", "0.0.0.0", "--port", "8000"]
