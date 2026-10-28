# 12-2 エージェントの API（roadmap_examples.p12_api）を動かすイメージ.
# examples/ で: docker build -t roadmap-agent .
# 版はタグ（パッチの版と OS まで）と digest（中身の目印）で固定する. 更新するときは両方を書き換える
FROM python:3.12.16-slim-trixie@sha256:dacd1678a25ee255c22ea5de948374f5542db3562dd365038b14349051ea156c

# uv は公式のイメージから写す（https://docs.astral.sh/uv/guides/integration/docker/ ）. 版は extra.versions.uv と同じ（check_versions.py が照合）
COPY --from=ghcr.io/astral-sh/uv:0.8.17@sha256:e4644cb5bd56fdc2c5ea3ee0525d9d21eed1603bccd6a21f887a938be7e85be1 /uv /uvx /bin/
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
