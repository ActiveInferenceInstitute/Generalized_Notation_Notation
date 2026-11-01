# Dockerfile — linux/amd64 test image
FROM --platform=$BUILDPLATFORM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates python3-dev gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Install uv — pinned version for reproducible builds (supply-chain hardening).
# Pin matches exact uv 0.13.6 in CI and uv.toml. The default runtime uses the
# documented h5py/HDF5 profile exception; this image needs separate qualification.
ARG UV_VERSION=0.13.6
RUN test "$UV_VERSION" = "0.13.6" && \
    curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
      https://github.com/astral-sh/uv/releases/download/0.13.6/uv-installer.sh \
      --output /tmp/uv-installer.sh && \
    echo 'f7a56e7afaec49a3bcbc3ac3cdbe917e7c6ce30e0d815308a12e531aa5ef23b7  /tmp/uv-installer.sh' | sha256sum --check - && \
    UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh /tmp/uv-installer.sh && \
    /usr/local/bin/uv --version | awk '{print $2}' | grep -Fx 0.13.6 && \
    rm /tmp/uv-installer.sh

# Create non-root user for runtime security
RUN groupadd --gid 1000 gnn && \
    useradd --uid 1000 --gid gnn --create-home gnn

WORKDIR /workspace

# Copy project (owned by root for build layer caching); .dockerignore keeps
# host-side .venv/__pycache__/output/ artifacts out of the build context.
COPY . /workspace

# Create venv and install deps via uv (as root for build performance).
# Frozen dev dependencies match the ordinary CI profile, including core JAX.
# Optional/native and pipeline contracts have separate hosted lanes.
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --frozen --extra dev
RUN chown -R gnn:gnn /workspace /opt/venv
USER gnn

# Run the same ordinary marker selection as ci.yml with the installed runtime.
# Native/container qualification in installed-platforms.yml has its own scope.
CMD ["uv","run","--frozen","--no-sync","python","-m","pytest","tests/","-n","auto","--dist","worksteal","-m","not pipeline and not mcp and not ollama and not env_heavy and not toolchain","--cov=gnn","--cov-report=term-missing","--cov-report=html:output/2_tests_output/htmlcov","--tb=short","-q"]
