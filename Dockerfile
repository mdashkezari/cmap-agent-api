# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.10

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    unixodbc \
    unixodbc-dev \
    tdsodbc \
    freetds-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PYTHONUNBUFFERED=1

# Build metadata (used by GET /version)
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
ENV CMAP_AGENT_GIT_SHA=${GIT_SHA}
ENV CMAP_AGENT_BUILD_TIME=${BUILD_TIME}

WORKDIR /app

# Copy only what's needed for env creation (better caching)
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/
COPY env/geo.yml /app/geo.yml

RUN micromamba create -y -n cmap-agent -f /app/geo.yml && \
    micromamba clean -a -y

ENV PATH=/opt/conda/envs/cmap-agent/bin:$PATH

COPY . /app

EXPOSE 8000
CMD ["micromamba","run","-n","cmap-agent","uvicorn", "cmap_agent.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
