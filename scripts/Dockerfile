# ---------- base image ----------
FROM continuumio/miniconda3 AS base

# ---------- Conda env ----------
COPY environment-scripts.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate env for subsequent layers
SHELL ["conda", "run", "--no-capture-output", "-n", "rag-demo-scripts", "/bin/bash", "-c"]

# ---------- app code ----------
WORKDIR /app
COPY scripts/ /app/scripts/
COPY cassandra/schema.cql /app/cassandra/schema.cql   # handy for local tests

# ---------- entrypoint ----------
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-demo-scripts", "python", "-m", "scripts.ingest_repo"]