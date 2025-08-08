# ---------- base image ----------
FROM continuumio/miniconda3 AS base

# ---------- Conda env ----------
COPY environment-scripts.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate env for subsequent layers - Fix environment name
SHELL ["conda", "run", "--no-capture-output", "-n", "rag-demo", "/bin/bash", "-c"]
# ---------- app code ----------
WORKDIR /app

# Copy just what pip needs first (better layer cache)
COPY ingest/pyproject.toml /app/pyproject.toml
# If you have a README and want it included:
# COPY README.md /app/README.md

# Copy the source package
COPY ingest/src/ /app/src/

# Optional: other files
COPY cassandra/schema.cql /app/cassandra/schema.cql

RUN pip install -e .

ENV DEV_MODE=true
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-demo", "python", "-m", "app"]