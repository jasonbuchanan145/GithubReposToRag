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
COPY scripts/ /app/scripts/
COPY cassandra/schema.cql /app/cassandra/schema.cql

# Create __init__.py to make scripts a proper Python package
RUN touch /app/scripts/__init__.py

# Set development mode to always recreate data
ENV DEV_MODE=true

# ---------- entrypoint ----------
# Fix: Use consistent environment name and proper module path
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-demo", "python", "-m", "scripts.ingest_repos"]