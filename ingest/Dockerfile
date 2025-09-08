# ---------- base image ----------
FROM continuumio/miniconda3 AS base

# ---------- Conda env ----------
COPY environment-ingest.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate env for subsequent layers - Fix environment name
SHELL ["conda", "run", "--no-capture-output", "-n", "rag-demo", "/bin/bash", "-c"]
# ---------- app code ----------
WORKDIR /app

COPY ingest/pyproject.toml /app/pyproject.toml

# Copy the source package
COPY ingest/src/ /app/src/
COPY rag_shared/ /app/rag_shared/

RUN pip install -e .

ENV DEV_MODE=true
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-demo", "python", "-m", "app"]