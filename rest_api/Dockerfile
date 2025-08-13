# ---------- base image ----------
FROM continuumio/miniconda3 AS base

# ---------- Conda env ----------
COPY environment-rest-api.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate env for subsequent layers - Fix environment name
SHELL ["conda", "run", "--no-capture-output", "-n", "rag-demo", "/bin/bash", "-c"]
# ---------- app code ----------
WORKDIR /app

# Copy just what pip needs first (better layer cache)
COPY rest_api/pyproject.toml /app/pyproject.toml

# Copy the source package
COPY rest_api/src/ /app/src/

RUN pip install -e .

ENV DEV_MODE=true
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-demo", "python", "-m", "app"]