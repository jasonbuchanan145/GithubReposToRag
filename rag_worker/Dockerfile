FROM continuumio/miniconda3 AS base

# Create the worker env
COPY environment-worker.yaml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate env for subsequent layers - Fix environment name
SHELL ["conda", "run", "--no-capture-output", "-n", "worker", "/bin/bash", "-c"]
# ---------- app code ----------
WORKDIR /app
COPY rag_worker/pyproject.toml /app/pyproject.toml
COPY rag_worker/src /app/src
COPY rag_shared/ /app/rag_shared/

# Make shared imports work
ENV PYTHONPATH=/app
RUN pip install -e .
RUN python -c "import arq, sys; print('arq', arq.__version__)"