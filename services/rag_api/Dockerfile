# ---------- base image ----------
FROM continuumio/miniconda3 AS base

# ---------- Conda env ----------
COPY ../../environment-service.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "rag-demo-service", "/bin/bash", "-c"]

# ---------- app code ----------
WORKDIR /app
COPY services/rag_api/ /app/

# ---------- runtime ----------
EXPOSE 8000
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-demo", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]