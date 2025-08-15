# ---------- base image ----------
FROM continuumio/miniconda3 AS base

# Create the API env
COPY environment-rest-api.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate env for subsequent layers - Fix environment name
SHELL ["conda", "run", "--no-capture-output", "-n", "rag-api", "/bin/bash", "-c"]
# ---------- app code ----------
WORKDIR /app
COPY rest_api/pyproject.toml /app/pyproject.toml

# Copy only what the API needs + shared package
COPY rest_api/src /app/src
COPY rag_shared/ /app/rag_shared/

# Make shared imports work
ENV PYTHONPATH=/app
RUN pip install -e .

# Expose FastAPI port
EXPOSE 8000

# Default command: run API
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag-api", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]