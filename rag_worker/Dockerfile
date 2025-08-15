# Worker image with the heavy deps (LlamaIndex, Cassandra, Torch, etc.)
FROM mambaorg/micromamba:1.5.8

# Create the worker env
COPY environment-worker.yaml /tmp/environment.yml
RUN micromamba env create -f /tmp/environment.yml -y && micromamba clean --all -y

# Run all subsequent commands inside the env
SHELL ["micromamba", "run", "-n", "rag-worker", "/bin/bash", "-lc"]

# App code
WORKDIR /app
# Copy only what the worker needs + shared package
COPY rag_worker/ /app/rag_worker/
COPY rag_shared/ /app/rag_shared/

# Make shared imports work
ENV PYTHONPATH=/app

# Default command: run the arq worker
CMD ["arq", "rag_worker.worker.WorkerSettings"]