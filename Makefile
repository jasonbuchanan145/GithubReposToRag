WORKER_ENV ?= worker
REST_ENV   ?= rag-api
INGEST_ENV ?= rag-demo

.PHONY: envs
envs:
	conda env create -f environment-worker.yaml -n $(WORKER_ENV) || true
	conda env create -f environment-rest-api.yaml -n $(REST_ENV)   || true
	conda env create -f environment-scripts.yml -n $(INGEST_ENV)   || true

.PHONY: test-worker
test-worker:
	conda run -n $(WORKER_ENV) python -m pip install -e ./rag_worker
	conda run -n $(WORKER_ENV) python -m pip install -U "pytest>=7.0" "pytest-asyncio>=0.21" "pytest-mock>=3.10" "pytest-cov>=4.0"
	conda run -n $(WORKER_ENV) pytest ./rag_worker/tests

.PHONY: test-rest
test-rest:
	conda run -n $(REST_ENV) python -m pip install -e ./rest_api
	conda run -n $(REST_ENV) pytest ./rest_api/tests

.PHONY: test-ingest
test-ingest:
	conda run -n $(INGEST_ENV) python -m pip install -e ./ingest
	conda run -n $(INGEST_ENV) pytest ./ingest/tests

.PHONY: test-all
test-all: test-worker test-rest test-ingest
