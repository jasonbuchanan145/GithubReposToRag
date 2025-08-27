# app/main.py
import logging
import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.controllers.jobs_controller import router as jobs_router
from app.health import register_health_endpoints
from rag_shared.config import (
    LOG_LEVEL,
    CASSANDRA_USERNAME, CASSANDRA_PASSWORD, CASSANDRA_HOST, CASSANDRA_PORT, CASSANDRA_KEYSPACE,
    QWEN_ENDPOINT,
)
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI(title="RAG API Service", description="RAG service with query routing", version="2.0.0")

# Prometheus metrics for API
REQUEST_COUNT = Counter(
    "rest_api_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "rest_api_request_duration_seconds",
    "Request duration in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    labelnames=["method", "path", "status"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic request/response metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start
    # use route path if available, otherwise raw path
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    labels = {
        "method": request.method,
        "path": path,
        "status": str(response.status_code),
    }
    REQUEST_COUNT.labels(**labels).inc()
    REQUEST_LATENCY.labels(**labels).observe(elapsed)
    return response

# Expose Prometheus metrics
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Mount static files
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

app.include_router(jobs_router)

@app.on_event("startup")
async def on_startup():
    register_health_endpoints(
        app=app,
        CASSANDRA_USERNAME=CASSANDRA_USERNAME,
        CASSANDRA_PASSWORD=CASSANDRA_PASSWORD,
        CASSANDRA_HOST=CASSANDRA_HOST,
        CASSANDRA_PORT=CASSANDRA_PORT,
        CASSANDRA_KEYSPACE=CASSANDRA_KEYSPACE,
        QWEN_ENDPOINT=QWEN_ENDPOINT,
        global_index=None,  # indices now managed inside services per-level
    )