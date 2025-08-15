# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.controllers.jobs_controller import router as jobs_router
from app.health import register_health_endpoints
from rag_shared.config import (
    LOG_LEVEL,
    CASSANDRA_USERNAME, CASSANDRA_PASSWORD, CASSANDRA_HOST, CASSANDRA_PORT, CASSANDRA_KEYSPACE,
    QWEN_ENDPOINT,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI(title="RAG API Service", description="RAG service with query routing", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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