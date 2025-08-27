import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from arq import create_pool
from arq.connections import RedisSettings
from llama_index.core.base.response.schema import Response
from rag_shared.bus import ProgressBus, CancelFlags
from rag_shared.config import (
    REDIS_URL, MAX_RAG_ATTEMPTS, MIN_SOURCE_NODES
)
from worker.services.rag_engine import RAGEngine
from prometheus_client import Counter, Histogram, start_http_server

# Configure root logging to show all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Also ensure our specific logger is set up
logger = logging.getLogger("rag_worker")
logger.setLevel(logging.INFO)

# Set logging level for all relevant loggers
logging.getLogger("worker").setLevel(logging.INFO)
logging.getLogger("rag_shared").setLevel(logging.INFO)

# Metrics setup
METRICS_PORT = int(os.getenv("METRICS_PORT", "9000"))
try:
    start_http_server(METRICS_PORT)
    logging.info(f"Prometheus metrics server started on port {METRICS_PORT}")
except Exception:
    logging.exception("Failed to start Prometheus metrics server")

WORKER_JOBS_TOTAL = Counter("rag_worker_jobs_total", "Total RAG jobs processed", ["status"])  # status: success|error|cancelled
WORKER_JOB_DURATION = Histogram("rag_worker_job_duration_seconds", "Duration of RAG jobs")
WORKER_LLM_CALLS_TOTAL = Counter("rag_worker_llm_calls_total", "Total LLM calls from worker", ["result"])  # result: ok|error
WORKER_LLM_DURATION = Histogram("rag_worker_llm_duration_seconds", "Duration of LLM calls in worker")

bus = ProgressBus(REDIS_URL)
flags = CancelFlags(REDIS_URL)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = RAGEngine()
    return engine

def _extract_sources(resp: Response) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        nodes = getattr(resp, "source_nodes", None) or []
        for n in nodes:
            text = n.node.text or ""
            out.append({
                "text": (text[:300] + "...") if len(text) > 300 else text,
                "metadata": getattr(n.node, "metadata", {}),
                "score": getattr(n, "score", None),
            })
    except Exception:
        logger.exception("extract_sources failed")
    return out

def _should_retry(sources_count: int, attempt: int, forced: Optional[str]) -> bool:
    if forced == "none":
        return False
    if attempt >= MAX_RAG_ATTEMPTS:
        return False
    return sources_count < MIN_SOURCE_NODES


async def run_rag_job(ctx, job_id: str, req: Dict[str, Any]) -> None:
    """
    ctx is provided by arq, but we don't rely on it.
    req shape: { query: str, force_level?: str|null, top_k?: int, repo_name?: str }
    """
    t_job = time.perf_counter()
    query = (req.get("query") or "").strip()
    forced = req.get("force_level")
    attempts = 0
    last_resp: Optional[Response] = None
    last_sources: List[Dict[str, Any]] = []

    await bus.emit(job_id, "started", {
        "query": query, "force_level": forced, "max_attempts": MAX_RAG_ATTEMPTS
    })

    try:
        while attempts < MAX_RAG_ATTEMPTS:
            attempts += 1

            # cooperative cancel before expensive work
            if await flags.is_cancelled(job_id):
                await bus.emit(job_id, "final", {"answer": "", "sources": None, "cancelled": True})
                WORKER_JOBS_TOTAL.labels(status="cancelled").inc()
                return

            await bus.emit(job_id, "iteration", {
                "attempt": attempts, "query": query, "force_level": forced
            })

            eng = get_engine()
            loop = asyncio.get_event_loop()
            # --- main LLM call (timed) ---------------------------------------
            t_llm = time.perf_counter()
            resp: Response = await loop.run_in_executor(None, eng.run_once, query, forced)
            WORKER_LLM_DURATION.observe(time.perf_counter() - t_llm)
            WORKER_LLM_CALLS_TOTAL.labels(result="ok").inc()
            last_resp = resp

            # --- extract sources (threaded; can be heavier) ------------------
            last_sources = await loop.run_in_executor(None, _extract_sources, resp)
            await bus.emit(job_id, "retrieval", {
                "attempt": attempts, "sources_found": len(last_sources)
            })

            # If the (trimmed) query is empty, do NOT refine/retry.
            if not query:
                break

            if not _should_retry(len(last_sources), attempts, forced):
                break

            # --- propose refinement via LLM (defensive against MagicMocks) ---
            refinement = await loop.run_in_executor(None, eng.propose_refinement, query)
            if not isinstance(refinement, dict):
                # tests that patch .propose_refinement without setting a dict return
                # can yield a MagicMock; normalize to empty dict to avoid .get() on a mock
                refinement = {}

            new_query = refinement.get("query") or query
            new_forced = refinement.get("force_level") or forced

            query = new_query
            forced = new_forced

            await bus.emit(job_id, "refinement", {
                "next_attempt": attempts + 1,
                "new_query": query,
                "new_force_level": forced,
            })

        # --- final ------------------------------------------------------------
        await bus.emit(job_id, "final", {
            # keep your original shape; if you prefer, you can switch to last_resp.response
            "answer": str(last_resp) if last_resp else "",
            "sources": last_sources or None
        })
        WORKER_JOBS_TOTAL.labels(status="success").inc()

    except Exception as e:
        logger.exception("Worker job failed")
        WORKER_JOBS_TOTAL.labels(status="error").inc()
        # send error then final (so UI can stop)
        await bus.emit(job_id, "error", {"message": str(e)})
        await bus.emit(job_id, "final", {"answer": "", "sources": None, "error": True})
    finally:
        WORKER_JOB_DURATION.observe(time.perf_counter() - t_job)



class WorkerSettings:
    """
    ARQ Worker Settings configuration.
    """
    # Use Redis URL from environment (set by Kubernetes deployment)
    redis_settings = RedisSettings.from_dsn(REDIS_URL)

    # List of functions that this worker can execute
    functions = [run_rag_job]

    # Optional: Worker configuration
    max_jobs = 10
    job_timeout = 300  # 5 minutes
    keep_result = 3600  # Keep results for 1 hour