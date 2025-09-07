
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from arq import create_pool
from arq.connections import RedisSettings

from prometheus_client import Counter, Histogram, start_http_server

from rag_shared.bus import ProgressBus, CancelFlags
from rag_shared.config import (
    REDIS_URL, MAX_RAG_ATTEMPTS, MIN_SOURCE_NODES,
    METRICS_PORT as DEFAULT_METRICS_PORT,  # you can keep your env var
    DEFAULT_NAMESPACE,
)

from worker.services.agent_graph import GraphAgent  # <-- the agent above
from worker.services.qwen_llm import QwenLLM

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_worker")
logger.setLevel(logging.INFO)
logging.getLogger("worker").setLevel(logging.INFO)
logging.getLogger("rag_shared").setLevel(logging.INFO)

# ---------- Metrics ----------
METRICS_PORT = int(os.getenv("METRICS_PORT", str(DEFAULT_METRICS_PORT or 9000)))
try:
    start_http_server(METRICS_PORT)
    logging.info(f"Prometheus metrics server started on port {METRICS_PORT}")
except Exception:
    logging.exception("Failed to start Prometheus metrics server")

WORKER_JOBS_TOTAL = Counter("rag_worker_jobs_total", "Total RAG jobs processed", ["status"])  # success|error|cancelled
WORKER_JOB_DURATION = Histogram("rag_worker_job_duration_seconds", "Duration of RAG jobs")
WORKER_LLM_CALLS_TOTAL = Counter("rag_worker_llm_calls_total", "Total LLM calls from worker", ["result"])  # ok|error
WORKER_LLM_DURATION = Histogram("rag_worker_llm_duration_seconds", "Duration of LLM calls in worker")
WORKER_RETRIEVAL_DURATION = Histogram("rag_worker_retrieval_seconds", "Time spent in GraphRAG retrieval+planning")

# ---------- Bus / cancel ----------
bus = ProgressBus(REDIS_URL)
flags = CancelFlags(REDIS_URL)

import asyncio

def make_progress_callback(job_id: str, loop: asyncio.AbstractEventLoop, bus: ProgressBus):
    """
    Create a thread-safe callback that can be called from the agent's executor thread.
    It schedules `bus.emit(job_id, "turn", payload)` onto the main event loop.
    """
    def _cb(payload: Dict[str, Any]):
        try:
            fut = asyncio.run_coroutine_threadsafe(
                bus.emit(job_id, "turn", payload),
                loop
            )
            # Optional: swallow errors so the agent isn't affected by bus failures
            # fut.result(timeout=0.0)  # fire-and-forget
        except Exception:
            logger.exception("turn emit failed")
    return _cb

# ---------- LLM metering wrapper ----------
class MeteredLLM:
    """Wraps your QwenLLM to record Prometheus metrics for every .complete() call."""
    def __init__(self, base: QwenLLM):
        self._base = base

    def complete(self, prompt: str):
        t0 = time.perf_counter()
        try:
            out = self._base.complete(prompt)
            WORKER_LLM_DURATION.observe(time.perf_counter() - t0)
            WORKER_LLM_CALLS_TOTAL.labels(result="ok").inc()
            return out
        except Exception:
            WORKER_LLM_DURATION.observe(time.perf_counter() - t0)
            WORKER_LLM_CALLS_TOTAL.labels(result="error").inc()
            raise

# ---------- Agent singleton ----------
_agent: Optional[GraphAgent] = None
def get_agent() -> GraphAgent:
    global _agent
    if _agent is None:
        metered = MeteredLLM(QwenLLM())
        _agent = GraphAgent(namespace=DEFAULT_NAMESPACE, max_iters=MAX_RAG_ATTEMPTS, llm=metered)
    return _agent

async def run_rag_job(ctx, job_id: str, req: Dict[str, Any]) -> None:
    """
    req: { query: str, force_level?: "project"|"package"|"file"|"code"|"none"|null,
           namespace?: str }
    """
    t_job = time.perf_counter()
    query = (req.get("query") or "").strip()
    forced = req.get("force_level")
    namespace = req.get("namespace") or DEFAULT_NAMESPACE

    await bus.emit(job_id, "started", {
        "query": query, "force_level": forced, "max_attempts": MAX_RAG_ATTEMPTS
    })

    attempts = 0
    final_answer: str = ""
    final_sources: List[Dict[str, Any]] = []
    last_debug: Dict[str, Any] = {}

    try:

        # cooperative cancel before expensive work
        if await flags.is_cancelled(job_id):
            await bus.emit(job_id, "final", {"answer": "", "sources": None, "cancelled": True})
            WORKER_JOBS_TOTAL.labels(status="cancelled").inc()
            return

        await bus.emit(job_id, "iteration", {
            "attempt": attempts, "query": query, "force_level": forced, "namespace": namespace
        })

        agent = get_agent()


        loop = asyncio.get_event_loop()
        progress_cb = make_progress_callback(job_id, loop, bus)
        t_rag = time.perf_counter()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run(query, namespace=namespace, progress_cb=progress_cb)
        )
        WORKER_RETRIEVAL_DURATION.observe(time.perf_counter() - t_rag)
        final_answer = result.get("answer", "")
        final_sources = result.get("sources", [])  # [{block, score?, metadata, text}]
        last_debug = result.get("debug", {})

        await bus.emit(job_id, "retrieval", {
            "attempt": attempts,
            "scope": result.get("scope", ""),
            "sources_found": len(final_sources),
            "turns": last_debug.get("turns", []),            # breadcrumbs (plan/retrieve/judge/rewrite)
            "final_ctx_blocks": last_debug.get("final_ctx_blocks", 0),
        })

        # Basic sharpening: if the agent suggested filters in debug, bake them in
        # (we stored decisions in turns; pull last judge suggestion if any)
        try:
            judges = [t for t in last_debug.get("turns", []) if t.get("stage") == "judge"]
            if judges:
                decision = judges[-1].get("decision") or {}
                sf = (decision.get("suggest_filters") or {})
                parts = [query]
                for k in ("repo", "module", "topics"):
                    v = sf.get(k)
                    if isinstance(v, str) and v:
                        parts.append(f"{k}:{v}")
                query = " ".join(dict.fromkeys(parts))
        except Exception:
            pass

        # --- final emit ---
        await bus.emit(job_id, "final", {"answer": final_answer, "sources": final_sources or None})
        WORKER_JOBS_TOTAL.labels(status="success").inc()
    except Exception as e:
        logger.exception("Worker job failed")
        WORKER_JOBS_TOTAL.labels(status="error").inc()
        await bus.emit(job_id, "error", {"message": str(e)})
        await bus.emit(job_id, "final", {"answer": "", "sources": None, "error": True})
    finally:
        WORKER_JOB_DURATION.observe(time.perf_counter() - t_job)

# ---------- ARQ settings ----------

class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    functions = [run_rag_job]
    max_jobs = 10
    job_timeout = 300
    keep_result = 3600
