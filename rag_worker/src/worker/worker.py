import asyncio
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.response.schema import Response
from rag_shared.bus import ProgressBus, CancelFlags
from rag_shared.config import (
    REDIS_URL, MAX_RAG_ATTEMPTS, MIN_SOURCE_NODES
)
from worker.services.rag_engine import RAGEngine

logger = logging.getLogger("rag_worker")
logger.setLevel(logging.INFO)

bus = ProgressBus(REDIS_URL)
flags = CancelFlags(REDIS_URL)
engine = RAGEngine()

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

# ---- main job function (called by arq worker) ------------------------

async def run_rag_job(ctx, job_id: str, req: Dict[str, Any]) -> None:
    """
    ctx is provided by arq, but we don't rely on it.
    req shape: { query: str, force_level?: str|null, top_k?: int, repo_name?: str }
    """
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
                return

            await bus.emit(job_id, "iteration", {
                "attempt": attempts, "query": query, "force_level": forced
            })

            # LlamaIndex is sync-heavy; run in a thread to keep the worker responsive
            loop = asyncio.get_event_loop()
            resp: Response = await loop.run_in_executor(None, engine.run_once, query, forced)
            last_resp = resp

            # extract sources (can be a bit heavy too)
            last_sources = await loop.run_in_executor(None, _extract_sources, resp)
            await bus.emit(job_id, "retrieval", {
                "attempt": attempts, "sources_found": len(last_sources)
            })

            if not _should_retry(len(last_sources), attempts, forced):
                break

            # propose refinement via LLM (sync → thread)
            refinement = await loop.run_in_executor(None, engine.propose_refinement, query)
            query = refinement.get("query") or query
            forced = refinement.get("force_level") or forced

            await bus.emit(job_id, "refinement", {
                "next_attempt": attempts + 1,
                "new_query": query,
                "new_force_level": forced,
            })

        await bus.emit(job_id, "final", {
            "answer": str(last_resp) if last_resp else "",
            "sources": last_sources or None
        })

    except Exception as e:
        logger.exception("Worker job failed")
        # send error then final (so UI can stop)
        await bus.emit(job_id, "error", {"message": str(e)})
        await bus.emit(job_id, "final", {"answer": "", "sources": None, "error": True})

# ---- arq settings ----------------------------------------------------

class WorkerSettings:
    functions = [run_rag_job]