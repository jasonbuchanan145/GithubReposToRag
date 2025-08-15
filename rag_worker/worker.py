import asyncio
from typing import Any, Dict, Optional, List
from llama_index.core.response.schema import Response
from rag_core.bus.progress_bus import ProgressBus, CancelFlags
from rag_core.services.rag_engine import RAGEngine
from rag_core.config import REDIS_URL, MAX_RAG_ATTEMPTS, MIN_SOURCE_NODES

bus = ProgressBus(REDIS_URL)
flags = CancelFlags(REDIS_URL)
engine = RAGEngine()

async def run_rag_job(ctx, job_id: str, req: Dict[str, Any]) -> None:
    q = req.get("query") or ""
    forced = req.get("force_level")
    await bus.emit(job_id, "started", {"query": q, "force_level": forced, "max_attempts": MAX_RAG_ATTEMPTS})

    attempts = 0
    last_resp: Optional[Response] = None
    last_sources: List[Dict[str, Any]] = []

    while attempts < MAX_RAG_ATTEMPTS:
        attempts += 1
        if await flags.is_cancelled(job_id):
            await bus.emit(job_id, "final", {"answer": "", "sources": None, "cancelled": True})
            return

        await bus.emit(job_id, "iteration", {"attempt": attempts, "query": q, "force_level": forced})

        resp: Response = await asyncio.get_event_loop().run_in_executor(None, engine.run_once, q, forced)
        last_resp = resp

        sources = await asyncio.get_event_loop().run_in_executor(None, engine._extract_sources, resp)
        last_sources = sources
        await bus.emit(job_id, "retrieval", {"attempt": attempts, "sources_found": len(sources)})

        if len(sources) >= MIN_SOURCE_NODES or await flags.is_cancelled(job_id) or attempts >= MAX_RAG_ATTEMPTS:
            break

        ref = await asyncio.get_event_loop().run_in_executor(None, engine._propose_refinement, q)
        q = ref.get("query") or q
        forced = ref.get("force_level") or forced
        await bus.emit(job_id, "refinement", {"next_attempt": attempts + 1, "new_query": q, "new_force_level": forced})

    await bus.emit(job_id, "final", {"answer": str(last_resp) if last_resp else "", "sources": last_sources or None})

class WorkerSettings:
    functions = [run_rag_job]