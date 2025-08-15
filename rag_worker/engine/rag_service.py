# app/services/rag_service.py
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from llama_index.core.response.schema import Response
from app.services.router_service import RouterService
from llama_index.core import Settings
from app.config import MAX_RAG_ATTEMPTS, MIN_SOURCE_NODES

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None  # optional override; defaults handled in RouterService
    repo_name: Optional[str] = None  # reserved for future metadata filters
    force_level: Optional[str] = None  # "none" | "code" | "package" | "project"

class RAGResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

class RAGService:
    def __init__(self):
        self.router = RouterService()

    # --- feedback loop helpers ---
    def _should_retry(self, sources_count: int, attempt: int, force_level: Optional[str]) -> bool:
        # No retries for forced no-RAG, or once attempts exhausted
        if force_level == "none":
            return False
        if attempt >= MAX_RAG_ATTEMPTS:
            return False
        return sources_count < MIN_SOURCE_NODES

    def _propose_refinement(self, query: str) -> Dict[str, Optional[str]]:
        """Use Qwen to refine the search query and optionally suggest a level.
        Returns {"query": str, "force_level": Optional[str]}.
        """
        prompt = (
            "Improve the retrieval recall for a codebase with levels code, package, project. "
            "Given the user's question, propose better search keywords and the most appropriate level. "
            "Respond as JSON with keys query and force_level (code|package|project|none).\n\n"
            f"Question: {query}\n"
        )
        try:
            raw = Settings.llm.complete(prompt).text.strip()
            import json, re
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0)) if m else json.loads(raw)
            q2 = str(data.get("query") or query)
            lvl = data.get("force_level")
            if lvl not in {"code", "package", "project", "none", None}:
                lvl = None
            return {"query": q2, "force_level": lvl}
        except Exception:
            return {"query": query, "force_level": None}

    # NEW: streaming generator for SSE
    def run_stream(self, req: QueryRequest) -> Iterator[str]:
        """
        Yields text/event-stream messages:
          - event: iteration / refinement / final / error / ping
          - data: JSON payload
        """
        def sse(event: str, data: Dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            attempts = 0
            q, forced = req.query, req.force_level
            last_resp: Optional[Response] = None
            last_sources: List[Dict[str, Any]] = []

            yield sse("iteration", {"attempt": 1, "max_attempts": MAX_RAG_ATTEMPTS,
                                    "query": q, "force_level": forced})

            while attempts < MAX_RAG_ATTEMPTS:
                attempts += 1

                # run retrieval
                resp = self.router.route(q, force_level=forced)
                last_resp = resp
                sources = self._extract_sources(resp)
                last_sources = sources

                yield sse("retrieval", {
                    "attempt": attempts,
                    "sources_found": len(sources),
                    "force_level": forced,
                })

                # done?
                if not self._should_retry(len(sources or []), attempts, forced):
                    break

                # propose refinement
                refine = self._propose_refinement(q)
                q = refine.get("query") or q
                forced = refine.get("force_level") or forced

                yield sse("refinement", {
                    "next_attempt": attempts + 1,
                    "max_attempts": MAX_RAG_ATTEMPTS,
                    "new_query": q,
                    "new_force_level": forced,
                })

            yield sse("final", {
                "answer": str(last_resp) if last_resp else "",
                "sources": last_sources or None,
            })

        except Exception as e:
            yield sse("error", {"message": str(e)})