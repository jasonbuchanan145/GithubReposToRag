# rag_worker/engine/rag_engine.py
from typing import Dict, Optional

from llama_index.core import Settings
from llama_index.core.response.schema import Response

from worker.services.router_service import RouterService  # your existing per-level indices


class RAGEngine:
    def __init__(self):
        self.router = RouterService()

    def run_once(self, query: str, force_level: Optional[str]) -> Response:
        """Single routed query (no retries)."""
        return self.router.route(query, force_level=force_level)

    def propose_refinement(self, query: str) -> Dict[str, Optional[str]]:
        """Ask the LLM to refine search terms and optionally pick a level."""
        prompt = (
            "Improve retrieval for a codebase with levels code, package, project. "
            "Return JSON with keys query and force_level (code|package|project|none).\n\n"
            f"Question: {query}\n"
        )
        raw = Settings.llm.complete(prompt).text.strip()
        import json, re
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0)) if m else json.loads(raw)
            q2 = str(data.get("query") or query)
            lvl = data.get("force_level")
            if lvl not in {"code", "package", "project", "none", None}:
                lvl = None
            return {"query": q2, "force_level": lvl}
        except Exception:
            return {"query": query, "force_level": None}
