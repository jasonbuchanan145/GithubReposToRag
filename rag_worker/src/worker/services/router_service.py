# worker/services/graph_orchestrator.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rag_shared.config import (
    ROUTER_TOP_K,
    CASSANDRA_HOST, CASSANDRA_KEYSPACE,
    CASSANDRA_USERNAME, CASSANDRA_PASSWORD,
    EMBED_MODEL, DEFAULT_NAMESPACE,
)
from worker.services.qwen_llm import QwenLLM
from graph_rag_retrievers import make_graph_retriever_factory, TableNames

# ----- Types -----

Scope = str  # "project" | "package" | "file" | "code"
SCOPE_TO_LEVEL = {
    "project": "repo",   # L1
    "package": "module", # L2
    "file":    "file",   # L3
    "code":    "chunk",  # L4
}
LEVEL_TO_SCOPE = {v: k for k, v in SCOPE_TO_LEVEL.items()}

@dataclass
class RAGResult:
    answer: str
    scope: Scope
    used_docs: List[Dict]  # [{block:int, repo:str, module:str, file_path:str, scope:str}]
    debug: Dict

# ----- Orchestrator -----

class GraphRAGOrchestrator:
    """
    Native GraphRAG retrieval:
      1) Scope judge (LLM) -> project/package/file/code
      2) GraphRetriever for chosen scope (Cassandra + edges over metadata_s)
      3) Optional stage-down (project->package->code) if the question is deep
      4) Synthesize final answer from top-k blocks
    """

    def __init__(self, namespace: Optional[str] = None):
        self.llm = QwenLLM()
        self.namespace = namespace or DEFAULT_NAMESPACE

        # Build retrievers
        factory = make_graph_retriever_factory(
            hosts=CASSANDRA_HOSTS,
            keyspace=CASSANDRA_KEYSPACE,     # "vector_store"
            username=CASSANDRA_USERNAME,
            password=CASSANDRA_PASSWORD,
            tables=TableNames(
                repo="embeddings_repo",
                module="embeddings_module",
                file="embeddings_file",
                chunk="embeddings",
            ),
            embedding_model=EMBED_MODEL,      # matches ingestion
        )
        self.retrievers = {
            "repo":   factory.for_repo(k=6, start_k=2, max_depth=2),
            "module": factory.for_module(k=8, start_k=2, adjacent_k=6, max_depth=2),
            "file":   factory.for_file(k=8, start_k=2, adjacent_k=6, max_depth=2),
            "chunk":  factory.for_chunk(k=10, start_k=3, adjacent_k=8, max_depth=2),
        }

    # ---------- Public API ----------

    def route(self, query: str, *, force_level: Optional[str] = None) -> RAGResult:
        """
        Main entrypoint. force_level ∈ {"project","package","file","code","none"}.
        Returns RAGResult with synthesized answer and used document metadata.
        """
        # 1) decide scope
        if force_level and force_level != "none":
            scope = force_level
        elif force_level == "none":
            # Direct LLM only (no retrieval)
            text = self.llm.complete(query).text
            return RAGResult(answer=text, scope="none", used_docs=[], debug={"path":"llm-only"})
        else:
            scope = self._judge_scope(query)

        level = SCOPE_TO_LEVEL.get(scope, "repo")  # default safe
        base_filter = {"namespace": self.namespace} if self.namespace else {}

        # 2) retrieve neighborhood at chosen level
        docs = self._retrieve(level, query, base_filter)

        # 3) zero/low-hit fallbacks and stage-down for deep questions
        # Heuristic: if scope is high-level but query mentions a repo/module,
        # or if the answer looks debugging-ish, drill down.
        down_docs = []
        down_chain = []
        if level in ("repo", "module"):
            hint = self._extract_filters_from_query(query)  # e.g., {"repo":"payments","module":"messaging"}
            if hint:
                base_filter |= hint
            if level == "repo":
                # stage-down to module
                down_docs = self._retrieve("module", query, base_filter) or []
                down_chain.append("repo→module")
                # optionally stage-down again if it looks code-y
                if self._looks_codey(query):
                    base_filter2 = dict(base_filter)  # keep same scope filters
                    down_docs2 = self._retrieve("chunk", query, base_filter2) or []
                    if down_docs2:
                        down_chain.append("module→chunk")
                        down_docs.extend(down_docs2)
            elif level == "module" and self._looks_codey(query):
                down_docs = self._retrieve("chunk", query, base_filter) or []
                down_chain.append("module→chunk")

        # 4) merge & cap context
        merged = (docs or []) + down_docs
        ctx_docs = merged[:ROUTER_TOP_K] if merged else []
        formatted_ctx, sources = self._format_context(ctx_docs)

        # 5) synthesize answer
        sys_instructions = (
            "You are a senior developer assistant. Use only the provided context blocks when making claims.\n"
            "If you cite files/modules, reference the block numbers like [1], [2]. "
            "If context is insufficient, say so and suggest the next repo/module to inspect."
        )
        prompt = f"{sys_instructions}\n\nQuestion: {query}\n\nContext:\n{formatted_ctx}\n\nAnswer:"
        try:
            text = self.llm.complete(prompt).text
        except Exception as e:
            text = f"(LLM error) {e}"

        return RAGResult(
            answer=text,
            scope=scope,
            used_docs=sources,
            debug={
                "level": level,
                "fallbacks": down_chain,
                "base_filter": base_filter,
                "ctx_count": len(ctx_docs),
            },
        )

    # ---------- Internals ----------

    def _judge_scope(self, query: str) -> Scope:
        """
        Ask the LLM to pick one of: project | package | file | code
        and (optionally) surface repo/module hints. Falls back to heuristics.
        """
        sys = (
            "Choose the best search scope for the user's question about a codebase. "
            "Return JSON with fields: scope that exists in {project, package, file, code}, "
            "and optional filters {repo, module}. Keep it compact."
        )
        ex = 'Example output: {"scope":"package","filters":{"repo":"payments","module":"messaging"}}'
        msg = f"{sys}\nQuestion: {query}\n{ex}\nJSON:"
        try:
            raw = self.llm.complete(msg).text.strip()
            raw = raw[raw.find("{"): raw.rfind("}")+1]  # crude guard
            data = json.loads(raw)
            scope = data.get("scope") or "project"
        except Exception:
            scope = self._heuristic_scope(query)
        return scope

    def _heuristic_scope(self, q: str) -> Scope:
        ql = q.lower()
        if any(k in ql for k in ("how do i", "architecture", "what repos", "overall", "component")):
            return "project"
        if any(k in ql for k in ("package", "module", "subsystem", "service ")):
            return "package"
        if any(k in ql for k in ("file ", ".py", ".java", "src/", "file:", "path:", "stacktrace", "traceback")):
            return "file"
        return "code"

    def _extract_filters_from_query(self, q: str) -> Dict[str, str]:
        # super simple hints; you can expand with regex on repo/module names
        out: Dict[str, str] = {}
        ql = q.lower()
        # e.g., "in payments-processor", "repo payments-processor"
        for key in ("repo", "repository"):
            if key in ql:
                # naive token grab; replace with better NER if needed
                toks = ql.split()
                try:
                    idx = toks.index(key)
                    out["repo"] = toks[idx+1].strip(",.")
                except Exception:
                    pass
        return out

    def _looks_codey(self, q: str) -> bool:
        ql = q.lower()
        signals = ("error", "stacktrace", "traceback", "exception", "class ", "function ", "method ",
                   "nullpointer", "undefined", "segfault", "timeout", "retry", "reconnect", "activemq", "jms")
        return any(s in ql for s in signals)

    def _retrieve(self, level: str, query: str, base_filter: Dict[str, str]):
        retriever = self.retrievers[level]
        return retriever.invoke(query, filter=base_filter) or []

    def _format_context(self, docs) -> Tuple[str, List[Dict]]:
        blocks: List[str] = []
        srcs: List[Dict] = []
        for i, d in enumerate(docs, start=1):
            md = getattr(d, "metadata", {}) or {}
            scope = md.get("scope") or LEVEL_TO_SCOPE.get(md.get("level",""), "")
            repo = md.get("repo","")
            module = md.get("module","")
            fpath = md.get("file_path","")
            text = getattr(d, "page_content", "") or getattr(d, "text", "")
            blocks.append(f"[{i}] scope={scope} repo={repo} module={module} file={fpath}\n{text[:2000]}")
            srcs.append({"block": i, "scope": scope, "repo": repo, "module": module, "file_path": fpath})
        return "\n\n".join(blocks) if blocks else "(no context)", srcs

