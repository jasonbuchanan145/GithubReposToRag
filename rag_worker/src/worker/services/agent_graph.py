# worker/services/agent_graph.py
from __future__ import annotations

import logging
from typing import TypedDict, Optional, Dict, List, Any, Tuple, Callable
import json
import re

from langgraph.graph import StateGraph, END
from rag_shared.config import (
    CASSANDRA_HOSTS, CASSANDRA_KEYSPACE,
    CASSANDRA_USERNAME, CASSANDRA_PASSWORD,
    EMBED_MODEL, DEFAULT_NAMESPACE,
    ROUTER_TOP_K,
)
from graph_rag_retrievers import make_graph_retriever_factory, TableNames
from worker.services.qwen_llm import QwenLLM  # your existing LLM client

# ---------- State ----------

class AgentState(TypedDict, total=False):
    query: str
    scope: str                          # "project"|"package"|"file"|"code"
    filters: Dict[str, str]             # {"namespace": "...", "repo": "...", "module": "...", "topics": "..."}
    attempt: int
    docs: List[Any]                     # LangChain Document-like (per turn)
    needs_more: bool
    rewrite: Optional[str]
    answer: Optional[str]
    debug: Dict[str, Any]               # breadcrumbs

TECH_SYNONYMS = {"activemq": ["activemq", "jms", "amq", "failovertransport", "redeliverypolicy", "broker", "stomp"]}

def looks_codey(q: str) -> bool:
    ql = q.lower()
    return any(s in ql for s in (
        "stacktrace", "traceback", "exception", "error", "class ", "function ", "method ",
        "nullpointer", "undefined", "timeout", "reconnect", "retry", "activemq", "jms"
    ))

def extract_repo_hint(q: str) -> Optional[str]:
    m = re.search(r"(?:repo(?:sitory)?[:\s]+)([\w\-./]+)", q, re.I)
    return m.group(1) if m else None

def _score_of(doc: Any) -> Optional[float]:
    # Try common locations for similarity/score; GraphRAG/LangChain often tuck it into metadata.
    md = getattr(doc, "metadata", {}) or {}
    for k in ("_score", "score", "similarity", "distance"):
        v = md.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)  # if it's a str "0.78"
        except Exception:
            pass
    # Some vectorstores attach .score attribute
    if hasattr(doc, "score"):
        try:
            return float(getattr(doc, "score"))
        except Exception:
            return None
    return None
def _notify(self, payload: Dict[str, Any]) -> None:
    cb = getattr(self, "_progress_cb", None)
    if cb:
        try:
            cb(payload)
        except Exception:
            pass

def _doc_to_source(i: int, doc: Any) -> Dict[str, Any]:
    md = getattr(doc, "metadata", {}) or {}
    text = getattr(doc, "page_content", "") or getattr(doc, "text", "") or ""
    return {
        "block": i,
        "score": _score_of(doc),
        "metadata": {
            "scope": md.get("scope", ""),
            "namespace": md.get("namespace", ""),
            "repo": md.get("repo", ""),
            "module": md.get("module", ""),
            "file_path": md.get("file_path", ""),
            "topics": md.get("topics", ""),
        },
        "text": text[:1200],  # trim for transport; your UI can request more on expand
    }

class GraphAgent:
    """
    Native GraphRAG retrieval agent (LangGraph):
      - plan scope
      - retrieve via GraphRetriever (Cassandra + metadata_s edges)
      - judge coverage / stage-down
      - retry up to N
      - synthesize + return sources
    """
    def _notify(self, payload: Dict[str, Any]) -> None:
        cb = getattr(self, "_progress_cb", None)
        if cb:
            try:
                cb(payload)
            except Exception:
                logging.exception("Progress callback failed")
    def __init__(self, namespace: Optional[str] = None, max_iters: int = 3,
                 llm: Optional[Any] = None, progress_cb: Optional[Callable[[dict], None]] = None):
        self.llm = llm or QwenLLM()
        self.namespace = namespace or DEFAULT_NAMESPACE
        self.max_iters = max_iters
        self._progress_cb = progress_cb

        factory = make_graph_retriever_factory(
            hosts=CASSANDRA_HOSTS,
            keyspace=CASSANDRA_KEYSPACE,
            username=CASSANDRA_USERNAME,
            password=CASSANDRA_PASSWORD,
            tables=TableNames(
                repo="embeddings_repo",
                module="embeddings_module",
                file="embeddings_file",
                chunk="embeddings",
            ),
            embedding_model=EMBED_MODEL,
        )
        self.retrievers = {
            "project": factory.for_repo(k=6, start_k=2, max_depth=2),
            "package": factory.for_module(k=8, start_k=2, adjacent_k=6, max_depth=2),
            "file":    factory.for_file(k=8, start_k=2, adjacent_k=6, max_depth=2),
            "code":    factory.for_chunk(k=10, start_k=3, adjacent_k=8, max_depth=2),
        }

        # Build the graph
        g = StateGraph(AgentState)
        g.add_node("plan_scope", self.plan_scope)
        g.add_node("retrieve", self.retrieve)
        g.add_node("judge", self.judge)
        g.add_node("rewrite_or_end", self.rewrite_or_end)
        g.add_node("synthesize", self.synthesize)

        g.set_entry_point("plan_scope")
        g.add_edge("plan_scope", "retrieve")
        g.add_edge("retrieve", "judge")
        g.add_edge("judge", "rewrite_or_end")
        g.add_conditional_edges("rewrite_or_end", self._route_from_decision,
                                {"retry": "retrieve", "synthesize": "synthesize", "end": END})
        g.add_edge("synthesize", END)

        self.app = g.compile()

    # ---------- Nodes ----------

    def plan_scope(self, state: AgentState) -> AgentState:
        q = state["query"]
        filters = dict(state.get("filters") or {})
        if self.namespace:
            filters.setdefault("namespace", self.namespace)
        rh = extract_repo_hint(q)
        if rh:
            filters["repo"] = rh

        sys = (
            "Choose the best search scope for a codebase question. "
            "Return JSON: {scope: project|package|file|code, filters?:{repo?,module?,topics?}}"
        )
        ex = 'Example: {"scope":"package","filters":{"repo":"payments","module":"messaging","topics":"activemq"}}'
        msg = f"{sys}\nQuestion: {q}\n{ex}\nJSON:"
        try:
            raw = self.llm.complete(msg).text.strip()
            raw = raw[raw.find("{"): raw.rfind("}")+1]
            data = json.loads(raw)
            scope = data.get("scope") or ("code" if looks_codey(q) else "project")
            for k, v in (data.get("filters") or {}).items():
                if isinstance(v, str) and v:
                    filters[k] = v
        except Exception:
            scope = "code" if looks_codey(q) else "project"

        for tech, syns in TECH_SYNONYMS.items():
            if any(t in q.lower() for t in syns) and "topics" not in filters:
                filters["topics"] = tech
                break

        dbg = dict(state.get("debug") or {})
        dbg.setdefault("turns", []).append({"stage": "plan", "scope": scope, "filters": dict(filters)})
        evt = {"stage": "plan", "scope": scope, "filters": dict(filters), "attempt": state.get("attempt", 0)}
        self._notify(evt)
        return {**state, "scope": scope, "filters": filters, "attempt": state.get("attempt", 0), "debug": dbg}

    def retrieve(self, state: AgentState) -> AgentState:
        scope = state["scope"]
        q = state["query"]
        filters = state.get("filters") or {}
        retriever = self.retrievers[scope]
        docs = retriever.invoke(q, filter=filters) or []
        docs = docs[: ROUTER_TOP_K]
        dbg = dict(state.get("debug") or {})
        dbg.setdefault("turns", []).append({"stage": "retrieve", "scope": scope, "filters": dict(filters), "hits": len(docs)})
        self._notify({"stage": "retrieve", "scope": scope, "filters": dict(filters), "hits": len(docs)})
        return {**state, "docs": docs, "debug": dbg}

    def judge(self, state: AgentState) -> AgentState:
        q = state["query"]
        docs = state.get("docs") or []
        inv = []
        for i, d in enumerate(docs, start=1):
            md = getattr(d, "metadata", {}) or {}
            inv.append({
                "i": i, "repo": md.get("repo", ""), "module": md.get("module", ""),
                "file": md.get("file_path", ""), "topics": md.get("topics", "")
            })

        rubric = (
            "Judge if the context is enough to answer now. Return JSON: "
            "{coverage:0..1, needs_more:boolean, suggest_filters?:{repo?,module?,topics?}, "
            "stage_down?: 'package'|'code'|null, rewrite?:string}"
        )
        msg = f"{rubric}\nQuestion: {q}\nInventory: {json.dumps(inv, ensure_ascii=False)}\nJSON:"
        try:
            raw = self.llm.complete(msg).text.strip()
            raw = raw[raw.find("{"): raw.rfind("}")+1]
            data = json.loads(raw)
        except Exception:
            data = {"coverage": 0.4, "needs_more": looks_codey(q), "stage_down": "code" if looks_codey(q) else None}

        filters = dict(state.get("filters") or {})
        for k, v in (data.get("suggest_filters") or {}).items():
            if isinstance(v, str) and v:
                filters[k] = v

        next_scope = state["scope"]
        if data.get("stage_down") in {"package", "code"}:
            next_scope = data["stage_down"]

        dbg = dict(state.get("debug") or {})
        dbg.setdefault("turns", []).append({"stage": "judge", "decision": data})
        self._notify({"stage": "judge", "decision": data})
        return {**state, "needs_more": bool(data.get("needs_more")), "rewrite": data.get("rewrite"),
                "filters": filters, "scope": next_scope, "debug": dbg}

    def rewrite_or_end(self, state: AgentState) -> AgentState:
        if not state.get("needs_more"):
            return state
        attempt = int(state.get("attempt", 0)) + 1
        if attempt >= self.max_iters:
            return {**state, "needs_more": False, "attempt": attempt}

        rewrite = state.get("rewrite") or state["query"]
        filters = state.get("filters") or {}
        parts = [rewrite]
        if "repo" in filters:   parts.append(f"repo:{filters['repo']}")
        if "module" in filters: parts.append(f"module:{filters['module']}")
        if "topics" in filters: parts.append(f"topic:{filters['topics']}")

        for tech, syns in TECH_SYNONYMS.items():
            if tech in " ".join(parts).lower():
                parts.extend(syns)
                # dedupe while preserving order
                parts = list(dict.fromkeys(parts))
                break

        sharpened = " ".join(parts)
        dbg = dict(state.get("debug") or {})
        dbg.setdefault("turns", []).append({"stage": "rewrite", "attempt": attempt+1, "query": sharpened, "filters": dict(filters)})
        self._notify({"stage": "rewrite", "action": "retry", "attempt": attempt + 1,
                      "query": sharpened, "filters": dict(filters)})
        return {**state, "query": sharpened, "attempt": attempt, "debug": dbg}

    def synthesize(self, state: AgentState) -> AgentState:
        q = state["query"]
        docs = state.get("docs") or []
        # Build blocks and a sources list (with scores if present)
        blocks, sources = [], []
        for i, d in enumerate(docs[:ROUTER_TOP_K], start=1):
            md = getattr(d, "metadata", {}) or {}
            text = getattr(d, "page_content", "") or getattr(d, "text", "")
            blocks.append(f"[{i}] repo={md.get('repo','')} module={md.get('module','')} file={md.get('file_path','')}\n{text[:1800]}")
            sources.append(_doc_to_source(i, d))

        sys = ("You are a senior developer assistant. Answer using ONLY the context blocks. "
               "Cite blocks as [1], [2]. If context is insufficient, say so and suggest next repo/module.")
        prompt = f"{sys}\n\nQuestion: {q}\n\nContext:\n" + "\n\n".join(blocks) + "\n\nAnswer:"

        try:
            text = self.llm.complete(prompt).text
        except Exception as e:
            text = f"(LLM error) {e}"

        dbg = dict(state.get("debug") or {})
        dbg["final_ctx_blocks"] = len(blocks)
        dbg["sources_count"] = len(sources)
        dbg["final_scope"] = state.get("scope", "")
        self._notify({"stage": "synthesize", "final_ctx_blocks": len(blocks), "sources_count": len(sources)})
        return {**state, "answer": text, "debug": dbg, "docs": docs, "sources": sources}

    # ---------- Router for conditional edge ----------

    def _route_from_decision(self, state: AgentState):
        if state.get("needs_more"):
            return "retry"
        if state.get("answer"):
            return "synthesize"
        return "synthesize"

    # Public API: allow per-call override of the callback
    def run(self, question: str, *, namespace: Optional[str] = None,
            progress_cb: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
        prev = self._progress_cb
        if progress_cb is not None:
            self._progress_cb = progress_cb
        try:
            init: AgentState = {"query": question}
            if namespace or self.namespace:
                init["filters"] = {"namespace": namespace or self.namespace}
            final = self.app.invoke(init)
            return {
                "answer": final.get("answer", ""),
                "sources": final.get("sources", []),
                "debug": final.get("debug", {}),
                "scope": final.get("scope", ""),
            }
        finally:
            self._progress_cb = prev