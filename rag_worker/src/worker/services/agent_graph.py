# worker/services/agent_graph.py
from __future__ import annotations

import logging
from typing import TypedDict, Optional, Dict, List, Any, Tuple, Callable
import json
import re

from langgraph.graph import StateGraph, END
from rag_shared.config import (
    CASSANDRA_HOST, CASSANDRA_KEYSPACE,
    CASSANDRA_USERNAME, CASSANDRA_PASSWORD,
    EMBED_MODEL, DEFAULT_NAMESPACE,
    ROUTER_TOP_K,
)
from worker.services.graph_rag_retrievers import make_graph_retriever_factory, TableNames
from worker.services.qwen_llm import QwenLLM
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

    def _expand_query_semantically(self, original_query: str, context_info: Dict[str, Any] = None) -> List[str]:
        """Generate semantically related queries for better embedding search"""
        context_info = context_info or {}

        sys_prompt = (
            "Generate 3-4 semantically related search queries for a codebase question. "
            "Focus on technical synonyms, related concepts, and different ways to express the same need. "
            "Return JSON array of strings: [\"query1\", \"query2\", \"query3\"]"
        )

        context_str = ""
        if context_info.get("repo"):
            context_str += f" Repository: {context_info['repo']}"
        if context_info.get("scope"):
            context_str += f" Scope: {context_info['scope']}"

        prompt = (f"{sys_prompt}\n\n"
                 f"Original question: {original_query}{context_str}\n\n"
                 f"Examples for 'authentication cache':\n"
                 f"[\"OAuth2 configuration caching\", \"security settings cache mechanism\", "
                 f"\"Spring Security cache authentication\", \"authentication token caching\"]\n\n"
                 f"JSON array:")

        try:
            response = self.llm.complete(prompt).text.strip()
            # Extract JSON array from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                queries = json.loads(response[start:end])
                return [q for q in queries if isinstance(q, str) and q.strip()]
        except Exception as e:
            logging.warning(f"Query expansion failed: {e}")

        # Fallback: basic keyword expansion
        fallbacks = []
        query_lower = original_query.lower()

        # Technical synonyms
        if "auth" in query_lower or "login" in query_lower:
            fallbacks.extend(["authentication mechanism", "security configuration", "OAuth2 setup"])
        if "cache" in query_lower or "caching" in query_lower:
            fallbacks.extend(["caching strategy", "cache configuration", "data caching implementation"])
        if "config" in query_lower or "configuration" in query_lower:
            fallbacks.extend(["application settings", "environment configuration", "setup parameters"])

        return fallbacks[:3] if fallbacks else [original_query]
    def __init__(self, namespace: Optional[str] = None, max_iters: int = 3,
                 llm: Optional[Any] = None, progress_cb: Optional[Callable[[dict], None]] = None):
        self.llm = llm or QwenLLM()
        self.namespace = namespace or DEFAULT_NAMESPACE
        self.max_iters = max_iters
        self._progress_cb = progress_cb

        factory = make_graph_retriever_factory(
            hosts=CASSANDRA_HOST,
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
            "project": factory.for_repo(k=10, start_k=2, max_depth=2),
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
                elif isinstance(v, list) and v:
                    # Handle LLM returning arrays like "repos": ["cache-comparison"] 
                    # Convert to singular form
                    singular_k = k.rstrip('s') if k.endswith('s') else k
                    filters[singular_k] = v[0]  # Take first item
        except Exception as e:
            logging.warning(f"Failed to parse scope planning response: {e}")
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
        attempt = state.get("attempt", 0)

        # Try original query first
        docs = retriever.invoke(q, filter=filters) or []
        original_count = len(docs)

        # If we have few results or this is a retry, try semantic expansion
        if len(docs) < 3 or attempt > 0:
            context_info = {"repo": filters.get("repo"), "scope": scope}
            expanded_queries = self._expand_query_semantically(q, context_info)

            all_docs = list(docs)  # Start with original results
            seen_content = set()  # Deduplicate by content

            for doc in docs:
                content_hash = hash(getattr(doc, "page_content", "") or getattr(doc, "text", ""))
                seen_content.add(content_hash)

            for exp_query in expanded_queries:
                if len(all_docs) >= ROUTER_TOP_K:
                    break
                try:
                    exp_docs = retriever.invoke(exp_query, filter=filters) or []
                    for doc in exp_docs:
                        if len(all_docs) >= ROUTER_TOP_K:
                            break
                        content_hash = hash(getattr(doc, "page_content", "") or getattr(doc, "text", ""))
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            seen_content.add(content_hash)
                except Exception as e:
                    logging.warning(f"Expanded query '{exp_query}' failed: {e}")
                    continue

            docs = all_docs[:ROUTER_TOP_K]

            if len(docs) > original_count:
                self._notify({"stage": "retrieve_expanded", "original_hits": original_count, 
                             "expanded_hits": len(docs), "expanded_queries": expanded_queries})

        # Score and sort by relevance if available
        scored_docs = []
        for doc in docs:
            score = _score_of(doc)
            scored_docs.append((score or 0.0, doc))

        # Sort by score (higher is better for most similarity metrics)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        docs = [doc for _, doc in scored_docs]

        dbg = dict(state.get("debug") or {})
        dbg.setdefault("turns", []).append({
            "stage": "retrieve", "scope": scope, "filters": dict(filters), 
            "hits": len(docs), "original_hits": original_count, "attempt": attempt
        })
        self._notify({"stage": "retrieve", "scope": scope, "filters": dict(filters), "hits": len(docs)})
        return {**state, "docs": docs, "debug": dbg}

    def judge(self, state: AgentState) -> AgentState:
        q = state["query"]
        docs = state.get("docs") or []

        # Build richer inventory including content snippets for better judgment
        inv = []
        for i, d in enumerate(docs, start=1):
            md = getattr(d, "metadata", {}) or {}
            content = getattr(d, "page_content", "") or getattr(d, "text", "")
            # Include first 200 chars of content for semantic relevance assessment
            content_preview = content[:200] + "..." if len(content) > 200 else content
            inv.append({
                "i": i, 
                "repo": md.get("repo", ""), 
                "module": md.get("module", ""),
                "file": md.get("file_path", ""), 
                "topics": md.get("topics", ""),
                "content_preview": content_preview,
                "relevance_score": _score_of(d)
            })

        rubric = (
            "Judge if the retrieved content is semantically relevant and sufficient to answer the question. "
            "Consider both metadata relevance AND content preview relevance. Return JSON: "
            "{coverage:0..1, needs_more:boolean, suggest_filters?:{repo?,module?,topics?}, "
            "stage_down?: 'package'|'file'|'code'|null, rewrite?:string, semantic_match:boolean}"
        )

        context_quality = "good" if any(inv) else "empty"
        if inv and all(not item.get("content_preview", "").strip() for item in inv):
            context_quality = "metadata_only"
        elif inv and any("auth" in item.get("content_preview", "").lower() or 
                         "cache" in item.get("content_preview", "").lower() for item in inv):
            context_quality = "semantically_relevant"

        msg = (f"{rubric}\n\nQuestion: {q}\n"
               f"Context quality: {context_quality}\n"
               f"Retrieved items: {json.dumps(inv, ensure_ascii=False)}\nJSON:")
        try:
            raw = self.llm.complete(msg).text.strip()
            raw = raw[raw.find("{"): raw.rfind("}")+1]
            data = json.loads(raw)
        except Exception as e:
            logging.warning(f"Failed to parse judge response: {e}")
            # Auto-stage down if we have insufficient high-level context
            curr_scope = state["scope"]
            if curr_scope == "project":
                data = {"coverage": 0.2, "needs_more": True, "stage_down": "package"}
            elif curr_scope == "package":
                data = {"coverage": 0.3, "needs_more": True, "stage_down": "file"}
            else:
                data = {"coverage": 0.4, "needs_more": False}

        filters = dict(state.get("filters") or {})
        # Handle both string and array responses from LLM
        for k, v in (data.get("suggest_filters") or {}).items():
            if isinstance(v, str) and v:
                filters[k] = v
            elif isinstance(v, list) and v:
                singular_k = k.rstrip('s') if k.endswith('s') else k
                filters[singular_k] = v[0]

        # Auto-stage down if coverage is too low and we haven't hit max attempts
        next_scope = state["scope"]
        stage_down = data.get("stage_down")
        if stage_down in {"package", "file", "code"}:
            next_scope = stage_down
        elif data.get("coverage", 0) < 0.3 and len(docs) > 0:
            # Auto-progression through scopes when context is insufficient
            if state["scope"] == "project":
                next_scope = "package"
            elif state["scope"] == "package":
                next_scope = "file"
            elif state["scope"] == "file":
                next_scope = "code"

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
            logging.info(f"Max iterations ({self.max_iters}) reached, ending search")
            return {**state, "needs_more": False, "attempt": attempt}

        # Check if we're stuck getting the same results
        docs = state.get("docs", [])
        if attempt > 1 and len(docs) > 0:
            # If we keep getting repo-level summaries, force drill-down
            all_repo_level = all(not getattr(d, "metadata", {}).get("file_path") for d in docs)
            if all_repo_level and state.get("scope") in ["project", "package"]:
                logging.info("Forcing drill-down to file level due to insufficient specific context")
                return {**state, "scope": "file", "attempt": attempt}

        base_query = state.get("rewrite") or state["query"]
        filters = state.get("filters") or {}

        # Generate a more focused semantic query instead of keyword stuffing
        if attempt == 1:
            # First retry: expand with context
            context_parts = []
            if "repo" in filters:   
                context_parts.append(filters['repo'])
            if "module" in filters: 
                context_parts.append(filters['module'])

            # Use LLM to create a better semantic query
            context_str = " ".join(context_parts) if context_parts else ""
            rewrite_prompt = (
                f"Rewrite this codebase question to be more specific and searchable: '{base_query}'"
                f"{' Context: ' + context_str if context_str else ''}"
                f"\nReturn only the rewritten question, no explanation:"
            )

            try:
                sharpened = self.llm.complete(rewrite_prompt).text.strip()
                # Clean up any quotes or extra formatting
                sharpened = sharpened.strip('"\'').strip()
                if not sharpened or len(sharpened) < 10:
                    raise ValueError("Rewrite too short")
            except Exception as e:
                logging.warning(f"LLM rewrite failed: {e}")
                # Fallback to context-aware manual rewrite
                parts = [base_query]
                if context_str:
                    parts.append(f"in {context_str}")
                sharpened = " ".join(parts)
        else:
            # Later retries: use semantic expansion
            expanded_queries = self._expand_query_semantically(base_query, {
                "repo": filters.get("repo"), "scope": state.get("scope")
            })
            sharpened = expanded_queries[0] if expanded_queries else base_query
        dbg = dict(state.get("debug") or {})
        dbg.setdefault("turns", []).append({"stage": "rewrite", "attempt": attempt+1, "query": sharpened, "filters": dict(filters)})
        self._notify({"stage": "rewrite", "action": "retry", "attempt": attempt + 1,
                      "query": sharpened, "filters": dict(filters)})
        return {**state, "query": sharpened, "attempt": attempt, "debug": dbg}

    def synthesize(self, state: AgentState) -> AgentState:
        q = state["query"]
        docs = state.get("docs") or []
        # Build blocks and a sources list (with scores if present)
        # Limit context to prevent overwhelming the LLM
        max_blocks = min(5, len(docs))  # Max 5 blocks
        blocks, sources = [], []
        for i, d in enumerate(docs[:max_blocks], start=1):
            md = getattr(d, "metadata", {}) or {}
            text = getattr(d, "page_content", "") or getattr(d, "text", "")
            # Shorter text chunks for focused context
            text_snippet = text[:800] if len(text) > 800 else text
            blocks.append(f"[{i}] repo={md.get('repo','')} module={md.get('module','')} file={md.get('file_path','')}\n{text_snippet}")
            sources.append(_doc_to_source(i, d))

        # Assess question type and context quality
        question_type = "overview" if any(word in q.lower() for word in ["projects", "repositories", "overview", "tell me about", "what is", "describe"]) else "specific"
        has_content = len([b for b in blocks if len(b.split('\n', 1)[-1].strip()) > 50]) > 0

        if question_type == "overview" and has_content:
            sys = ("You are a senior developer assistant. Use the provided context blocks to give a comprehensive answer. "
                   "Cite sources as [1], [2], etc. Synthesize information across blocks when relevant. "
                   "If the question asks for an overview of available projects/repositories, describe what you see in the context.")
        else:
            sys = ("You are a senior developer assistant. Answer using the provided context blocks. "
                   "Cite blocks as [1], [2]. If the specific information needed is not in the context, "
                   "say so clearly and suggest looking in specific repos/modules that might contain the answer.")

        prompt = f"{sys}\n\nQuestion: {q}\n\nContext:\n" + "\n\n".join(blocks) + "\n\nAnswer:"

        try:
            text = self.llm.complete(prompt).text

            # If LLM claims insufficient context despite having good retrieval, try again with more encouraging prompt
            if (has_content and len(docs) >= 3 and 
                any(phrase in text.lower() for phrase in ["insufficient", "don't see enough", "can't answer", "not enough information"])):

                retry_sys = ("You are a helpful developer assistant. The user is asking about available projects. "
                           "Use the context provided to describe the projects you can see. Don't be overly conservative - "
                           "if you have project descriptions, share them! Cite sources as [1], [2].")
                retry_prompt = f"{retry_sys}\n\nQuestion: {q}\n\nContext:\n" + "\n\n".join(blocks) + "\n\nAnswer:"

                try:
                    retry_text = self.llm.complete(retry_prompt).text
                    if not any(phrase in retry_text.lower() for phrase in ["insufficient", "don't see enough", "can't answer"]):
                        text = retry_text
                        logging.info("Synthesis retry successful - overcame conservative response")
                except Exception as retry_e:
                    logging.warning(f"Synthesis retry failed: {retry_e}")

        except Exception as e:
            text = f"(LLM error) {e}"

        dbg = dict(state.get("debug") or {})
        dbg["final_ctx_blocks"] = len(blocks)
        dbg["sources_count"] = len(sources)
        dbg["final_scope"] = state.get("scope", "")
        dbg["question_type"] = question_type if 'question_type' in locals() else "unknown"
        dbg["has_content"] = has_content if 'has_content' in locals() else False
        dbg["answer_length"] = len(text)

        # Check if answer seems insufficient despite good context
        answer_seems_insufficient = any(phrase in text.lower() for phrase in ["insufficient", "don't see enough", "can't answer"])
        if answer_seems_insufficient and has_content and len(docs) >= 3:
            dbg["synthesis_issue"] = "LLM_overly_conservative"

        self._notify({"stage": "synthesize", "final_ctx_blocks": len(blocks), "sources_count": len(sources), 
                     "answer_length": len(text), "synthesis_issue": dbg.get("synthesis_issue")})
        return {**state, "answer": text, "debug": dbg, "docs": docs, "sources": sources}

    # ---------- Router for conditional edge ----------

    def _route_from_decision(self, state: AgentState):
        if state.get("needs_more"):
            return "retry"
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