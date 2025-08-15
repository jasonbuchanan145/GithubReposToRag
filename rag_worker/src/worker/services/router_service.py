import logging
from typing import Dict, Optional, Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.response.schema import Response   # <-- keep only this Response

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_shared.config import (
    ROUTER_TOP_K,
    EMBED_MODEL,
    CODE_TABLE, PACKAGE_TABLE, PROJECT_TABLE,
)
from worker.services.qwen_llm import QwenLLM
from worker.services.cassandra_store import vector_store_for_table

logger = logging.getLogger(__name__)


class DirectLLMQueryEngine(BaseQueryEngine):
    """A minimal query engine that just uses the LLM without any retrieval."""

    def query(self, query_str: str, **kwargs: Any) -> Response:
        text = Settings.llm.complete(query_str).text
        return Response(text=text)

    async def aquery(self, query_str: str, **kwargs: Any) -> Response:
        if hasattr(Settings.llm, "acomplete"):
            text = (await Settings.llm.acomplete(query_str)).text
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, lambda: Settings.llm.complete(query_str).text)
        return Response(text=text)


class RouterService:
    def __init__(self):
        # Initialize global LlamaIndex settings once
        Settings.llm = QwenLLM()
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

        # Build per-level engines
        self._engines: Dict[str, RetrieverQueryEngine] = {}
        self._build_level_engine("code", CODE_TABLE)
        self._build_level_engine("package", PACKAGE_TABLE)
        self._build_level_engine("project", PROJECT_TABLE)

        # Direct LLM engine (no RAG)
        self._direct_engine = DirectLLMQueryEngine()

        # Router tools
        tools = [
            QueryEngineTool(
                query_engine=self._direct_engine,
                metadata=ToolMetadata(
                    name="no_rag",
                    description="General questions, chit-chat, or anything not about the codebase.",
                ),
            ),
            QueryEngineTool(
                query_engine=self._engines["code"],
                metadata=ToolMetadata(
                    name="code",
                    description="Fine-grained code/symbol/file questions.",
                ),
            ),
            QueryEngineTool(
                query_engine=self._engines["package"],
                metadata=ToolMetadata(
                    name="package",
                    description="Module/package-level behaviors/APIs.",
                ),
            ),
            QueryEngineTool(
                query_engine=self._engines["project"],
                metadata=ToolMetadata(
                    name="project",
                    description="Repo-wide architecture and patterns.",
                ),
            ),
        ]

        selector = LLMSingleSelector.from_defaults()
        self._router = RouterQueryEngine(
            selector=selector,
            query_engine_tools=tools,
        )

    def _build_level_engine(self, level: str, table: str) -> None:
        vstore = vector_store_for_table(table)
        index = VectorStoreIndex.from_vector_store(vstore)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=ROUTER_TOP_K)
        synth = TreeSummarize(verbose=False)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
        self._engines[level] = engine

    def route(self, query: str, *, force_level: Optional[str] = None) -> Response:
        """Route the query to the right engine. If force_level is provided, bypass routing."""
        if force_level:
            if force_level == "none":
                return self._direct_engine.query(query)
            if force_level not in self._engines:
                raise ValueError(f"Unknown force_level: {force_level}")
            return self._engines[force_level].query(query)
        return self._router.query(query)
