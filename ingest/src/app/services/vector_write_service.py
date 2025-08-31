# app/services/vector_write_service.py
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Iterable, Tuple
from itertools import islice

from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Cassandra as LCCassandra
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_graph_retriever.transformers import ShreddingTransformer

from app.config import SETTINGS  # expects: cassandra_keyspace, embed_model, embeddings_table_* names
from app.services.cassandra_service import CassandraService  # provides .connect().session
from llama_index.core.schema import BaseNode  # only for input typing


class VectorWriteService:
    """
    Drop-in replacement that:
      - converts LlamaIndex Nodes -> LangChain Documents
      - shreds list metadata (topics/imports/labels) for GraphRAG traversal
      - writes to Cassandra 5 via LangChain's vector store (CassIO backend)
    NOTE: The 'stores' argument from the controller is ignored (it contained LlamaIndex stores).
    """

    # Keep traversal-focused allow-lists small for faster SAI indexes
    _ALLOW_FIELDS_BY_SCOPE: Dict[str, Iterable[str]] = {
        "catalog": ("namespace", "repo", "owner", "language", "topics", "labels", "component_kind"),
        "repo":    ("namespace", "repo", "owner", "language", "topics", "labels"),
        "module":  ("namespace", "repo", "module", "language", "topics", "imports", "labels"),
        "file":    ("namespace", "repo", "module", "file_path", "language", "topics", "imports", "labels"),
        "chunk":   ("namespace", "repo", "module", "file_path", "symbol", "language", "topics", "imports"),
    }

    # Map scopes -> table names from SETTINGS (must match your controller’s config)
    _TABLE_BY_SCOPE: Dict[str, str] = {
        "catalog": getattr(SETTINGS, "embeddings_table_catalog", "embeddings_catalog"),
        "repo":    getattr(SETTINGS, "embeddings_table_repo",    "embeddings_repo"),
        "module":  getattr(SETTINGS, "embeddings_table_module",  "embeddings_module"),
        "file":    getattr(SETTINGS, "embeddings_table_file",    "embeddings_file"),
        "chunk":   getattr(SETTINGS, "embeddings_table_chunk",   "embeddings"),  # legacy/current
    }

    @staticmethod
    def write_nodes_per_scope(
            *,
            embedder: Any,  # kept for signature compatibility; not used (we instantiate LC embeddings)
            stores: Dict[str, Any],  # kept for signature compatibility; ignored (LLMIndex stores)
            catalog_nodes: List[BaseNode],
            repo_nodes: List[BaseNode],
            module_nodes: List[BaseNode],
            file_nodes: List[BaseNode],
            chunk_nodes: List[BaseNode],
            batch_size: int = 128,
    ) -> None:
        """Embed + write per-scope using LangChain Cassandra store with shredding."""
        # 1) Cassandra session & embeddings
        cass = CassandraService()
        session = cass.connect().session  # uses SETTINGS for contact points/keyspace elsewhere
        keyspace = SETTINGS.cassandra_keyspace
        emb = HuggingFaceEmbeddings(model_name=SETTINGS.embed_model)
        shredder = ShreddingTransformer()

        # 2) Scope -> nodes map (controller already separates these)
        scope_nodes: List[Tuple[str, List[BaseNode]]] = [
            ("catalog", catalog_nodes),
            ("repo",    repo_nodes),
            ("module",  module_nodes),
            ("file",    file_nodes),
            ("chunk",   chunk_nodes),
        ]

        for scope, nodes in scope_nodes:
            if not nodes:
                continue

            table = VectorWriteService._TABLE_BY_SCOPE[scope]
            allow_fields = tuple(VectorWriteService._ALLOW_FIELDS_BY_SCOPE[scope])

            # Create (or reuse) the LC Cassandra vector store for this scope
            store = LCCassandra(
                embedding=emb,
                session=session,
                keyspace=keyspace,
                table_name=table,
                # index only the fields we traverse on; keeps SAI indexes tight
                metadata_indexing=("allow", list(allow_fields)),
            )

            logging.info(f"📝 Writing {len(nodes)} {scope} nodes to {keyspace}.{table}")
            # 3) Convert to LC Documents and write in batches (with shredding)
            docs, ids = VectorWriteService._nodes_to_docs(scope, nodes)
            if not docs:
                continue

            # Shred list metadata so edges over lists are traversable by GraphRAG
            docs = shredder.transform_documents(docs)

            for batch_docs, batch_ids in _batched(docs, ids, batch_size=batch_size):
                store.add_documents(batch_docs, ids=batch_ids)

            logging.info(f"✅ Finished writing {len(nodes)} {scope} nodes to {keyspace}.{table}")

    # ---------- helpers ----------

    @staticmethod
    def _nodes_to_docs(scope: str, nodes: List[BaseNode]) -> Tuple[List[LCDocument], List[str]]:
        """Convert LlamaIndex nodes to LangChain Documents, preserving metadata."""
        out_docs: List[LCDocument] = []
        out_ids: List[str] = []

        for n in nodes:
            # Content: prefer get_content(); fallback to .text
            try:
                content = n.get_content(metadata_mode="none")  # v0.10+
            except Exception:
                content = getattr(n, "text", None) or ""

            md = dict(getattr(n, "metadata", {}) or {})
            # Ensure scope is set (controller usually sets via _attach_common_metadata)
            md.setdefault("scope", scope)

            # Normalize a few common keys
            if "path" in md and "file_path" not in md:
                md["file_path"] = md["path"]

            # Id: prefer node_id attribute; fallback to provided hash/id
            node_id = getattr(n, "node_id", None) or md.get("id") or md.get("doc_id") or None
            if not node_id:
                # last resort: derive from stable fields (scope+repo+file_path+start/ end lines if present)
                seed = f"{scope}|{md.get('namespace','')}|{md.get('repo','')}|{md.get('module','')}|{md.get('file_path','')}|{md.get('start', '')}|{md.get('end','')}"
                # short stable hash
                import hashlib
                node_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()

            out_docs.append(LCDocument(page_content=content or "", metadata=md))
            out_ids.append(str(node_id))

        return out_docs, out_ids


def _batched(docs: List[LCDocument], ids: List[str], batch_size: int):
    """Yield (docs_batch, ids_batch) tuples."""
    it_docs = iter(docs)
    it_ids = iter(ids)
    while True:
        d_batch = list(islice(it_docs, batch_size))
        i_batch = list(islice(it_ids, batch_size))
        if not d_batch:
            break
        yield d_batch, i_batch