# graph_rag_retrievers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# LangChain vector store for Cassandra (CassIO-backed)
from langchain_community.vectorstores import Cassandra as LCCassandra
from langchain_huggingface import HuggingFaceEmbeddings
# GraphRAG: retriever, strategy, and adapter
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager, MMR
from langchain_graph_retriever.adapters.cassandra import CassandraAdapter
from langchain_graph_retriever.transformers import ShreddingTransformer

import rag_shared.config
from rag_shared import config


# ---------- Configuration ----------

@dataclass(frozen=True)
class CassandraConn:
    hosts: Sequence[str] = ("localhost",)
    port: int = 9042
    username: Optional[str] = None
    password: Optional[str] = None
    keyspace: str = "code_index"

@dataclass(frozen=True)
class TableNames:
    repo: str = "repo_overview_vectors"
    module: str = "module_summary_vectors"
    file: str = "file_summary_vectors"
    chunk: str = "code_chunk_vectors"

@dataclass(frozen=True)
class RetrieverConfig:
    conn: CassandraConn
    tables: TableNames
    # Embedding model must match your ingestion dimension
    hf_embedding_model: str = rag_shared.config.EMBED_MODEL
    # If you want to restrict which metadata fields get indexed in Cassandra,
    # the LangChain Cassandra VS supports allow/deny lists. We default to allow-just-needed.
    index_fields_by_table: Dict[str, Iterable[str]] = None

    def with_defaults(self) -> "RetrieverConfig":
        if self.index_fields_by_table is not None:
            return self
        # Fields we’ll traverse on; keep this tight for performance.
        index_fields_by_table = {
            "repo": {"namespace", "repo", "owner", "language", "topics", "labels"},
            "module": {"namespace", "repo", "module", "language", "topics", "imports", "labels"},
            "file": {"namespace", "repo", "module", "file_path", "language", "topics", "imports", "labels"},
            "chunk": {"namespace", "repo", "module", "file_path", "symbol", "language", "topics", "imports"},
        }
        return RetrieverConfig(
            conn=self.conn,
            tables=self.tables,
            hf_embedding_model=self.hf_embedding_model,
            index_fields_by_table=index_fields_by_table,
        )

# ---------- Factory ----------

class GraphRetrieverFactory:
    """
    Creates GraphRAG GraphRetrievers per summary level (repo/module/file/chunk)
    against Cassandra 5 vector tables. Designed for use by your LLM-as-Judge router.
    """

    def __init__(self, cfg: RetrieverConfig):
        cfg = cfg.with_defaults()
        self.cfg = cfg

        # 1) Cassandra session
        auth = None
        if cfg.conn.username and cfg.conn.password:
            auth = PlainTextAuthProvider(cfg.conn.username, cfg.conn.password)
        self._cluster = Cluster(list(cfg.conn.hosts), port=cfg.conn.port, auth_provider=auth)
        self._session = self._cluster.connect()
        self._session.set_keyspace(cfg.conn.keyspace)

        # 2) Embeddings (must match the dimension used at ingest time)
        self._emb = HuggingFaceEmbeddings(model_name=cfg.hf_embedding_model)

        # 3) Create vector stores per table (LangChain Cassandra VS)
        self._stores = {
            "repo": self._make_store(cfg.tables.repo, cfg.index_fields_by_table["repo"]),
            "module": self._make_store(cfg.tables.module, cfg.index_fields_by_table["module"]),
            "file": self._make_store(cfg.tables.file, cfg.index_fields_by_table["file"]),
            "chunk": self._make_store(cfg.tables.chunk, cfg.index_fields_by_table["chunk"]),
        }

        # 4) Adapters with shredding support for list metadata traversal
        #    (Cassandra requires shredding to traverse list fields as edges)
        shredder = ShreddingTransformer()
        self._adapters = {
            name: CassandraAdapter(store, shredder=shredder) for name, store in self._stores.items()
        }

        # 5) Edge specs per level. Edges are defined as pairs of metadata fields; equal values link nodes.
        #    "$id" is also supported if you stored parent/child IDs in metadata. :contentReference[oaicite:1]{index=1}
        self._edges = {
            # Repo-level: connect by namespace, owner, dominant language, topics/labels
            "repo": [
                ("namespace", "namespace"),
                ("owner", "owner"),
                ("language", "language"),
                ("topics", "topics"),      # requires shredding to traverse lists
                ("labels", "labels"),      # requires shredding
            ],
            # Module-level: cluster by repo/module + topical relations
            "module": [
                ("repo", "repo"),
                ("module", "module"),
                ("language", "language"),
                ("topics", "topics"),      # requires shredding
                ("imports", "imports"),    # requires shredding if list of modules
                ("labels", "labels"),
            ],
            # File-level: link by file_path/module/repo + topics/imports
            "file": [
                ("repo", "repo"),
                ("module", "module"),
                ("file_path", "file_path"),
                ("language", "language"),
                ("topics", "topics"),      # requires shredding
                ("imports", "imports"),    # requires shredding
            ],
            # Chunk-level: fine-grained links by file_path/module/symbol/topic/imports
            "chunk": [
                ("repo", "repo"),
                ("module", "module"),
                ("file_path", "file_path"),
                ("symbol", "symbol"),
                ("language", "language"),
                ("topics", "topics"),      # requires shredding
                ("imports", "imports"),    # requires shredding
            ],
        }

    # ----- public: build retrievers for your router -----

    def for_repo(self, *, k:int=6, start_k:int=2, max_depth:int=2) -> GraphRetriever:
        # Eager strategy: broaden quickly for “what repos/modules do you know?”-type queries. :contentReference[oaicite:2]{index=2}
        return self._mk("repo", Eager(k=k, start_k=start_k, max_depth=max_depth))

    def for_module(self, *, k:int=8, start_k:int=2, adjacent_k:int=6, max_depth:int=2) -> GraphRetriever:
        # MMR strategy: balances relevance/diversity for module discovery/debug. :contentReference[oaicite:3]{index=3}
        return self._mk("module", MMR(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth, lambda_mult=0.4))

    def for_file(self, *, k:int=8, start_k:int=2, adjacent_k:int=6, max_depth:int=2) -> GraphRetriever:
        return self._mk("file", MMR(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth, lambda_mult=0.35))

    def for_chunk(self, *, k:int=10, start_k:int=3, adjacent_k:int=8, max_depth:int=2) -> GraphRetriever:
        # Deep dive on code chunks (debugging paths)
        return self._mk("chunk", MMR(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth, lambda_mult=0.3))

    # ----- internals -----

    def _make_store(self, table_name: str, allow_fields: Iterable[str]) -> LCCassandra:
        # Limit indexed metadata to just the fields we traverse on for speed.
        # LangChain Cassandra VS supports `metadata_indexing=("allow", {...})`. :contentReference[oaicite:4]{index=4}
        return LCCassandra(
            embedding=self._emb,
            session=self._session,
            keyspace=self.cfg.conn.keyspace,
            table_name=table_name,
            metadata_indexing=("allow", list(allow_fields)),

        )

    def _mk(self, level: str, strategy) -> GraphRetriever:
        adapter = self._adapters[level]
        edges = self._edges[level]
        # GraphRetriever accepts either a VectorStore or an Adapter; we pass the adapter to retain shredding hooks. :contentReference[oaicite:5]{index=5}
        return GraphRetriever(store=adapter, edges=edges, strategy=strategy)

    # optional: close cluster
    def close(self):
        try:
            self._cluster.shutdown()
        except Exception:
            pass


# ---------- Convenience: one-liner bootstrap ----------

def make_graph_retriever_factory(
        *,
        hosts: Sequence[str],
        keyspace: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tables: Optional[TableNames] = None,
        embedding_model: str = config.EMBED_MODEL,
) -> GraphRetrieverFactory:
    tables = tables or TableNames()
    cfg = RetrieverConfig(
        conn=CassandraConn(
            hosts=hosts, port=9042, username=username, password=password, keyspace=keyspace
        ),
        tables=tables,
        hf_embedding_model=embedding_model,
    )
    return GraphRetrieverFactory(cfg)
