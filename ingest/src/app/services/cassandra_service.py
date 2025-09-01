# app/services/cassandra_service.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session

import cassio
from langchain_community.vectorstores import Cassandra as LcCassandra
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import SETTINGS

try:
    # Optional: shared default if your SETTINGS lacks an embed model
    from rag_shared.config import EMBED_MODEL as DEFAULT_EMBED_MODEL
except Exception:
    DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Simple handles structure
# -----------------------------
@dataclass
class CassandraHandles:
    cluster: Cluster
    session: Session


# -----------------------------
# Adapter around LC Cassandra
# -----------------------------
class CassandraVectorStoreAdapter:
    """
    Thin wrapper around LangChain's Cassandra vector store so the rest of your
    code can keep calling familiar methods.
    """
    def __init__(
            self,
            lc_store: LcCassandra,
            *,
            id_key: str = "doc_id",
            text_key: str = "text",
            metadata_key: str = "metadata",
    ):
        self._store = lc_store
        self.id_key = id_key
        self.text_key = text_key
        self.metadata_key = metadata_key

    # ---- Writes ----
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[Iterable[Dict[str, Any]]] = None,
            ids: Optional[Iterable[str]] = None,
            **kwargs,
    ) -> List[str]:
        return self._store.add_texts(
            list(texts),
            metadatas=list(metadatas or []),
            ids=list(ids or []),
        )

    def add_documents(
            self,
            docs: Iterable[Document],
            ids: Optional[Iterable[str]] = None,
            **kwargs,
    ) -> List[str]:
        return self._store.add_documents(list(docs), ids=list(ids or []))

    # ---- Reads ----
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        return self._store.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs):
        return self._store.as_retriever(**kwargs)

    @property
    def raw(self) -> LcCassandra:
        return self._store


class CassandraService:
    """
    One-stop shop for:
      - Connecting to Cassandra
      - Initializing cassio
      - Building a reusable HuggingFaceEmbeddings
      - Returning LC/CassIO vector stores via an adapter

    Usage:
        cass = CassandraService()
        cass.connect()  # sets self.cluster/self.session and keyspace
        stores = {
            "catalog": cass.vector_store(table=SETTINGS.embeddings_table_catalog),
            "repo":    cass.vector_store(table=SETTINGS.embeddings_table_repo),
            "module":  cass.vector_store(table=SETTINGS.embeddings_table_module),
            "file":    cass.vector_store(table=SETTINGS.embeddings_table_file),
            "chunk":   cass.vector_store(table=SETTINGS.embeddings_table_chunk),
        }
    """

    cluster: Optional[Cluster] = None
    session: Optional[Session] = None

    def __init__(self) -> None:
        self._settings = SETTINGS

        # pick an embedding model from SETTINGS, with reasonable fallbacks
        embed_model = (
                getattr(self._settings, "embedding_model_name", None)
                or getattr(self._settings, "embed_model", None)
                or DEFAULT_EMBED_MODEL
        )

        # Build once; reuse everywhere
        self._embeddings: Embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    # ---------- Connection / init ----------
    def connect(self) -> CassandraHandles:
        """Create cluster + session, ensure keyspace, init cassio, set default keyspace."""
        auth = PlainTextAuthProvider(
            username=self._settings.cassandra_username,
            password=self._settings.cassandra_password,
        )
        cluster = Cluster(
            [self._settings.cassandra_host],
            port=self._settings.cassandra_port,
            auth_provider=auth,
        )
        session = cluster.connect()

        # Keyspace (dev default: SimpleStrategy; tune for prod/replication)
        session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self._settings.cassandra_keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
            """
        )
        session.set_keyspace(self._settings.cassandra_keyspace)
        logging.info("Connected to Cassandra keyspace=%s", self._settings.cassandra_keyspace)

        # save instance handles
        self.cluster = cluster
        self.session = session

        # make cassio aware of this session (self-hosted path)
        cassio.init(session=session)

        return CassandraHandles(cluster=cluster, session=session)

    def close(self) -> None:
        if self.session:
            try:
                self.session.shutdown()
            except Exception:
                pass
            self.session = None
        if self.cluster:
            try:
                self.cluster.shutdown()
            except Exception:
                pass
            self.cluster = None

    # ---------- Vector store ----------
    def vector_store(
            self,
            session: Optional[Session] = None,
            *,
            table: str,
    ) -> CassandraVectorStoreAdapter:
        """
        Return a vector store bound to the given table (created if needed).
        Backwards compatible: you may pass a session explicitly, or rely on the
        one established by connect().
        """
        sess = session or self.session
        if sess is None:
            raise RuntimeError("No Cassandra session. Call connect() or pass `session=`.")

        lc = LcCassandra(
            embedding=self._embeddings,
            session=sess,
            keyspace=self._settings.cassandra_keyspace,
            table_name=table
        )
        return CassandraVectorStoreAdapter(lc)

    # ---------- Utilities ----------
    def count_rows(self, table: str, session: Optional[Session] = None) -> int:
        """
        Count rows in a specific table. (Note: COUNT(*) can be heavy on large tables.)
        """
        sess = session or self.session
        if sess is None:
            raise RuntimeError("No Cassandra session. Call connect() or pass `session=`.")

        rs = sess.execute(f"SELECT COUNT(*) FROM {self._settings.cassandra_keyspace}.{table}")
        row = next(iter(rs), None)
        return int(getattr(row, "count", 0) if row is not None else 0)
