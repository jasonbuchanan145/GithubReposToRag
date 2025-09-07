from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain_community.vectorstores import Cassandra as LCCassandra
from langchain_huggingface import HuggingFaceEmbeddings

# GraphRAG imports
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

# ----- Table name holder -------------------------------------------------------

@dataclass(frozen=True)
class TableNames:
    repo: str
    module: str
    file: str
    chunk: str

# ----- Factory ----------------------------------------------------------------

class GraphRetrieverFactory:
    """
    Builds GraphRetriever instances (LangChain retriever) for each scope over
    Cassandra tables using the same embedder used at ingestion time.

    Usage:
        factory = GraphRetrieverFactory(hosts, keyspace, username, password, tables, embedding_model="intfloat/e5-small-v2")
        retr = factory.for_repo(k=6, start_k=2, max_depth=2)
        docs = retr.invoke("show me repos using activemq", filter={"namespace": "acme"})
    """

    def __init__(
            self,
            *,
            hosts: list[str],
            keyspace: str,
            username: Optional[str],
            password: Optional[str],
            tables: TableNames,
            embedding_model: str = "intfloat/e5-small-v2",
    ) -> None:
        self.hosts = hosts
        self.keyspace = keyspace
        self.username = username
        self.password = password
        self.tables = tables

        # Use the same embedding family/dim used to create the tables (384-d for e5-small-v2)
        self._emb = HuggingFaceEmbeddings(model_name=embedding_model)

        # Connect a single shared session
        auth = None
        if username:
            auth = PlainTextAuthProvider(username=username, password=password or "")
        self._cluster = Cluster(self.hosts, auth_provider=auth)
        self._session = self._cluster.connect()
        # Keyspace must already exist (you created vector_store and tables)

        # Cache vector stores per table name
        self._stores: Dict[str, LCCassandra] = {}

    # --------- internal helpers ---------

    def _store(self, table_name: str) -> LCCassandra:
        st = self._stores.get(table_name)
        if st:
            return st
        st = LCCassandra(
            embedding=self._emb,
            session=self._session,
            keyspace=self.keyspace,
            table_name=table_name,
            # We rely on your existing SAI index + vector schema
        )
        self._stores[table_name] = st
        return st

    @staticmethod
    def _edges_for(scope: str):
        """
        Edges tell GraphRetriever which metadata fields define connectivity.
        We traverse by shared values of these keys.

        - repo scope: mostly high-level hops by namespace/repo/topics
        - module scope: tighten to repo/module/topics
        - file scope: repo/module/file_path/topics
        - code/chunk scope: include file_path for chunk↔file clustering
        """
        if scope == "project":
            return [("namespace", "namespace"), ("repo", "repo"), ("topics", "topics")]
        if scope == "package":
            return [("namespace", "namespace"), ("repo", "repo"), ("module", "module"), ("topics", "topics")]
        if scope == "file":
            return [("namespace", "namespace"), ("repo", "repo"), ("module", "module"), ("file_path", "file_path"), ("topics", "topics")]
        # "code"
        return [("namespace", "namespace"), ("repo", "repo"), ("module", "module"), ("file_path", "file_path"), ("topics", "topics")]

    # --------- public builders (return LangChain retrievers) ---------

    def for_repo(self, *, k: int = 6, start_k: int = 2, max_depth: int = 2):
        store = self._store(self.tables.repo)
        return GraphRetriever(
            store=store,
            edges=self._edges_for("project"),
            strategy=Eager(k=k, start_k=start_k, max_depth=max_depth),
        )

    def for_module(self, *, k: int = 8, start_k: int = 2, adjacent_k: int = 6, max_depth: int = 2):
        store = self._store(self.tables.module)
        return GraphRetriever(
            store=store,
            edges=self._edges_for("package"),
            strategy=Eager(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth),
        )

    def for_file(self, *, k: int = 8, start_k: int = 2, adjacent_k: int = 6, max_depth: int = 2):
        store = self._store(self.tables.file)
        return GraphRetriever(
            store=store,
            edges=self._edges_for("file"),
            strategy=Eager(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth),
        )

    def for_chunk(self, *, k: int = 10, start_k: int = 3, adjacent_k: int = 8, max_depth: int = 2):
        store = self._store(self.tables.chunk)
        return GraphRetriever(
            store=store,
            edges=self._edges_for("code"),
            strategy=Eager(k=k, start_k=start_k, adjacent_k=adjacent_k, max_depth=max_depth),
        )

    # Optional: if you want to close cleanly in tests
    def close(self):
        try:
            self._session.shutdown()
        finally:
            self._cluster.shutdown()

# Convenience creator so your agent code can keep a single import
def make_graph_retriever_factory(
        *,
        hosts: list[str],
        keyspace: str,
        username: Optional[str],
        password: Optional[str],
        tables: TableNames,
        embedding_model: str,
) -> GraphRetrieverFactory:
    return GraphRetrieverFactory(
        hosts=hosts,
        keyspace=keyspace,
        username=username,
        password=password,
        tables=tables,
        embedding_model=embedding_model,
    )
