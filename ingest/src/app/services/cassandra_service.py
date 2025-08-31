from __future__ import annotations
import logging
from dataclasses import dataclass
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index.vector_stores.cassandra import CassandraVectorStore
from app.config import SETTINGS


@dataclass
class CassandraHandles:
    cluster: Cluster
    session: "Session"


class CassandraService:
    def __init__(self):
        self._settings = SETTINGS

    def connect(self) -> CassandraHandles:
        auth = PlainTextAuthProvider(
            username=self._settings.cassandra_username,
            password=self._settings.cassandra_password,
        )
        cluster = Cluster([self._settings.cassandra_host], port=self._settings.cassandra_port, auth_provider=auth)
        session = cluster.connect()
        session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self._settings.cassandra_keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
            """
        )
        session.set_keyspace(self._settings.cassandra_keyspace)
        # Audit table
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_runs (
                                                       run_id UUID PRIMARY KEY,
                                                       namespace TEXT,
                                                       repo TEXT,
                                                       branch TEXT,
                                                       collection TEXT,
                                                       component_kind TEXT,
                                                       started_at TIMESTAMP,
                                                       finished_at TIMESTAMP,
                                                       node_count INT
            )
            """
        )
        logging.info("Connected to Cassandra keyspace=%s", self._settings.cassandra_keyspace)
        return CassandraHandles(cluster=cluster, session=session)

    def vector_store(self, session, *, table: str) -> CassandraVectorStore:
        """Return a vector store bound to the given table (created if needed)."""
        return CassandraVectorStore(
           session=session,
            table=table,
            embedding_dimension=self._settings.embed_dim,
            keyspace=self._settings.cassandra_keyspace,
        )

    def count_rows_total(self, session) -> int:
        rs = session.execute(
            f"SELECT COUNT(*) FROM {self._settings.cassandra_keyspace}.{self._settings.embeddings_table}"
        )
        # ensure concrete value
        row = list(rs)[0]
        return int(getattr(row, "count", 0))