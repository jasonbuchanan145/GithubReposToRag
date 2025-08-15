from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from llama_index.vector_stores.cassandra import CassandraVectorStore
from rag_shared.config import (
    CASSANDRA_HOST, CASSANDRA_PORT, CASSANDRA_USERNAME, CASSANDRA_PASSWORD,
    CASSANDRA_KEYSPACE, EMBED_DIM
)

_session = None

def get_cassandra_session():
    global _session
    if _session is not None:
        return _session
    auth = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)
    cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT, auth_provider=auth)
    _session = cluster.connect(CASSANDRA_KEYSPACE)
    return _session

def vector_store_for_table(table: str) -> CassandraVectorStore:
    session = get_cassandra_session()
    return CassandraVectorStore(
        session=session,
        table=table,
        embedding_dimension=EMBED_DIM,
        keyspace=CASSANDRA_KEYSPACE,
    )
