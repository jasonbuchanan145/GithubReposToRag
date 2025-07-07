-- Keyspace
CREATE KEYSPACE IF NOT EXISTS rag
    WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 3};

-- Table to hold chunk metadata and vectors (768‑dim by default)
CREATE TABLE IF NOT EXISTS rag.chunks (
                                          repo        text,
                                          file_path   text,
                                          chunk_id    int,
                                          content     text,
                                          embedding   vector<float,768>,
                                          PRIMARY KEY ((repo), file_path, chunk_id)
    );

-- SAI ANN index on the embedding column
CREATE CUSTOM INDEX IF NOT EXISTS ann_chunks_embedding
    ON rag.chunks(embedding) USING 'StorageAttachedIndex'
    WITH OPTIONS = {
        'similarity_function' : 'cosine',
        'ann_algorithm'       : 'hnsw',        -- or ivf_flat
        'hnsw_ef_search'      : '32',
        'hnsw_m'              : '16'
        };