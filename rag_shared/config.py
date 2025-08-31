import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis-master:6379/0")
SSE_PING_SECONDS = int(os.getenv("SSE_PING_SECONDS", "15"))

MAX_RAG_ATTEMPTS = int(os.getenv("MAX_RAG_ATTEMPTS", "3"))
MIN_SOURCE_NODES = int(os.getenv("MIN_SOURCE_NODES", "1"))
# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Cassandra
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "rag-demo-cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD", "testyMcTesterson")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "vector_store")

DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "embeddings")
CODE_TABLE = os.getenv("CODE_TABLE", DEFAULT_TABLE)
PACKAGE_TABLE = os.getenv("PACKAGE_TABLE", DEFAULT_TABLE)
PROJECT_TABLE = os.getenv("PROJECT_TABLE", DEFAULT_TABLE)

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# LLM (Qwen via vLLM/OpenAI API compat)
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "http://qwen:8000")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
QWEN_MAX_OUTPUT = int(os.getenv("QWEN_MAX_OUTPUT", "4096"))
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.7"))
QWEN_TOP_P = float(os.getenv("QWEN_TOP_P", "0.9"))

# Router defaults
ROUTER_TOP_K = int(os.getenv("ROUTER_TOP_K", "5"))
# Feedback loop (RAG retries)
MAX_RAG_ATTEMPTS = int(os.getenv("MAX_RAG_ATTEMPTS", "3"))  # total attempts including first
MIN_SOURCE_NODES = int(os.getenv("MIN_SOURCE_NODES", "1"))  # retry if fewer than this

REDIS_URL = os.getenv("REDIS_URL", "redis://redis-master:6379/0")

SSE_PING_SECONDS = int(os.getenv("SSE_PING_SECONDS", "15"))
MAX_RAG_ATTEMPTS  = int(os.getenv("MAX_RAG_ATTEMPTS", "3"))
MIN_SOURCE_NODES  = int(os.getenv("MIN_SOURCE_NODES", "1"))