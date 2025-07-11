import os
from fastapi import FastAPI
from vllm import LLM, SamplingParams
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement

app = FastAPI()
llm = LLM(model="Qwen/Qwen2.5-7B-Chat-Int8", trust_remote_code=True)

# Get Cassandra connection details from environment variables
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD", "testyMcTesterson")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "vector_store")

# Configure authentication provider
auth_provider = PlainTextAuthProvider(username=CASSANDRA_USERNAME, password=CASSANDRA_PASSWORD)

# Connect to Cassandra with auth
cluster = Cluster(
    [CASSANDRA_HOST], 
    port=CASSANDRA_PORT,
    auth_provider=auth_provider
)
session = cluster.connect(CASSANDRA_KEYSPACE)

@app.post("/rag")
async def rag(query: str, top_k: int = 5):
    cql = (
        "SELECT content FROM rag.chunks "
        "ORDER BY embedding ANN OF %s LIMIT %s;"
    )
stmt = SimpleStatement(cql, fetch_size=top_k)
rows = session.execute(stmt, (list(qvec), top_k))
context = "\n".join(r.content for r in rows)

prompt = f"""You are an assistant answering questions about code.\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"""
reply = llm.generate(prompt, SamplingParams(max_tokens=512))[0]["text"]
return {"answer": reply.strip()}