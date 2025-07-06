from fastapi import FastAPI
from vllm import LLM, SamplingParams
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

app = FastAPI()
llm = LLM(model="Qwen/Qwen2.5-7B-Chat-Int8", trust_remote_code=True)

cluster = Cluster(["cassandra"])
session = cluster.connect("rag")

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