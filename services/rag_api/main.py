"""RAG API service using LlamaIndex."""

import logging
import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Query
# LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.cassandra import CassandraVectorStore
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI()

# Cassandra configuration
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD", "testyMcTesterson")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "vector_store")

# Get Qwen endpoint from environment variable
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "http://qwen:8000")

# Initialize LlamaIndex components
def initialize_llamaindex():
    """Initialize LlamaIndex with our custom LLM and embedding model."""
    # Setup custom LLM for Qwen
    from llama_index.core.llms import CustomLLM
    from llama_index.core.response.schema import StreamingResponse
    import requests
    from typing import List, Any, Dict

    class QwenLLM(CustomLLM):
        """Custom LLM implementation for Qwen."""

        def __init__(self):
            super().__init__()
            self.endpoint = QWEN_ENDPOINT

        def complete(self, prompt: str, **kwargs: Any) -> str:
            """Complete the prompt."""
            response = requests.post(
                f"{self.endpoint}/v1/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['text']
            else:
                raise ValueError(f"Error calling Qwen API: {response.status_code}")

        def stream_complete(self, prompt: str, **kwargs: Any) -> StreamingResponse:
            """Stream complete the prompt."""
            # For simplicity, we'll just return a non-streaming response
            return StreamingResponse(self.complete(prompt, **kwargs))

        def chat(self, messages: List[Dict], **kwargs: Any) -> str:
            """Chat with multiple messages."""
            # Format messages into a prompt
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return self.complete(prompt, **kwargs)

    # Configure LlamaIndex to use our embedding model
    from sentence_transformers import SentenceTransformer
    from llama_index.core.embeddings import HuggingFaceEmbedding

    # Load the embedding model
    EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Configure global settings
    Settings.llm = QwenLLM()
    Settings.embed_model = embed_model

    logging.info(f"✅ LlamaIndex initialized with custom LLM and embedding model")

# Create connection to Cassandra and set up the vector store
def setup_vector_store() -> CassandraVectorStore:
    """Set up the Cassandra vector store."""
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider

    logging.info(f"🔌 Connecting to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT}")

    auth_provider = PlainTextAuthProvider(
        username=CASSANDRA_USERNAME, 
        password=CASSANDRA_PASSWORD
    )

    cluster = Cluster(
        [CASSANDRA_HOST], 
        port=CASSANDRA_PORT,
        auth_provider=auth_provider
    )

    # Connect to the keyspace
    session = cluster.connect(CASSANDRA_KEYSPACE)

    # Create the vector store
    vector_store = CassandraVectorStore(
        session=session,
        table="embeddings",
        embedding_dimension=384,  # Dimension for e5-small-v2
        keyspace=CASSANDRA_KEYSPACE
    )

    logging.info(f"✅ Connected to Cassandra vector store")
    return vector_store

# Create the index and query engine
def create_query_engine(vector_store: CassandraVectorStore, top_k: int = 5):
    """Create a query engine from the vector store."""
    # Create the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create a retriever with the specified top_k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )

    # Create a response synthesizer that uses tree summarization
    response_synthesizer = TreeSummarize(
        verbose=True
    )

    # Create the query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    return query_engine

# Initialize LlamaIndex on startup
initialize_llamaindex()

# Create the vector store and query engine
vector_store = setup_vector_store()
default_query_engine = create_query_engine(vector_store)

# Define response model
class RAGResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

@app.post("/rag")
async def rag(
    query: str, 
    top_k: int = Query(5, description="Number of chunks to retrieve"),
    repo_name: Optional[str] = Query(None, description="Filter by repository name"),
    enable_followup: bool = Query(True, description="Enable follow-up queries")
) -> RAGResponse:
    """RAG endpoint for querying code repositories."""
    logging.info(f"📝 Received query: {query}")

    # Check if query is about a specific repository or asking for high-level info
    repo_specific = repo_name is not None or any(keyword in query.lower() for keyword in [
        "tell me about", "describe", "what is", "overview of", "summary of", "explain"
    ])

    # Create specialized query engines based on classification
    # For high-level queries, create an engine that prioritizes summaries
    if repo_specific:
        logging.info("🔍 Using high-level query engine with summary prioritization")

        # Create a metadata filter for content type and repo if specified
        metadata_filters = []

        # If repo name is specified, add it to the filters
        if repo_name:
            repo_filter = lambda meta: meta.get("repo") == repo_name
            metadata_filters.append(repo_filter)

        # Create a specialized index that prioritizes summaries first
        from llama_index.core.retrievers import BaseRetriever

        # Step 1: Create a retriever for repository summaries
        summary_filter = lambda meta: meta.get("content_type") == "repository_summary"
        summary_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store),
            similarity_top_k=min(3, top_k),
            filters=summary_filter if not repo_name else lambda meta: summary_filter(meta) and repo_filter(meta)
        )

        # Step 2: Create a retriever for directory summaries
        dir_filter = lambda meta: meta.get("content_type") == "directory_summary"
        dir_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store),
            similarity_top_k=min(3, top_k),
            filters=dir_filter if not repo_name else lambda meta: dir_filter(meta) and repo_filter(meta)
        )

        # Step 3: Create a retriever for file summaries and code chunks
        code_filter = lambda meta: meta.get("content_type") in ["file_summary", "code_chunk"]
        code_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store),
            similarity_top_k=top_k - 6,  # Adjust based on other retrievers
            filters=code_filter if not repo_name else lambda meta: code_filter(meta) and repo_filter(meta)
        )

        # Create a custom retriever that combines results from all three
        class HierarchicalRetriever(BaseRetriever):
            def __init__(self, summary_retriever, dir_retriever, code_retriever):
                self.summary_retriever = summary_retriever
                self.dir_retriever = dir_retriever
                self.code_retriever = code_retriever
                super().__init__()

            def _retrieve(self, query_str):
                # Get results from each retriever
                summary_nodes = self.summary_retriever.retrieve(query_str)
                dir_nodes = self.dir_retriever.retrieve(query_str)
                code_nodes = self.code_retriever.retrieve(query_str)

                # Combine all nodes, putting summaries first
                return summary_nodes + dir_nodes + code_nodes

        # Create the hierarchical retriever
        hierarchical_retriever = HierarchicalRetriever(
            summary_retriever, dir_retriever, code_retriever
        )

        # Create the query engine with the hierarchical retriever
        query_engine = RetrieverQueryEngine(
            retriever=hierarchical_retriever,
            response_synthesizer=TreeSummarize(verbose=True)
        )
    else:
        logging.info("🔍 Using code-specific query engine")
        # For technical queries, use a standard retriever focused on code chunks
        metadata_filter = None
        if repo_name:
            metadata_filter = lambda meta: meta.get("repo") == repo_name

        # Create a retriever optimized for code
        retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store),
            similarity_top_k=top_k,
            filters=metadata_filter
        )

        # Create the query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=TreeSummarize(verbose=True)
        )

    # Execute the query
    try:
        logging.info(f"🔍 Executing query with top_k={top_k}, repo_filter={repo_name}")
        response = query_engine.query(query)

        # Extract sources for citation
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source = {
                    "text": node.node.text[:200] + "...",  # Truncate long texts
                    "metadata": node.node.metadata
                }
                sources.append(source)

        logging.info(f"✅ Query completed successfully with {len(sources)} sources")

        return RAGResponse(
            answer=str(response),
            sources=sources
        )
    except Exception as e:
        logging.error(f"❌ Error executing query: {e}")
        return RAGResponse(
            answer=f"Error processing your query: {str(e)}",
            sources=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
