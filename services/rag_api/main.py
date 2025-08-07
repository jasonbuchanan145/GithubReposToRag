"""RAG API service using LlamaIndex."""

import logging
import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.cassandra import CassandraVectorStore
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel
import requests
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(
    title="RAG API Service",
    description="RAG service for querying code repositories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cassandra configuration
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "rag-demo-cassandra")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD", "testyMcTesterson")
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "vector_store")

# Get Qwen endpoint from environment variable
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "http://qwen:8000")

class QwenLLM(CustomLLM):
    """Custom LLM implementation for Qwen model matching ingest service."""

    model_name: str = "Qwen/Qwen3-4B-FP8"
    context_window: int = 11712  # Match the configured max_model_len
    num_output: int = 1024

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt with Qwen model."""
        logging.debug(f"🤖 QwenLLM.complete called with prompt length: {len(prompt)}")
        try:
            response_text = self._call_qwen_api(prompt, **kwargs)
            return CompletionResponse(text=response_text)
        except Exception as e:
            logging.error(f"🤖 QwenLLM.complete failed: {str(e)}")
            return CompletionResponse(text=f"Error: {str(e)}")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream completion (optional but recommended)."""
        response = self.complete(prompt, **kwargs)
        yield response

    def _call_qwen_api(self, prompt: str, **kwargs) -> str:
        """Make API call to Qwen service using vLLM OpenAI API."""
        logging.debug(f"🔗 Making API call to Qwen with prompt length: {len(prompt)}")
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", self.num_output),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
            }

            response = requests.post(
                f"{QWEN_ENDPOINT}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["text"]
                else:
                    return "No response generated"
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logging.warning(error_msg)
                return error_msg

        except requests.RequestException as e:
            error_msg = f"Connection Error: {str(e)}"
            logging.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            logging.error(error_msg)
            return error_msg

def initialize_llamaindex():
    """Initialize LlamaIndex with our custom LLM and embedding model."""
    # Load the embedding model
    EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Configure global settings
    Settings.llm = QwenLLM()
    Settings.embed_model = embed_model

    logging.info(f"✅ LlamaIndex initialized with QwenLLM and embedding model")

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

# Global variables for reuse
global_vector_store = None
global_index = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global global_vector_store, global_index

    logging.info("🚀 Starting RAG API service...")

    # Initialize LlamaIndex
    initialize_llamaindex()

    # Create the vector store
    global_vector_store = setup_vector_store()

    # Create the index from vector store
    global_index = VectorStoreIndex.from_vector_store(global_vector_store)

    logging.info("✅ RAG API service initialized successfully")

@app.get("/status")
async def status():
    """Service status endpoint."""
    return {
        "status": "healthy",
        "qwen_endpoint": QWEN_ENDPOINT,
        "cassandra_host": CASSANDRA_HOST,
        "index_ready": global_index is not None
    }

# Define response model
class RAGResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    repo_name: Optional[str] = None

@app.post("/rag")
async def rag(request: QueryRequest) -> RAGResponse:
    """RAG endpoint for querying code repositories."""
    logging.info(f"📝 Received query: {request.query}")

    try:
        # Create metadata filter if repo is specified
        metadata_filter = None
        if request.repo_name:
            metadata_filter = {"repo": request.repo_name}

        # Create retriever with optimized settings
        retriever = VectorIndexRetriever(
            index=global_index,
            similarity_top_k=request.top_k,
            filters=metadata_filter
        )

        # Create query engine with faster response synthesizer
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=TreeSummarize(verbose=False)  # Disable verbose for performance
        )

        # Execute the query
        logging.info(f"🔍 Executing query with top_k={request.top_k}, repo_filter={request.repo_name}")
        response = query_engine.query(request.query)

        # Extract sources for citation
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source = {
                    "text": node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text,
                    "metadata": node.node.metadata,
                    "score": getattr(node, 'score', None)
                }
                sources.append(source)

        logging.info(f"✅ Query completed successfully with {len(sources)} sources")

        return RAGResponse(
            answer=str(response),
            sources=sources
        )
    except Exception as e:
        logging.error(f"❌ Error executing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
