import json
import logging
import os
import time
from typing import Any
from typing import List

import requests
from cassandra.auth import PlainTextAuthProvider
# Ensure Cassandra keyspace exists
from cassandra.cluster import Cluster
# LlamaIndex imports
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import CustomLLM
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.response.schema import StreamingResponse
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.cassandra import CassandraVectorStore
from nbconvert.filters import strip_ans
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class NotebookAwareDocumentTransformer:
    """Transform documents to handle notebooks and other special file types."""

    @staticmethod
    def transform_documents(documents: List[Document]) -> List[Document]:
        """Process documents to handle special file types like notebooks."""
        transformed_docs = []

        for doc in documents:
            file_path = doc.metadata.get('file_path', '')

            # Process Jupyter notebooks
            if file_path.endswith('.ipynb'):
                logging.info(f"🔬 Processing Jupyter notebook: {file_path}")

                # Get the notebook content
                notebook_path = file_path
                if not os.path.exists(notebook_path) and 'file_content' in doc.metadata:
                    # If we have content but not the file (e.g., from GitHub API)
                    with tempfile.NamedTemporaryFile(suffix='.ipynb', mode='w', delete=False) as temp_file:
                        temp_file.write(doc.text)
                        notebook_path = temp_file.name

                # Process the notebook to extract meaningful content
                processed_content = JupyterNotebookProcessor.process_notebook(notebook_path)

                # Create a new document with the processed content
                transformed_doc = Document(
                    text=processed_content,
                    metadata={
                        **doc.metadata,
                        'content_type': 'notebook',
                        'is_processed': True
                    }
                )
                transformed_docs.append(transformed_doc)

                # Clean up temporary file if created
                if notebook_path != file_path and os.path.exists(notebook_path):
                    os.unlink(notebook_path)
            else:
                # Keep other documents as is
                transformed_docs.append(doc)

        return transformed_docs

# Get environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

GITHUB_USER = os.environ.get("GITHUB_USER", "jasonbuchanan145")
DATA_DIR = os.environ.get("DATA_DIR", None)

# Cassandra configuration
CASSANDRA_HOST = os.environ.get("CASSANDRA_HOST", "localhost")
CASSANDRA_PORT = int(os.environ.get("CASSANDRA_PORT", "9042"))
CASSANDRA_USERNAME = os.environ.get("CASSANDRA_USERNAME", "cassandra")
CASSANDRA_PASSWORD = os.environ.get("CASSANDRA_PASSWORD", "cassandra")
CASSANDRA_KEYSPACE = os.environ.get("CASSANDRA_KEYSPACE", "vector_store")

# Check if we should skip ingestion
def should_skip_ingestion() -> bool:
    """Check if we should skip ingestion based on existing data."""
    if DATA_DIR and os.path.exists(os.path.join(DATA_DIR, ".skip_ingest")):
        logging.info("🔄 Skipping ingestion as previous data was found")
        return True
    return False

# Setup LlamaIndex with Cassandra
def setup_llamaindex() -> None:
    """Configure LlamaIndex settings."""
    # Setup Qwen endpoint for LLM access

    class QwenLLM(CustomLLM):
        """Custom LLM implementation for Qwen."""

        def __init__(self):
            super().__init__()
            self.endpoint = os.environ.get("QWEN_ENDPOINT", "http://qwen:8000")

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



    # Load the embedding model
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/e5-small-v2")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Configure global settings
    Settings.llm = QwenLLM()
    Settings.embed_model = embed_model

    auth_provider = PlainTextAuthProvider(
        username=CASSANDRA_USERNAME, 
        password=CASSANDRA_PASSWORD
    )

    cluster = Cluster(
        [CASSANDRA_HOST], 
        port=CASSANDRA_PORT,
        auth_provider=auth_provider
    )

    session = cluster.connect()

    # Create keyspace if it doesn't exist
    session.execute(
        f"""CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}"""
    )

    # Close the connection
    cluster.shutdown()

def setup_cassandra_vector_store() -> CassandraVectorStore:

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

    return vector_store

def ingest_repository(repo_url: str) -> None:
    """Ingest a single repository using LlamaIndex."""
    repo_name = repo_url.rstrip("/").split("/")[-1]
    logging.info(f"🔄 Processing repository: {repo_name}")

    # Define where to save repository data if using persistent storage
    repo_data_dir = None
    if DATA_DIR:
        repo_data_dir = os.path.join(DATA_DIR, "repos", repo_name)
        os.makedirs(repo_data_dir, exist_ok=True)
        logging.info(f"📂 Using persistent storage at: {repo_data_dir}")

    # Create GitHub client
    github_client = GithubClient(github_token=GITHUB_TOKEN)

    # Configure the repository reader
    reader = GithubRepositoryReader(
        github_client=github_client,
        owner=GITHUB_USER,
        repo=repo_name,
        filter_directories=(["node_modules", ".git", "__pycache__", "venv", ".idea"]),
        verbose=False,
        concurrent_requests=10
    )

    try:
        # Load documents from the repository
        logging.info(f"📥 Loading documents from repository {repo_name}...")
        documents = reader.load_data(branch="main")
        logging.info(f"📄 Loaded {len(documents)} files from {repo_name}")

        # Count notebooks
        notebook_count = sum(1 for doc in documents if doc.metadata.get('file_path', '').endswith('.ipynb'))
        if notebook_count > 0:
            logging.info(f"📓 Found {notebook_count} Jupyter notebooks in repository")
            # Process notebooks and other special file types
            logging.info(f"🔄 Processing documents with special handling for notebooks...")
            documents = NotebookAwareDocumentTransformer.transform_documents(documents)

        # Save raw documents if using persistent storage
        if repo_data_dir:
            raw_docs_path = os.path.join(repo_data_dir, "raw_documents.json")
            with open(raw_docs_path, "w") as f:
                json.dump([doc.to_dict() for doc in documents], f, indent=2)
            logging.info(f"💾 Saved raw documents to {raw_docs_path}")

        # Set up code-specific node parser for chunking
        code_splitter = CodeSplitter(
            language="python",  # Default, will be overridden based on file extension
            chunk_lines=40,
            chunk_lines_overlap=15,
            max_chars=4000
        )

        # Set up extractors for hierarchical summarization
        summary_extractor = SummaryExtractor(
            summaries=[
                "A brief summary of the code or text",
                "The main purpose and functionality"
            ],
            show_progress=True
        )

        title_extractor = TitleExtractor(nodes=5)
        keyword_extractor = KeywordExtractor(keywords=10)

        # Create ingestion pipeline
        logging.info(f"🔍 Creating ingestion pipeline with summarization...")
        pipeline = IngestionPipeline(
            transformations=[
                code_splitter,
                summary_extractor,
                title_extractor,
                keyword_extractor
            ]
        )

        # Process documents
        logging.info(f"⚙️ Processing documents through ingestion pipeline...")
        nodes = pipeline.run(documents=documents)
        logging.info(f"🧩 Created {len(nodes)} nodes from {len(documents)} documents")

        # Save processed nodes if using persistent storage
        if repo_data_dir:
            nodes_path = os.path.join(repo_data_dir, "processed_nodes.json")
            with open(nodes_path, "w") as f:
                json.dump([node.to_dict() for node in nodes], f, indent=2)
            logging.info(f"💾 Saved processed nodes to {nodes_path}")

        # Set up Cassandra vector store
        vector_store = setup_cassandra_vector_store()

        # Create vector index and store in Cassandra
        logging.info(f"📊 Creating vector index in Cassandra...")
        index = VectorStoreIndex(
            nodes=nodes,
            vector_store=vector_store,
            show_progress=True
        )

        # Add metadata to each node for repository identification
        for node in nodes:
            # Update metadata to include repository information
            node.metadata["repo"] = repo_name

        logging.info(f"✅ Successfully ingested repository: {repo_name}")
        return True

    except Exception as e:
        logging.error(f"❌ Error ingesting repository {repo_name}: {e}")
        return False

def fetch_repositories(username: str) -> List[str]:
    """Fetch repositories for a GitHub user using GraphQL API."""
    logging.info(f"🔍 Fetching repositories for user: {username}")

    query = """
    query($login: String!, $after: String) {
      user(login: $login) {
        repositories(first: 100, after: $after, isFork: false, ownerAffiliations: OWNER) {
          pageInfo { endCursor hasNextPage }
          nodes {
            name
            cloneUrl
            isArchived
          }
        }
      }
    }
    """

    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    repositories = []
    after = None

    while True:
        payload = {"query": query, "variables": {"login": username, "after": after}}
        response = requests.post(
            "https://api.github.com/graphql", 
            json=payload, 
            headers=headers, 
            timeout=30
        )
        response.raise_for_status()

        data = response.json()["data"]["user"]["repositories"]
        for node in data["nodes"]:
            if not node["isArchived"]:
                repositories.append(node["cloneUrl"])

        if not data["pageInfo"]["hasNextPage"]:
            break

        after = data["pageInfo"]["endCursor"]

    logging.info(f"📚 Found {len(repositories)} repositories")
    return repositories

def main():
    """Main function to run the repository ingestion process."""
    # Check if we should skip ingestion
    if should_skip_ingestion():
        return

    # Configure LlamaIndex
    setup_llamaindex()

    # Fetch repositories
    repositories = fetch_repositories(GITHUB_USER)

    # Process each repository
    successful = 0
    for i, repo_url in enumerate(repositories):
        logging.info(f"🔄 Processing repository {i+1}/{len(repositories)}: {repo_url}")
        if ingest_repository(repo_url):
            successful += 1

    # Mark ingestion as complete if using persistent volume
    if DATA_DIR:
        logging.info(f"✅ Ingestion complete. Processed {successful}/{len(repositories)} repositories successfully.")
        with open(os.path.join(DATA_DIR, ".ingest_complete"), "w") as f:
            f.write(f"Ingestion completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(f"\nProcessed {successful}/{len(repositories)} repositories successfully.")

if __name__ == "__main__":
    main()
