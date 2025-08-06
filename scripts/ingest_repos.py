import json
import logging
import os
import tempfile
import time
from typing import Any, Generator
from typing import List

import requests
from cassandra.auth import PlainTextAuthProvider
# Ensure Cassandra keyspace exists
from cassandra.cluster import Cluster
# LlamaIndex imports
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter
# Remove the problematic StreamingResponse import - we'll create our own simple implementation
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.cassandra import CassandraVectorStore

from scripts.jupyter_notebook_handling import JupyterNotebookProcessor
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any, Dict, Optional, Sequence
import requests

from scripts.langauge_detector import create_code_splitter_safely

# Configure logging with DEBUG level for troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
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

                # Get the notebook content - for GitHub API, we always need to create a temp file
                notebook_path = file_path
                temp_file_created = False

                if not os.path.exists(notebook_path):
                    # Create temporary file from document content (GitHub API case)
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.ipynb', mode='w', delete=False) as temp_file:
                            temp_file.write(doc.text)
                            notebook_path = temp_file.name
                            temp_file_created = True
                            logging.debug(f"📄 Created temporary notebook file for processing: {notebook_path}")
                    except Exception as e:
                        logging.warning(f"⚠️ Could not create temporary file for {file_path}: {e}")
                        # Fall back to original document without processing
                        transformed_docs.append(doc)
                        continue

                # Process the notebook to extract meaningful content using your custom processor
                try:
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
                    logging.debug(f"✅ Successfully processed notebook: {file_path}")

                except Exception as e:
                    logging.warning(f"⚠️ JupyterNotebookProcessor failed for {file_path}: {e}")
                    # Fall back to using the original document
                    transformed_docs.append(doc)

                finally:
                    # Clean up temporary file if we created one
                    if temp_file_created and os.path.exists(notebook_path):
                        try:
                            os.unlink(notebook_path)
                            logging.debug(f"🧹 Cleaned up temporary file: {notebook_path}")
                        except Exception as cleanup_error:
                            logging.warning(f"Could not clean up temporary file {notebook_path}: {cleanup_error}")
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
    """Check if we should skip ingestion based on existing data.

    In development mode, this always returns False to recreate the data.
    """
    # Always reingest data during development
    dev_mode = os.environ.get("DEV_MODE", "true").lower() in ("true", "1", "yes")

    if dev_mode:
        logging.info("🔄 Development mode: Always recreating data")
        # If there's an existing keyspace, drop it to start fresh
        if DATA_DIR and os.path.exists(os.path.join(DATA_DIR, ".ingest_complete")):
            try:
                # Clear Cassandra table data
                clear_cassandra_data()
                logging.info("🧹 Cleared existing vector data")
            except Exception as e:
                logging.warning(f"Could not clear Cassandra data: {e}")
        return False

    # Production behavior - check for skip file
    if DATA_DIR and os.path.exists(os.path.join(DATA_DIR, ".skip_ingest")):
        logging.info("🔄 Skipping ingestion as previous data was found")
        return True
    return False

# Simple streaming response class to replace the missing import
class SimpleStreamingResponse:
    """Simple streaming response implementation."""
    def __init__(self, content: str):
        self.content = content

    def __iter__(self):
        yield self.content

# Setup LlamaIndex with Cassandra
def setup_llamaindex() -> None:

    class QwenLLM(CustomLLM):
        """Custom LLM implementation for Qwen model."""

        model_name: str = "Qwen/Qwen3-4B-FP8"  # Match the model in values.yaml
        context_window: int = 4096
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
            logging.debug(f"🤖 Prompt preview: {prompt[:200]}...")
            try:
                response_text = self._call_qwen_api(prompt, **kwargs)
                logging.debug(f"🤖 QwenLLM response length: {len(response_text)}")
                return CompletionResponse(text=response_text)
            except Exception as e:
                logging.error(f"🤖 QwenLLM.complete failed: {str(e)}")
                return CompletionResponse(text=f"Error: {str(e)}")

        @llm_completion_callback()
        def stream_complete(self, prompt: str, **kwargs: Any):
            """Stream completion (optional but recommended)."""
            # Implement streaming if your Qwen API supports it
            response = self.complete(prompt, **kwargs)
            yield response

        def _call_qwen_api(self, prompt: str, **kwargs) -> str:
            """Make API call to Qwen service using vLLM OpenAI API."""
            try:
                # Use correct vLLM service name and endpoint
                qwen_host = "qwen"  # Service name from your deployment
                qwen_port = "8000"

                # Use OpenAI-compatible completions endpoint
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.num_output,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                }

                response = requests.post(
                    f"http://{qwen_host}:{qwen_port}/v1/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Increased timeout for model processing
                )

                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["text"]
                    else:
                        return "No response generated"
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    print(error_msg)  # Debug logging
                    return error_msg

            except requests.RequestException as e:
                error_msg = f"Connection Error: {str(e)}"
                print(error_msg)  # Debug logging
                return error_msg
            except Exception as e:
                error_msg = f"Unexpected Error: {str(e)}"
                print(error_msg)  # Debug logging
                return error_msg
    # Initialize the QwenLLM class first
    llm = QwenLLM()

    # Load the embedding model
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/e5-small-v2")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Configure global settings
    Settings.llm = llm
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
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}""")

    # Close the connection
    cluster.shutdown()

def clear_cassandra_data() -> None:
    """Clear all data from the Cassandra vector store table."""
    logging.info(f"🧹 Clearing Cassandra vector store data...")

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

    # Truncate the embeddings table to clear all data
    session.execute(f"TRUNCATE TABLE {CASSANDRA_KEYSPACE}.embeddings")

    # Close the connection
    cluster.shutdown()

    logging.info(f"✅ Successfully cleared Cassandra vector store data")

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


def ingest_repository(repo_name: str) -> bool:
    """Ingest a single repository using LlamaIndex."""
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
        filter_directories=(
            ["node_modules", ".git", "__pycache__", "venv", ".idea"],
            GithubRepositoryReader.FilterType.EXCLUDE
        ),
        verbose=False,
        concurrent_requests=5,  # Reduced to avoid rate limiting
        timeout=60  # Increased timeout for API requests
    )

    try:
        # Load documents from the repository
        logging.info(f"📥 Loading documents from repository {repo_name}...")
        documents = reader.load_data(branch="main")
        logging.info(f"📄 Loaded {len(documents)} files from {repo_name}")

        # Filter out unwanted file types that are not useful for code analysis
        SKIP_EXTENSIONS = {
            # Data files
            '.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.feather',
            '.json', '.xml', '.jsonl', '.ndjson'
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.ico',
            '.tiff', '.tif', '.psd',
            # Audio/Video
            '.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.flv',
            # Archives
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2',
            # Binary files
            '.exe', '.dll', '.so', '.dylib', '.bin',
            # Large text dumps that aren't useful
            '.log', '.dump', '.backup',
            # Database files
            '.db', '.sqlite', '.sqlite3'
        }

        # Skip files by exact filename (case-insensitive)
        SKIP_FILENAMES = {
            'license', 'license.txt', 'license.md',
            'changelog', 'changelog.txt', 'changelog.md',
            'authors', 'authors.txt', 'authors.md',
            'contributors', 'contributors.txt', 'contributors.md',
            'copying', 'copying.txt', 'copying.md',
            'notice', 'notice.txt', 'notice.md',
            '.gitignore', '.gitattributes', '.gitmodules',
            '.dockerignore', '.eslintignore', '.prettierignore'
        }

        filtered_documents = []
        skipped_files = []

        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            file_extension = '.' + file_path.split('.')[-1].lower() if '.' in file_path else ''
            filename = file_path.split('/')[-1].lower()  # Get just the filename part

            # Skip by extension
            if file_extension in SKIP_EXTENSIONS:
                skipped_files.append(file_path)
                continue

            # Skip by filename
            if filename in SKIP_FILENAMES:
                skipped_files.append(file_path)
                continue

            filtered_documents.append(doc)

        if skipped_files:
            logging.info(f"⏩ Skipped {len(skipped_files)} files with unwanted extensions: {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")

        documents = filtered_documents
        logging.info(f"📄 Processing {len(documents)} files after filtering")

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

        # Test Qwen connectivity before proceeding
        logging.info("🔍 Testing Qwen service connectivity...")
        try:
            test_response = requests.get("http://qwen:8000/v1/models", timeout=10)
            logging.info(f"✅ Qwen service accessible - Status: {test_response.status_code}")
        except Exception as e:
            logging.warning(f"⚠️ Cannot reach Qwen service: {e}")

        # Set up code-specific node parser for chunking
        # In your ingest_repository function, around line 355, replace:
        try:
            # Pass additional context for better language detection
            code_splitter = create_code_splitter_safely(
                file_path=None,  # Will be determined per document
                language=None,   # Will auto-detect
                document_content=None  # Will be determined per document
            )
        except ImportError as e:
            logging.warning(f"CodeSplitter unavailable due to missing dependency: {e}")
            # Fallback to simple text splitter
            from llama_index.core.node_parser import SentenceSplitter
            code_splitter = SentenceSplitter(
                chunk_size=4000,
                chunk_overlap=200
            )

        # Set up extractors for hierarchical summarization
        logging.info(f"📋 Setting up extractors...")
        summary_extractor = SummaryExtractor(
            summaries=["self"],
            show_progress=True
        )
        logging.info(f"📋 SummaryExtractor created")

        title_extractor = TitleExtractor(nodes=5)
        logging.info(f"📋 TitleExtractor created")

        keyword_extractor = KeywordExtractor(keywords=10)
        logging.info(f"📋 KeywordExtractor created")

        # Create ingestion pipeline with LLM extractors
        logging.info(f"🔍 Creating ingestion pipeline with summarization...")
        pipeline = IngestionPipeline(
            transformations=[
                code_splitter,
                summary_extractor,
                title_extractor,
                keyword_extractor
            ]
        )
        logging.info(f"🔍 Pipeline created with {len(pipeline.transformations)} transformations")

        # Process documents
        logging.info(f"⚙️ Processing {len(documents)} documents through ingestion pipeline...")
        logging.info(f"⚙️ Starting pipeline.run()...")

        # Add timeout and progress tracking
        import time
        start_time = time.time()
        try:
            nodes = pipeline.run(documents=documents)
            elapsed_time = time.time() - start_time
            logging.info(f"🧩 Created {len(nodes)} nodes from {len(documents)} documents in {elapsed_time:.2f}s")
        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(f"❌ Pipeline failed after {elapsed_time:.2f}s: {str(e)}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

        # Save processed nodes if using persistent storage
        if repo_data_dir:
            nodes_path = os.path.join(repo_data_dir, "processed_nodes.json")
            with open(nodes_path, "w") as f:
                json.dump([node.to_dict() for node in nodes], f, indent=2)
            logging.info(f"💾 Saved processed nodes to {nodes_path}")

        # Set up Cassandra vector store
        vector_store = setup_cassandra_vector_store()

        # Add metadata to each node for repository identification BEFORE creating the index
        for node in nodes:
            # Update metadata to include repository information
            node.metadata["repo"] = repo_name

        # Create vector index and store in Cassandra
        logging.info(f"📊 Creating vector index in Cassandra...")
        index = VectorStoreIndex(
            nodes=nodes,
            vector_store=vector_store,
            show_progress=True
        )

        logging.info(f"✅ Successfully ingested repository: {repo_name}")
        return True

    except Exception as e:
        logging.error(f"❌ Error ingesting repository {repo_name}: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
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
            url
            isArchived
            isPrivate
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
            # Skip archived and private repositories
            if node["isArchived"]:
                logging.info(f"⏩ Skipping archived repository: {node['name']}")
            elif node["isPrivate"]:
                logging.info(f"🔒 Skipping private repository: {node['name']}")
            else:
                # Just return the repository name since GithubRepositoryReader uses owner/repo separately
                repositories.append(node["name"])

        if not data["pageInfo"]["hasNextPage"]:
            break

        after = data["pageInfo"]["endCursor"]

    logging.info(f"📚 Found {len(repositories)} repositories")
    return repositories

def main():
    # Remove any previous ingestion completion markers in dev mode
    dev_mode = os.environ.get("DEV_MODE", "true").lower() in ("true", "1", "yes")
    if dev_mode and DATA_DIR:
        ingest_complete_path = os.path.join(DATA_DIR, ".ingest_complete")
        if os.path.exists(ingest_complete_path):
            try:
                os.remove(ingest_complete_path)
                logging.info("🧹 Removed previous ingestion completion marker")
            except Exception as e:
                logging.warning(f"Could not remove ingestion marker: {e}")

    # Check if we should skip ingestion
    if should_skip_ingestion():
        logging.warning("Skipping ingestion as requested")
        return

    # Configure LlamaIndex
    setup_llamaindex()

    # Fetch repositories
    repositories = fetch_repositories(GITHUB_USER)

    # Process each repository
    successful = 0
    for i, repo_name in enumerate(repositories):
        logging.info(f"🔄 Processing repository {i+1}/{len(repositories)}: {repo_name}")
        if ingest_repository(repo_name):
            successful += 1

    # Mark ingestion as complete if using persistent volume
    if DATA_DIR:
        logging.info(f"✅ Ingestion complete. Processed {successful}/{len(repositories)} repositories successfully.")
        with open(os.path.join(DATA_DIR, ".ingest_complete"), "w") as f:
            f.write(f"Ingestion completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(f"\nProcessed {successful}/{len(repositories)} repositories successfully.")


if __name__ == "__main__":
    main()