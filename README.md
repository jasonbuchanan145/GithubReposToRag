# CodeRAG: Intelligent Code Repository Assistant

A specialized RAG (Retrieval Augmented Generation) system for intelligent code repository analysis and assistance. CodeRAG ingests repositories, builds vectorized knowledge bases, and provides contextual AI responses about repository structure, functionality, and implementation details.

## 🌟 Features

- **Repository Ingestion**: Efficiently analyzes GitHub repositories with specialized handling for different file types
- **Smart Notebook Processing**: Intelligently filters out boilerplate and execution noise from Jupyter notebooks
- **Hierarchical Analysis**: Creates summaries at repository, directory, and file levels
- **Context-Aware Queries**: Intelligently routes questions to the most relevant context (high-level overviews vs. code-specific details)
- **Cassandra Vector Storage**: Scalable persistence for embeddings using Cassandra
- **Kubernetes Deployment**: Complete Helm chart for easy deployment

## 🏗️ Architecture

```mermaid
graph TD
%% ===========================
%% Ingestion Pipeline (updated)
%% ===========================
subgraph "Ingestion Pipeline"
direction TB

%% Repo discovery & loading
R0[Repo Discovery] -->|"GraphQL fetch_repositories()"| R1[Repo List]
R1 -->|"for each repo/branch"| A[GithubRepositoryReader]
A -->|"concurrent_requests=6"| B[Raw Documents]
B -->|"optional dump (DATA_DIR)"| Bdump[(raw_documents_*.json)]

%% Filtering
B -->|"filter_documents()"| C[Filtered Documents]

%% Notebook decision BEFORE component-kind routing
C --> D{"Is Jupyter Notebook?"}
D -- Yes --> JN["Notebook Handling:\n• remove noise/outputs\n• strip artifacts\n • flag as standalone"]
D -- No --> ND["Generic processing"]

%% Normalize only notebooks, then join
JN --> NBD[Normalized Notebook Docs]
ND --> J["Preprocessed Docs"]
NBD --> J
%% Component kind inference after the join
J -->|"infer_component_kind()"| CK{Component Kind\nservice vs standalone}

%% Catalog (text) pipeline
subgraph "Catalog Pipeline"
direction TB
CP0["SentenceSplitter (chunk=1500, overlap=100)"]
CP1["SimpleNodeParser (chunk=1500, overlap=100)"]
CP2["Extractors\n• Summary\n• Title (nodes=3)\n• Keywords (10)"]
end

%% Code pipeline
subgraph "Code Pipeline"
direction TB
DP0["DynamicCodeSplitter (language inferred from metadata)"]
DP1["Extractors\n• Summary\n• Title (nodes=5)\n• Keywords (10)"]
end

%% Routing by component kind (high-level text vs source code)
CK -->|"routes text/docs"| CP0
CK -->|"routes source code"| DP0

%% Node outputs
CP0 --> CP1 --> CP2 --> Ncat[Catalog Nodes]
DP0 --> DP1 --> Ncode[Code Nodes]

%% Metadata & embeddings
subgraph "Metadata & Embedding"
direction TB
M0[Attach Common Metadata\nnamespace, repo, branch, collection,\ncomponent_kind, is_standalone,\ningest_run_id, path, language, doc_type]
M1["Embedding (intfloat/e5-small-v2, dim=384)"]
end

Ncat --> M0
Ncode --> M0
M0 --> M1 --> F["Vector DB Upsert (CassandraVectorStore)\nkeyspace=vector_store, table=embeddings"]

%% Auditing
subgraph "Audit & Stats (Cassandra)"
direction TB
A0[ingest_runs table]
A1[run_id, namespace, repo, branch,\ncollection, component_kind,\nstarted_at, finished_at, node_count]
end

%% Settings references
R0 -. uses .-> S0["SETTINGS (GITHUB_USER/TOKEN, DEFAULT_BRANCH)"]
M1 -. uses .-> S1["SETTINGS (EMBED_MODEL, EMBED_DIM)"]
F  -. uses .-> S2["SETTINGS (CASSANDRA_HOST/PORT/KEYSPACE/TABLE)"]

M0 -->|"counts"| A0
F  -->|"final node_count"| A1
end

%% ===========================
%% Query Pipeline (unchanged)
%% ===========================
subgraph "Query Pipeline"
Q[User Query] -->|REST API| API[RAG API Service]
API -->|Query Classification| QC{Query Type}
QC -->|High-Level| HL[Hierarchical Retriever]
QC -->|Code-Specific| CS[Code Retriever]

subgraph "Vector Store Integration"
DB[(Cassandra)]-->|Repository Summaries| HL
DB-->|Directory Summaries| HL
DB-->|File & Code Chunks| CS
end

HL --> R[Retrieval Results]
CS --> R
R -->|Context Enhancement| API
API -->|Augmented Prompt| LLM[LLM Response]
LLM -->|Response Formatting| API
API --> User
end

%% System edges outside subgraphs
F ---|Stored in| DB
Web[Web UI] -->|HTTP Requests| API

%% Feedback Loop (kept)
API -->|Interaction History| DB
LLM -->|Query Refinement| QC


```

## 🚀 Getting Started

### Prerequisites

- Minikube or Kubernetes cluster
- Docker
- Helm
- GitHub API token (for repository access)

### Setup

1. **Start the local environment:**
Start script currently only available for windows. 
```shell
   .\start.ps1
```

2. **Create GitHub token secret:**

   ```shell
   kubectl -n rag create secret generic github-token \
     --from-literal=token=your_github_token_here
   ```

3. **Access the web interface:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000

## 📝 Usage

### Ingesting a Repository

You can ingest a repository by using the CLI tool or the API:

```shell
# Using the CLI
kubectl -n rag exec -it deployment/rag-ingest -- python /app/llama_ingest.py --repo username/repository

# Using the API
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/username/repository"}'
```

### Querying the System

```shell
# API example
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d '{"query": "Explain the authentication flow in this repository"}'
```

## 🧠 Intelligent Query Handling

CodeRAG uses sophisticated query classification to provide the most relevant context:

- **High-Level Queries** like "Explain the architecture" or "Give an overview of the auth system" retrieve repository and directory summaries first.

- **Code-Specific Queries** like "How is the login function implemented?" or "What parameters does the API endpoint accept?" focus on retrieving code chunks.

## 🔧 Development

### Environment Setup

The project uses two Conda environments:

1. **Ingest Environment** (environment-scripts.yml): 
   - Used for repository ingestion and processing
   - Includes notebook processing tools and GitHub integration

2. **Service Environment** (environment-service.yml):
   - Used for the API service
   - Optimized for query processing and response generation

### Building From Source

```shell
# Build ingest container
docker build -t rag-ingest:latest -f scripts/Dockerfile .

# Build API container
docker build -t rag-api:latest -f services/rag_api/Dockerfile .

# Build frontend
cd frontend/nextjs-app
docker build -t rag-frontend:latest .
```

## 📚 Project Structure

- `/scripts` - Repository ingestion tools
- `/services/rag_api` - REST API for querying the system
- `/frontend` - Web interface
- `/helm` - Kubernetes deployment manifests

## 📦 Technologies

- **LlamaIndex**: Core RAG functionality
- **Sentence Transformers**: Text embeddings
- **vLLM**: Optimized LLM inference
- **Cassandra**: Vector database
- **FastAPI**: REST API
- **Kubernetes**: Deployment and orchestration

## 🤝 Contributing

Contributions are welcome, although this project is still in the non-functional MVP build out stage scheduled to be completed by August 15 2025.

## 📄 License

This project is licensed under the Apache License - see the LICENSE file for details.
