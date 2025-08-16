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

%% Layout helpers
classDef svc fill:#2b6cb0,stroke:#1a365d,color:#ffffff
classDef llm fill:#6b46c1,stroke:#44337a,color:#ffffff
classDef cass fill:#2d3748,stroke:#4a5568,color:#e2e8f0
classDef pipe fill:#2f855a,stroke:#22543d,color:#ffffff
classDef note fill:#4a5568,stroke:#2d3748,color:#e2e8f0

%% ===========================
%% Ingestion Pipeline
%% ===========================
subgraph Ingestion_Pipeline
 direction TB
 R0[Repo Discovery]:::pipe -->|GraphQL fetch_repositories| R1[Repo List]:::pipe
 R1 -->|for each repo or branch| A[GithubRepositoryReader]:::pipe
 A -->|concurrent_requests 6| B[Raw Documents]:::pipe
 B -->|optional dump DATA_DIR| Bdump[raw_documents_*.json]:::note
 B -->|filter_documents| C[Filtered Documents]:::pipe

 C --> D{Is Jupyter Notebook}:::pipe
 D -- Yes --> JN[Notebook Handling
remove noise and outputs
strip artifacts
flag as standalone]:::pipe
 D -- No --> ND[Generic processing]:::pipe

 JN --> NBD[Normalized Notebook Docs]:::pipe
 ND --> J[Preprocessed Docs]:::pipe
 NBD --> J
 J -->|infer_component_kind| CK{Component Kind
service vs standalone}:::pipe

 subgraph Catalog_Pipeline
  direction TB
  CP0[SentenceSplitter chunk 1500 overlap 100]:::pipe
  CP1[SimpleNodeParser chunk 1500 overlap 100]:::pipe
  CP2[Extractors
Summary
Title nodes 3
Keywords 10]:::pipe
 end

 subgraph Code_Pipeline
  direction TB
  DP0[DynamicCodeSplitter language inferred]:::pipe
  DP1[Extractors
Summary
Title nodes 5
Keywords 10]:::pipe
 end

 CK -->|routes text and docs| CP0
 CK -->|routes source code| DP0
 CP0 --> CP1 --> CP2 --> Ncat[Catalog Nodes]:::pipe
 DP0 --> DP1 --> Ncode[Code Nodes]:::pipe

 M0[Attach Common Metadata
namespace repo branch collection
component_kind is_standalone
ingest_run_id path language doc_type]:::pipe
 Ncat --> M0
 Ncode --> M0
 M0 --> M1[Embedding E5 Small V2 dim 384]:::llm
 M1 --> F[Vector DB Upsert to level tables]:::cass
end

%% ===========================
%% Query Pipeline
%% ===========================
subgraph Query_Pipeline
 direction TB
 U[User Query]:::pipe -->|REST API| API[RAG API Service]:::pipe
 API -->|Controller| CTRL[rag_controller.py]:::pipe
 CTRL -->|Service call| RSVC[RAGService.run]:::pipe
 RSVC -->|Delegates| ROUTER[RouterService.route]:::pipe
 ROUTER -->|LLM routing| QC{Which tool}:::pipe

 RSVC -. force_level none .-> NR[Direct LLM]:::pipe
 RSVC -. force_level code .-> CODE[Code Retriever]:::pipe
 RSVC -. force_level package .-> PKG[Package Retriever]:::pipe
 RSVC -. force_level project .-> PROJ[Project Retriever]:::pipe

 QC -->|no_rag| NR
 QC -->|code| CODE
 QC -->|package| PKG
 QC -->|project| PROJ

 subgraph RAG_Tools
  direction TB
  CODE --> CR[VectorIndexRetriever top_k ROUTER_TOP_K]:::pipe
  CR --> CIDX[VectorStoreIndex]:::pipe
  PKG --> PR[VectorIndexRetriever top_k ROUTER_TOP_K]:::pipe
  PR --> PIDX[VectorStoreIndex]:::pipe
  PROJ --> JR[VectorIndexRetriever top_k ROUTER_TOP_K]:::pipe
  JR --> JIDX[VectorStoreIndex]:::pipe
 end

 NR --> A1[Answer]:::pipe
 CODE --> S1[TreeSummarize Synthesizer]:::pipe
 PKG --> S2[TreeSummarize Synthesizer]:::pipe
 PROJ --> S3[TreeSummarize Synthesizer]:::pipe
 S1 --> A2[Answer with sources]:::pipe
 S2 --> A3[Answer with sources]:::pipe
 S3 --> A4[Answer with sources]:::pipe
 A1 --> RSVC
 A2 --> RSVC
 A3 --> RSVC
 A4 --> RSVC
 RSVC --> CTRL --> API --> U
end

%% ===========================
%% Core Services Sections
%% ===========================
subgraph Core_Services
 direction TB

 subgraph LLM_Services
  direction TB
  LLM_BUS[LLM Hub]:::svc
  QWEN[Qwen LLM]:::llm
  EMB_NODE[Embedding E5 Small V2 dim 384]:::llm
  LLM_BUS --> QWEN
  LLM_BUS --> EMB_NODE
 end

 subgraph Cassandra_Services
  direction TB
  CASS_BUS[Cassandra Hub]:::svc
  V_CODE[Cassandra Vector Store CODE_TABLE]:::cass
  V_PKG[Cassandra Vector Store PACKAGE_TABLE]:::cass
  V_PROJ[Cassandra Vector Store PROJECT_TABLE]:::cass
  AUD1[ingest_runs table]:::cass
  AUD2[node_count rollup]:::cass
  CASS_BUS --> V_CODE
  CASS_BUS --> V_PKG
  CASS_BUS --> V_PROJ
  CASS_BUS --> AUD1
  CASS_BUS --> AUD2
 end
end

%% Connect pipelines to hubs to avoid layout drift
%% LLM usage
NR --- LLM_BUS
S1 --- LLM_BUS
S2 --- LLM_BUS
S3 --- LLM_BUS
M1 --- LLM_BUS

%% Cassandra usage
CIDX --- V_CODE
PIDX --- V_PKG
JIDX --- V_PROJ
F --- CASS_BUS

%% External client
WEB[Web UI or Client]:::pipe -->|HTTP rag| API
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
