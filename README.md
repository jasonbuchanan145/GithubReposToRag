# CodeRAG: Intelligent Code Repository Assistant

A specialized RAG (Retrieval Augmented Generation) system for intelligent code repository analysis and assistance. CodeRAG ingests repositories, 
builds vectorized knowledge bases, and provides contextual AI responses about repository structure, functionality, and implementation details.

This project is intended in the first part of the arrange phase of LLM execution 

## 🌟 Features

- **Repository Ingestion**: Efficiently analyzes GitHub repositories with specialized handling for different file types
- **Smart Notebook Processing**: Intelligently filters out boilerplate and execution noise from Jupyter notebooks
- **Hierarchical Analysis**: Creates summaries at repository, directory, and file levels
- **Context-Aware Queries**: Intelligently routes questions to the most relevant context (high-level overviews vs. code-specific details)
- **Cassandra Vector Storage**: Scalable persistence for embeddings using Cassandra
- **Kubernetes Deployment**: Complete Helm chart for easy deployment

## 🏗️ Architecture

### System level diagram
```mermaid
flowchart LR

%% ========= Styles =========
    classDef client fill:#1f6feb,stroke:#0b3d91,color:#ffffff
    classDef svc fill:#2b6cb0,stroke:#1a365d,color:#ffffff
    classDef infra fill:#22543d,stroke:#1b3a2a,color:#e2e8f0
    classDef store fill:#2d3748,stroke:#4a5568,color:#e2e8f0
    classDef job fill:#805ad5,stroke:#553c9a,color:#ffffff
    classDef secret fill:#4a5568,stroke:#2d3748,color:#e2e8f0
    classDef ext fill:#6b7280,stroke:#374151,color:#ffffff
    classDef note fill:#4a5568,stroke:#2d3748,color:#e2e8f0

%% ========= External Client =========
    CLIENT[Web UI or Client]:::client
    MINIKUBE[Minikube services port forward proxy]:::infra

%% ========= Cluster =========
    subgraph K8s_Cluster
        direction TB
        subgraph Namespace rag
            direction TB
            API[REST API FastAPI<br/>Serves Web UI static files<br/>Service ClusterIP or NodePort]:::svc
            RQ[Redis Bitnami<br/>Queue and event bus<br/>StatefulSet and Service]:::infra
            WORKER[RAG Worker ARQ<br/>Dequeues jobs and emits events<br/>Deployment and Service]:::svc
            LLM[vLLM Qwen<br/>Deployment and Service]:::svc
            CASS[Cassandra Bitnami<br/>Tables CODE PACKAGE PROJECT<br/>StatefulSet and Service]:::store
            INGEST[Ingest Job<br/>GithubRepositoryReader and GithubClient]:::job
            GH_SECRET[Secret github_token<br/>created by start script]:::secret
        end
    end

    GITHUB[GitHub com]:::ext

%% ========= Paths =========
    CLIENT --> MINIKUBE
    MINIKUBE --> API
    API <-->|SSE stream and control| MINIKUBE

%% Job flow and events
    API -->|enqueue job| RQ
    RQ -->|dequeue job| WORKER
    WORKER -->|emit progress and tokens| RQ
    RQ -->|deliver events| API

%% Worker dependencies
    WORKER -->|LLM calls| LLM
    WORKER -->|read and write| CASS

%% Optional refinement loop shown with dashed arrows
    WORKER -. iterative prompts .-> LLM
    LLM -. responses for refinement .-> WORKER
    WORKER -. iterative reads .-> CASS
    CASS -. results for refinement .-> WORKER

%% Ingest path
    INGEST -->|uses secret| GH_SECRET
    INGEST -->|repo fetch| GITHUB
    INGEST -->|upserts| CASS

%% ========= Legend =========
    LEGEND[Dashed arrows indicate optional refinement loop with limited iterations]:::note
```
### Workflow level diagram
```mermaid
graph TD

%% ====== Styles ======
    classDef svc fill:#2b6cb0,stroke:#1a365d,color:#ffffff
    classDef llm fill:#6b46c1,stroke:#44337a,color:#ffffff
    classDef cass fill:#2d3748,stroke:#4a5568,color:#e2e8f0
    classDef pipe fill:#2f855a,stroke:#22543d,color:#ffffff
    classDef note fill:#4a5568,stroke:#2d3748,color:#e2e8f0
    classDef infra fill:#22543d,stroke:#1b3a2a,color:#e2e8f0
    classDef control fill:#805ad5,stroke:#553c9a,color:#ffffff

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

        C --> D{Is Jupyter Notebook?}:::pipe
        D -- Yes --> JN[Notebook Handling<br/>remove noise and outputs<br/>strip artifacts<br/>flag as standalone]:::pipe
        D -- No --> ND[Generic Processing]:::pipe

        JN --> NBD[Normalized Notebook Docs]:::pipe
        ND --> J[Preprocessed Docs]:::pipe
        NBD --> J
        J -->|infer_component_kind| CK{Component Kind<br/>service vs standalone}:::pipe

        subgraph Catalog_Pipeline
            direction TB
            CP0[SentenceSplitter<br/>chunk 1500 overlap 100]:::pipe
            CP1[SimpleNodeParser<br/>chunk 1500 overlap 100]:::pipe
            CP2[Extractors<br/>Summary<br/>Title nodes 3<br/>Keywords 10]:::pipe
        end

        subgraph Code_Pipeline
            direction TB
            DP0[DynamicCodeSplitter<br/>language inferred]:::pipe
            DP1[Extractors<br/>Summary<br/>Title nodes 5<br/>Keywords 10]:::pipe
        end

        CK -->|routes text and docs| CP0
        CK -->|routes source code| DP0
        CP0 --> CP1 --> CP2 --> Ncat[Catalog Nodes]:::pipe
        DP0 --> DP1 --> Ncode[Code Nodes]:::pipe

        M0[Attach Common Metadata<br/>namespace, repo, branch, collection<br/>component_kind, is_standalone<br/>ingest_run_id, path, language, doc_type]:::pipe
        Ncat --> M0
        Ncode --> M0
        M0 --> M1[Embedding E5 Small V2<br/>dim 384]:::llm
        M1 --> F[Vector DB Upsert<br/>to level tables]:::cass
    end

%% ===========================
%% Query Pipeline with Iterative Refinement
%% ===========================
subgraph Query_Pipeline
direction TB

U[User Query]:::pipe -->|HTTP REST| API[REST API - FastAPI<br/>app/main.py]:::svc

%% --- API Layer & Endpoints ---
subgraph REST_API_Service
direction TB
JR[JobsController<br/>app/controllers/jobs_controller.py]:::svc
JR -->|POST /rag/jobs<br/>enqueue| ENQ[enqueue_job<br/>name: run_rag_job<br/>args: job_id, QueryRequest]:::pipe
JR -->|GET /rag/jobs/:job_id/events<br/>SSE| SSE[SSE Stream -> client]:::pipe
JR -->|POST /rag/jobs/:job_id/cancel| CXL[set CancelFlags]:::pipe
end
API --> JR

%% --- Redis as queue + event bus ---
RQ[Redis<br/>ARQ queue / ProgressBus / CancelFlags]:::infra
ENQ -->|job payload<br/>query, force_level?, top_k?, repo_name?| RQ
CXL -->|cancel signal| RQ
RQ -->|SSE chunks<br/>bus.stream| SSE
SSE --> U

%% --- Worker Side Orchestrator ---
subgraph RAG_Worker_Service
direction TB
W[ARQ Worker<br/>worker/worker.py::run_rag_job]:::svc
ENG[RAGEngine<br/>worker/services/rag_engine.py]:::svc
ROUTER[RouterService<br/>worker/services/router_service.py]:::svc

%% Refinement Controller inside RAGEngine
subgraph REFINER[Refinement Controller]
direction TB
POL[Refinement Policy<br/>max_iterations = 3<br/>no quality threshold<br/>fixed retrieval params]:::control
PROMPT[Prompt Builder<br/>system  context  tools  style]:::control
CRIT[Self-Critique / Scorer<br/>LLM-as-judge - Qwen]:::control
ITER{more iterations remaining?}:::control
end
end

RQ -->|dequeue run_rag_job| W
W -->|attempt loop<br/>retry until MIN_SOURCE_NODES or limit| ENG
ENG -->|route: query, force_level| ROUTER
W -->|emit progress and tokens<br/>ProgressBus.emit: job_id, ...| RQ

%% --- Routed Tools / Retrieval ---
subgraph RAG_Tools
direction TB
NR[Direct LLM - no_rag<br/>DummyQueryEngine]:::pipe
CODE[Code Retriever]:::pipe
PKG[Package Retriever]:::pipe
PROJ[Project Retriever]:::pipe

CODE --> CR[VectorIndexRetriever<br/>top_k = ROUTER_TOP_K]:::pipe
CR --> CIDX[VectorStoreIndex<br/>CODE_TABLE]:::pipe

PKG --> PR[VectorIndexRetriever<br/>top_k = ROUTER_TOP_K]:::pipe
PR --> PIDX[VectorStoreIndex<br/>PACKAGE_TABLE]:::pipe

PROJ --> VRP[VectorIndexRetriever<br/>top_k = ROUTER_TOP_K]:::pipe
VRP --> JIDX[VectorStoreIndex<br/>PROJECT_TABLE]:::pipe

%% Synthesizers and Answers
NR --> A0[Draft Answer v0]:::pipe
CODE --> S1[TreeSummarize Synthesizer]:::pipe
PKG --> S2[TreeSummarize Synthesizer]:::pipe
PROJ --> S3[TreeSummarize Synthesizer]:::pipe
S1 --> A1[Draft Answer v1]:::pipe
S2 --> A2[Draft Answer v1]:::pipe
S3 --> A3[Draft Answer v1]:::pipe
end

ROUTER -->|LLM routing: no_rag / code / package / project| NR
ROUTER --> CODE
ROUTER --> PKG
ROUTER --> PROJ

%% Allow route change during refinement
CRIT -. may re-route based on critique .-> ROUTER

%% ===========================
%% Iterative Reprompting Loop (3 fixed iterations or cancel)
%% ===========================
A0 -->|evaluate| CRIT
A1 -->|evaluate| CRIT
A2 -->|evaluate| CRIT
A3 -->|evaluate| CRIT

CRIT -->|score and feedback| POL
POL -->|refine plan| PROMPT
PROMPT -->|assemble messages + context| CALL[(Call Qwen LLM)]:::llm
CALL --> RESP[Model Response - answer and citations]:::llm
RESP -->|append to working draft| DRAFT[Refined Answer v_k]:::pipe
DRAFT --> CRIT
CRIT --> ITER
ITER -- Yes --> PROMPT
ITER -- No --> FINAL[Final Answer + sources]:::pipe

%% Worker returns answers as streamed events
DRAFT --> W
FINAL --> W
W -->|stream tokens, progress, drafts, final| RQ

end

%% ===========================
%% Core Services (reference)
%% ===========================
subgraph Core_Services
direction TB

subgraph LLM_Services
direction TB
LLM_BUS[LLM Hub]:::svc
QWEN[Qwen LLM<br/>worker/services/qwen_llm.py]:::llm
EMB_NODE[Embedding E5 Small V2<br/>dim 384]:::llm
LLM_BUS --> QWEN
LLM_BUS --> EMB_NODE
end

subgraph Cassandra_Services
direction TB
CASS_BUS[Cassandra Hub]:::svc
V_CODE[Cassandra Vector Store<br/>CODE_TABLE]:::cass
V_PKG[Cassandra Vector Store<br/>PACKAGE_TABLE]:::cass
V_PROJ[Cassandra Vector Store<br/>PROJECT_TABLE]:::cass
AUD1[ingest_runs table]:::cass
AUD2[node_count rollup]:::cass
CASS_BUS --> V_CODE
CASS_BUS --> V_PKG
CASS_BUS --> V_PROJ
CASS_BUS --> AUD1
CASS_BUS --> AUD2
end
end

%% LLM usage
NR --- LLM_BUS
S1 --- LLM_BUS
S2 --- LLM_BUS
S3 --- LLM_BUS
M1 --- LLM_BUS
CALL --- QWEN

%% Cassandra usage
CIDX --- V_CODE
PIDX --- V_PKG
JIDX --- V_PROJ
F --- CASS_BUS

%% External client
WEB[Web UI or Client]:::pipe -->|HTTP rag| API

%% ====== Legend & Notes ======
LEG1[Legend<br/>Iterative loop: Build -> LLM -> Judge -> Decide<br/>Stop conditions: iteration count = 3 or cancel]:::note
LEG2[Events<br/>- ProgressBus emits: phase, score, tokens_used, iteration k<br/>- SSE streams: partial tokens, draft updates, final answer<br/>- Intermediate drafts/scores are streamed only not persisted]:::note

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

- **LlamaIndex**: Core RAG functionality and document processing
- **Sentence Transformers**: E5-small-v2 embeddings (384 dimensions)
- **vLLM + Qwen**: High-performance LLM inference with Qwen2.5-3B-Instruct
- **Cassandra**: Distributed vector database for embeddings storage
- **Redis**: Job queue (ARQ) and event streaming infrastructure
- **FastAPI**: REST API with Server-Sent Events (SSE) support
- **Kubernetes + Helm**: Container orchestration and deployment
- **Docker**: Containerization and image management

## 🤝 Contributing

Contributions are welcome! This project is currently in active development with the core MVP functionality being implemented. 

### Development Guidelines

- Follow the existing project structure and naming conventions
- Test changes with the provided Minikube environment
- Update documentation for any new features or architectural changes
- Ensure Docker images build successfully with the provided Dockerfiles

### Current Development Status

The system includes:
- ✅ Repository ingestion pipeline with GitHub integration
- ✅ Multi-level vector storage (code, package, project)
- ✅ Intelligent query routing and iterative refinement
- ✅ Kubernetes deployment with Helm charts
- ✅ Real-time job processing with progress streaming

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
