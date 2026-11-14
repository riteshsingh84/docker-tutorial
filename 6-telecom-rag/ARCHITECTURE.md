# Telecom RAG Chatbot — Architecture & Technical Specifications

This document provides a comprehensive technical breakdown of the multi-container **Telecom RAG (Retrieval-Augmented Generation) Chatbot** located in `6-telecom-rag/`.

---

## 1. System Overview & Container Topology

The application decomposes a RAG pipeline into three independent, decoupled microservices connected over an internal Docker bridge network (`telecom-rag_default`), with external inference handled by the Groq LPU Cloud.

```mermaid
flowchart TD
    subgraph Host ["Host Machine (Browser / User)"]
        User["User Web Browser"]
        LocalModels["./models (Local Weights Cache)"]
    end

    subgraph DockerNetwork ["Docker Network: telecom-rag_default"]
        subgraph ragAppContainer ["rag-app Service (:8501)"]
            StreamlitUI["Streamlit UI (src/app.py)"]
            RAGChain["RAG Chain (src/rag_chain.py)"]
            Retriever["Merged Retriever (src/retriever.py)"]
            RemoteEmbeddings["Remote Embeddings Client (src/remote_embeddings.py)"]
            IngestScripts["Ingest CLI (src/ingest/*)"]
        end

        subgraph embeddingsContainer ["embeddings Service (:8001)"]
            FastAPI["FastAPI Server (main.py)"]
            SentenceTransformer["SentenceTransformer Engine"]
            MountedWeights["/models (Read-Only Mount)"]
        end

        subgraph chromaContainer ["chroma Service (:8000)"]
            ChromaServer["ChromaDB HTTP Server"]
            NamedVolume[("chroma-data Volume")]
        end
    end

    subgraph Cloud ["External Cloud Infrastructure"]
        GroqAPI["Groq LPU Cloud (openai/gpt-oss-120b)"]
    end

    %% Interactions
    User <-->|"HTTP :8501"| StreamlitUI
    LocalModels -.->|"Volume Bind Mount"| MountedWeights
    MountedWeights --> SentenceTransformer
    SentenceTransformer <--> FastAPI

    StreamlitUI --> RAGChain
    RAGChain --> Retriever
    Retriever -->|"HTTP POST /embed"| FastAPI
    Retriever -->|"Query Collections"| ChromaServer
    ChromaServer <--> NamedVolume

    IngestScripts -->|"HTTP POST /embed"| FastAPI
    IngestScripts -->|"Store Vectors"| ChromaServer

    RAGChain -->|"HTTPS API Stream"| GroqAPI
```

---

## 2. Component Breakdown

| Container / Service | Base Image | Port | Memory Limit | Primary Purpose |
|---|---|---|---|---|
| **`rag-app`** | `python:3.12-slim` | `8501:8501` | `2GB` | Streamlit chat UI, LangChain orchestration, document ingestion, and external Groq API communication. |
| **`embeddings`** | `python:3.12-slim` *(or PyTorch CUDA for GPU)* | `8001` (internal) | `3GB` (CPU) / `6GB` (GPU) | Dedicated embedding microservice. Loads `sentence-transformers/all-MiniLM-L6-v2` into memory once and computes 384-dimensional dense vectors on demand over HTTP. |
| **`chroma`** | `chromadb/chroma:latest` | `8000` (internal) | Default | Vector database that stores document chunks, metadata, and embeddings across three distinct collections (`faq`, `tickets`, `guides`). Persists data to `chroma-data`. |

---

## 3. Detailed File Catalog

### Project Root (`6-telecom-rag/`)

| File | Purpose |
|---|---|
| **`compose.yaml`** | Core Docker Compose configuration declaring the three services (`embeddings`, `chroma`, `rag-app`), resource constraints (`mem_limit`), port bindings, environment variables, volume mappings, and container dependencies. |
| **`compose.gpu.yaml`** | Docker Compose override file that swaps `embeddings` with an NVIDIA CUDA runtime image (`Dockerfile.gpu`) and reserves all available GPU devices. |
| **`download_model_host.py`** | Host-side helper script. Downloads all 8 model configuration, tokenizer, and weight files directly from Hugging Face into `./models` on the host machine, bypassing corporate proxy and container restrictions. |
| **`models/`** | Directory containing pre-downloaded `all-MiniLM-L6-v2` model weights and configs (`model.safetensors`, `config.json`, `tokenizer.json`, etc.). Mounted into the container at `/models:ro`. |

---

### Embedding Microservice (`6-telecom-rag/embedding-service/`)

| File | Purpose |
|---|---|
| **`main.py`** | FastAPI application exposing `/embed` (generates embeddings for a batch of strings) and `/health`. Detects if `/models/config.json` exists locally and loads offline; otherwise falls back to the Hub name. |
| **`Dockerfile`** | Multi-stage, CPU-optimized container definition using Astral `uv` for fast package management. Pre-configures corporate proxy environment flags (`UV_INSECURE_HOST`). |
| **`Dockerfile.gpu`** | GPU-accelerated container definition based on `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`. |
| **`pyproject.toml`** | Python project definition pinning `fastapi`, `uvicorn`, `sentence-transformers`, and CPU PyTorch wheels (`https://download.pytorch.org/whl/cpu`). |
| **`uv.lock`** | Fully resolved, cryptographically hashed dependency lockfile for deterministic container builds. |
| **`.dockerignore`** | Prevents `.venv`, build artifacts, and local cache files from entering the Docker build context. |

---

### RAG Application (`6-telecom-rag/rag-app/`)

| File | Purpose |
|---|---|
| **`src/app.py`** | Streamlit web application. Renders chat messages, interactive sample question sidebar buttons, handles user input, and streams assistant replies using `st.write_stream()`. |
| **`src/rag_chain.py`** | Defines the LangChain LCEL pipeline (`retriever | prompt | llm | parser`). Configures `ChatGroq` (`openai/gpt-oss-120b`) with a proxy-tolerant HTTP client (`verify=False`). |
| **`src/retriever.py`** | Builds a unified multi-collection retriever. Queries ChromaDB for top `k=3` relevant matches from `faq`, `tickets`, and `guides`, then merges them into a single context block. |
| **`src/remote_embeddings.py`** | Custom LangChain `Embeddings` client that delegates `embed_documents` and `embed_query` calls to the `embeddings` microservice (`http://embeddings:8001/embed`) with detailed connection error handling. |
| **`src/main.py`** | Alternative terminal/CLI chat interface for testing the RAG pipeline directly without Streamlit. |
| **`src/ingest/faq.py`** | Ingestion worker: parses `data/faq.csv`, generates embeddings via `remote_embeddings.py`, and stores them in ChromaDB collection `faq`. |
| **`src/ingest/tickets.py`** | Ingestion worker: queries resolved tickets from SQLite `data/tickets.db`, generates embeddings, and stores them in ChromaDB collection `tickets`. |
| **`src/ingest/pdf.py`** | Ingestion worker: loads `data/telecom_guide.pdf`, applies `RecursiveCharacterTextSplitter` (chunk size 600, overlap 100), generates embeddings, and stores them in ChromaDB collection `guides`. |
| **`data/faq.csv`** | Structured Q&A dataset of telecom policies, roaming rules, and billing guidelines. |
| **`data/tickets.db`** | SQLite database containing real-world customer support tickets with problem descriptions and resolutions. |
| **`data/telecom_guide.pdf`** | Comprehensive customer manual detailing APN setup, eSIM activation, Wi-Fi calling, and hardware troubleshooting. |
| **`.env.example`** | Template environment file documenting `GROQ_API_KEY`. |
| **`.env`** | Local secrets file holding the active `GROQ_API_KEY` (git-ignored). |

---

## 4. Data Ingestion Architecture & Dataflow

Ingestion is executed once (or upon data changes) via isolated ephemeral containers:
```bash
docker compose run --rm rag-app python -m src.ingest.<faq|tickets|pdf>
```

```mermaid
flowchart LR
    subgraph DataSources ["Knowledge Sources"]
        FAQ["data/faq.csv\n(CSV Table)"]
        Tickets["data/tickets.db\n(SQLite Database)"]
        PDF["data/telecom_guide.pdf\n(PDF Manual)"]
    end

    subgraph IngestionScripts ["Ingestion Pipeline (rag-app)"]
        FAQParser["src.ingest.faq\n(1 Row = 1 Document)"]
        TicketParser["src.ingest.tickets\n(Resolved Tickets Filter)"]
        PDFChunker["src.ingest.pdf\n(RecursiveCharacterSplitter\n600 chars, 100 overlap)"]
    end

    subgraph VectorGeneration ["Embeddings Microservice (:8001)"]
        EmbedAPI["POST /embed\nBatch size: 64"]
        MiniLM["all-MiniLM-L6-v2\n(384-dim Vectors)"]
    end

    subgraph Storage ["ChromaDB (:8000)"]
        ColFAQ[("Collection: 'faq'")]
        ColTickets[("Collection: 'tickets'")]
        ColGuides[("Collection: 'guides'")]
        Disk[("chroma-data Volume")]
    end

    FAQ --> FAQParser
    Tickets --> TicketParser
    PDF --> PDFChunker

    FAQParser -->|"Raw Text"| EmbedAPI
    TicketParser -->|"Raw Text"| EmbedAPI
    PDFChunker -->|"Raw Text"| EmbedAPI

    EmbedAPI --> MiniLM
    MiniLM -->|"Float Vectors"| EmbedAPI

    EmbedAPI -->|"Vectors + Metadata"| ColFAQ
    EmbedAPI -->|"Vectors + Metadata"| ColTickets
    EmbedAPI -->|"Vectors + Metadata"| ColGuides

    ColFAQ --> Disk
    ColTickets --> Disk
    ColGuides --> Disk
```

---

## 5. Sequential Query Execution Diagram

The diagram below details the exact lifecycle of a user query from the moment it is entered in Streamlit until the LLM response streams back.

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant UI as Streamlit UI (rag-app:8501)
    participant Chain as LangChain LCEL (rag_chain.py)
    participant Ret as Merged Retriever (retriever.py)
    participant EmbedClient as RemoteEmbeddings (remote_embeddings.py)
    participant EmbedSvc as FastAPI (embeddings:8001)
    participant Model as all-MiniLM-L6-v2 (/models)
    participant Chroma as ChromaDB (chroma:8000)
    participant Groq as Groq LPU Cloud API

    User->>UI: Submit question: "Why is my mobile internet slow?"
    UI->>Chain: stream(question)
    
    rect rgb(240, 248, 255)
        Note over Chain,Model: Step 1: Query Vectorization
        Chain->>Ret: invoke(question)
        Ret->>EmbedClient: embed_query(question)
        EmbedClient->>EmbedSvc: HTTP POST /embed {"texts": [question]}
        EmbedSvc->>Model: model.encode([question])
        Model-->>EmbedSvc: 384-dimensional vector [0.021, -0.054, ...]
        EmbedSvc-->>EmbedClient: HTTP 200 {"embeddings": [[...]], "dim": 384}
        EmbedClient-->>Ret: Query vector
    end

    rect rgb(255, 250, 240)
        Note over Ret,Chroma: Step 2: Multi-Collection Similarity Retrieval
        par Query FAQ
            Ret->>Chroma: Collection 'faq'.similarity_search(vector, k=3)
            Chroma-->>Ret: Top 3 FAQ documents
        and Query Tickets
            Ret->>Chroma: Collection 'tickets'.similarity_search(vector, k=3)
            Chroma-->>Ret: Top 3 resolved ticket documents
        and Query Guides
            Ret->>Chroma: Collection 'guides'.similarity_search(vector, k=3)
            Chroma-->>Ret: Top 3 PDF chunk documents
        end
        Ret->>Chain: Merged 9 Document objects
    end

    rect rgb(245, 255, 245)
        Note over Chain,Groq: Step 3: Prompt Formatting & LLM Generation
        Chain->>Chain: _format_docs(docs) → Structured Markdown context
        Chain->>Chain: Inject context + question into System Prompt
        Chain->>Groq: HTTP POST /chat/completions (Stream = True, model="openai/gpt-oss-120b")
        loop Token Streaming
            Groq-->>Chain: Streaming text delta chunks
            Chain-->>UI: Yield token chunks
            UI-->>User: Render live markdown stream via st.write_stream()
        end
    end
```

---

## 6. Docker Networking, Storage & Secrets Architecture

### Networking (`telecom-rag_default`)
- **DNS Service Discovery**: Containers resolve each other using their Compose service names:
  - `rag-app` connects to `http://embeddings:8001` via `EMBEDDING_URL`.
  - `rag-app` connects to `http://chroma:8000` via `CHROMA_HOST` and `CHROMA_PORT`.
- **Port Exposure**:
  - Only port `8501` (`rag-app`) is bound to the host network (`"8501:8501"`), ensuring the internal vector database and embedding services remain inaccessible from outside the Docker host.

### Storage & Volumes
1. **Named Volume: `chroma-data`**:
   - Mapped to `/chroma/chroma` inside the `chroma` container.
   - Retains embedded collections across container restarts (`docker compose down && docker compose up`).
2. **Bind Mount: `./models:/models:ro`**:
   - Maps the local host directory `6-telecom-rag/models` directly into `/models` inside the `embeddings` container as read-only.
   - Eliminates model weight redownloads on container rebuilds and circumvents container-level corporate proxy restrictions.

### Secrets Management
- `GROQ_API_KEY` is loaded from `rag-app/.env` through Compose's `env_file` block:
  ```yaml
  env_file:
    - path: ./rag-app/.env
      required: false
  ```
- `.env` is explicitly ignored in `.gitignore` to prevent secret leakage.
