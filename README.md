# Docker Tutorial 🐳

A hands-on, step-by-step Docker tutorial that starts with a 10-line FastAPI "hello world" and ends with a full **multi-container RAG chatbot** for a telecom customer-care use case. Each numbered folder introduces one Docker concept and builds on the previous one. Built with **FastAPI**, **Redis**, **Streamlit**, **LangChain + Groq**, **ChromaDB**, and **Sentence Transformers**.

---

## Tutorial Roadmap

| Folder | Concept | What you learn |
|---|---|---|
| `0-hello-api` | Baseline | A FastAPI app running locally, no Docker yet |
| `1-hello-api-docker` | Dockerfile basics | `FROM`, `WORKDIR`, `COPY`, `RUN`, `CMD`; build and run your first image |
| `2-layer-caching` | Layer caching | Copy `requirements.txt` before the code so rebuilds stay fast |
| `3-volumes` | Volumes | Persist a counter file outside the container with `-v` |
| `4-multi-stage-builds` | Multi-stage builds | Run tests in a builder stage and ship a lean runtime image |
| `5-docker-compose` | Docker Compose | FastAPI + Redis wired together with networks, volumes and `depends_on` |
| `6-telecom-rag` | Capstone project | Streamlit app + embedding service + ChromaDB, with an optional GPU profile |

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose v2)
- [Groq](https://console.groq.com/keys) API Key (only for `6-telecom-rag`)
- Python `>= 3.12` and [uv](https://docs.astral.sh/uv/) (only if you want to run `0-hello-api` locally)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (optional, GPU embeddings only)

### Steps 0 to 4: Single-container examples

```bash
# 0. Run locally, without Docker
cd 0-hello-api
pip install -r requirements.txt
uvicorn main:app --reload

# 1-3. Build and run an image
cd ../1-hello-api-docker
docker build -t hello-api .
docker run -p 8000:8000 hello-api

# 3. Persist data with a volume
cd ../3-volumes
docker build -t counter-api .
docker run -p 8000:8000 -v counter-data:/data counter-api

# 4. Multi-stage build (tests run during the build)
cd ../4-multi-stage-builds
docker build -t hello-api:multi-stage .
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to try the API.

### Step 5: FastAPI + Redis with Compose

```bash
cd 5-docker-compose
docker compose up --build

# Hit the counter a few times, then restart: the count survives
curl http://localhost:8000/count
docker compose down && docker compose up
```

Change the host port with `API_PORT=9000 docker compose up`.

### Step 6: Telecom RAG chatbot

A production-style multi-container RAG application with:
- **`rag-app`**: Streamlit frontend & LangChain RAG pipeline
- **`embeddings`**: FastAPI microservice serving `sentence-transformers/all-MiniLM-L6-v2` vectors
- **`chroma`**: ChromaDB vector store persisting data to a named Docker volume

#### 1. Generate a Free Groq API Key

Groq offers free, ultra-fast LLM inference with generous rate limits (no credit card required):
1. Sign up or log in at **[console.groq.com](https://console.groq.com)**.
2. Go to **API Keys** in the left sidebar (or visit **[console.groq.com/keys](https://console.groq.com/keys)**).
3. Click **Create API Key**, give it a name (e.g. `telecom-rag`), and copy the key (starts with `gsk_...`).
4. Configure your `.env` file:
   ```bash
   cd 6-telecom-rag
   cp rag-app/.env.example rag-app/.env
   ```
   Open `rag-app/.env` and paste your key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

#### 2. Set Up the Embedding Model (`models/`)

To guarantee fast, offline startup and avoid corporate proxy / SSL inspection issues inside Docker:

```bash
# Run the host helper script to download the model files to ./models
py -3.14 download_model_host.py
# or: python download_model_host.py
```

> [!TIP]
> **Behind a strict corporate proxy/firewall?**
> If your network blocks automated CLI downloads of the 90 MB binary weights file (`model.safetensors`):
> 1. Download it directly in your browser: [sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors)
> 2. Move the file to `6-telecom-rag/models/model.safetensors`.
> 
> The local `models/` directory is mounted into the `embeddings` container, allowing it to load the model 100% offline.

#### 3. Start the Services

```bash
# Build and start all services (embeddings, chroma, rag-app)
docker compose up --build
```

**With an NVIDIA GPU** (faster embeddings via CUDA):
```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

#### 4. Load Knowledge Base into ChromaDB (Run Once)

In a second terminal, ingest the three telecom data sources into ChromaDB:

```bash
cd 6-telecom-rag

# 1. Ingest general telecom FAQs from CSV
docker compose run --rm rag-app python -m src.ingest.faq

# 2. Ingest past resolved support tickets from SQLite
docker compose run --rm rag-app python -m src.ingest.tickets

# 3. Chunk and ingest the telecom PDF guide
docker compose run --rm rag-app python -m src.ingest.pdf
```

#### 5. Open the Chatbot

Open **[http://localhost:8501](http://localhost:8501)** in your browser and start asking questions!

---

## Architecture (`6-telecom-rag`)

```
 Browser ──▶ rag-app (Streamlit :8501)
                │  ├──▶ embeddings (FastAPI :8001)   all-MiniLM-L6-v2
                │  ├──▶ chroma     (:8000)           faq · tickets · guides
                │  └──▶ Groq API                     LLM answer
                ▼
          streamed response
```

| Service | Image | Role |
|---|---|---|
| `rag-app` | built from `rag-app/` | Streamlit UI, retriever and LangChain RAG chain |
| `embeddings` | built from `embedding-service/` | Serves `/embed` and `/health` so the model loads once |
| `chroma` | `chromadb/chroma` | Vector store, data kept in the `chroma-data` volume |

The retriever merges three Chroma collections: `faq` (CSV), `tickets` (SQLite) and `guides` (PDF chunks).

---

## Docker Concepts Covered

- Writing a minimal, readable `Dockerfile`
- Ordering instructions for layer caching
- `.dockerignore` to keep images small
- Named volumes and bind-mount persistence
- Multi-stage builds with tests as a build gate
- BuildKit cache mounts with `uv sync --frozen`
- Compose services, service-name networking, `depends_on`, `env_file`, `mem_limit`
- Compose override files (`compose.gpu.yaml`) and GPU device reservations

---

## Example Prompts

| Try asking… | What happens |
|---|---|
| *"Why is my mobile internet so slow?"* | Retrieves FAQ + ticket matches and gives troubleshooting steps |
| *"How do I activate international roaming?"* | Pulls the how-to from the telecom guide PDF |
| *"I was charged for roaming but had a bundle active"* | Finds similar resolved support tickets |

---

## Useful Commands

### 1. Compose Lifecycle & Operations

```bash
docker compose ps                               # List running services & health status
docker compose ps -a                            # List all services including stopped/exited
docker compose up -d                            # Start all services in the background (detached)
docker compose restart <service>                # Restart a single service (e.g. embeddings)
docker compose stop                             # Stop services without removing containers/networks
docker compose down                             # Stop and remove containers and network (keeps volumes)
docker compose down -v                          # Stop and delete containers, network, AND named volumes
```

### 2. Logs & Debugging

```bash
docker compose logs -f                          # Tail live logs for all running services
docker compose logs -f rag-app                  # Tail live logs for a specific service
docker compose logs --tail=100 embeddings       # Show last 100 log lines of a service

# Interactive shell inside a running container
docker compose exec -it rag-app bash            # Open bash in rag-app (or 'sh')
docker compose exec -it embeddings sh           # Open sh in embeddings

# Run a one-off command in a temporary container
docker compose run --rm rag-app python -V       # Check python version and remove container after
```

### 3. Monitoring & Resource Usage

```bash
docker stats                                    # Live stream of CPU, memory, net & disk I/O
docker compose top                              # Display running processes across compose services
docker compose port rag-app 8501                # See which host port maps to container port 8501
```

### 4. Disk Usage & Cleanup (Free Up Space)

```bash
# Check how much disk space Docker is using
docker system df

# Clean up stopped containers and dangling images (safe)
docker container prune -f                       # Remove all stopped containers
docker image prune -f                           # Remove dangling (<none>) images
docker builder prune -f                         # Clear BuildKit build cache

# Deeper cleanup
docker image prune -a -f                        # Remove all unused images not in active containers
docker volume prune -f                          # Remove all unused local volumes

# Full cleanup / Reset (removes all stopped containers, unused images, volumes, and cache)
docker system prune -a --volumes -f
```

### 5. Inspecting Images & Volumes

```bash
docker image ls                                 # List all local images and their sizes
docker volume ls                                # List all Docker volumes
docker network ls                               # List all Docker networks
docker inspect <container_or_image>             # Deep JSON inspection (IP, environment, mounts)
```
