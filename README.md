# ⚖️ Legally — AI-Powered Legal Assistant for the Republic of Kazakhstan

> **Legally** is a production-grade, full-stack Retrieval-Augmented Generation (RAG) platform purpose-built for Kazakhstani law. It enables legal professionals and citizens to query the official legal corpus of the RK, analyse PDF contracts, receive AI-generated answers grounded in real legislation — with exact article citations — and evaluate AI quality through a Human-in-the-Loop (HITL) workflow.

---

## 📚 Table of Contents

1. [System Overview & Architecture](#system-overview--architecture)
2. [Project Structure](#project-structure)
3. [AI Engine Deep-Dive (Python / FastAPI)](#ai-engine-deep-dive-python--fastapi)
4. [Backend Deep-Dive (Go / Gin)](#backend-deep-dive-go--gin)
5. [Frontend Deep-Dive (React)](#frontend-deep-dive-react)
6. [Legal Corpus Coverage](#legal-corpus-coverage)
7. [Prerequisites](#prerequisites)
8. [Environment Configuration](#environment-configuration)
9. [Running the Application](#running-the-application)
10. [Building the Vector Database (First-Time Setup)](#building-the-vector-database-first-time-setup)
11. [API Reference](#api-reference)
12. [User Roles & Permissions](#user-roles--permissions)
13. [Benchmarking & Evaluation](#benchmarking--evaluation)
14. [Troubleshooting](#troubleshooting)
15. [Security & Compliance](#security--compliance)

---

## System Overview & Architecture

Legally is a **three-tier platform**. Each layer is independently deployable and communicates over internal REST APIs:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Browser  /  React 18 UI                              │
│                      http://localhost:3000                               │
│                                                                          │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌───────────────────┐ │
│  │  Login / Register   │  │  Chat Interface    │  │  Admin Dashboard  │ │
│  │  JWT Auth + Refresh │  │  RAG + Sources     │  │  HITL Evaluation  │ │
│  └─────────────────────┘  └────────────────────┘  └───────────────────┘ │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │ REST/JSON + JWT
┌─────────────────────────────────▼────────────────────────────────────────┐
│               Go / Gin Backend  (Orchestrator)                           │
│                   http://localhost:8080                                  │
│                                                                          │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Auth & JWT  │  │  MongoDB   │  │  PDF Parsing │  │  HITL Tasks   │  │
│  │  Middleware  │  │  Sessions  │  │  & Analysis  │  │  Management   │  │
│  └──────────────┘  └────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │ Internal REST (no auth, LAN-only)
┌─────────────────────────────────▼────────────────────────────────────────┐
│           Python / FastAPI  (AI Engine)                                  │
│                   http://localhost:8000                                  │
│                                                                          │
│  ┌────────────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   Hybrid RAG Pipeline  │  │  Groq LLM        │  │  Lazy Pinecone   │  │
│  │  Pinecone + BM25 +     │  │  llama-3.1-8b    │  │  Vector Store    │  │
│  │  BGE-M3 Reranker       │  │  (or Ollama)     │  │  (1024-dim)      │  │
│  └────────────────────────┘  └──────────────────┘  └──────────────────┘  │
│  ┌────────────────────────┐  ┌──────────────────┐                        │
│  │  HuggingFace Embeddings│  │  Pydantic Config │                        │
│  │  multilingual-e5-large │  │  & Env Loader    │                        │
│  └────────────────────────┘  └──────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────┘
```

| Layer | Tech Stack | Port |
|---|---|---|
| **Frontend** | React 18, Material-UI, React Router | 3000 |
| **Backend** | Go 1.20+, Gin, MongoDB + Atlas | 8080 |
| **AI Engine** | Python 3.12, FastAPI, LangChain, Pinecone, Groq | 8000 |

---

## Project Structure

```
LegalRAG/
│
├── ai_service/                 # Python AI Engine (FastAPI)
│   ├── api/
│   │   └── api.py              # FastAPI app — all AI endpoints
│   ├── core/
│   │   └── config.py           # Pydantic BaseSettings — typed config loader
│   ├── retrieval/
│   │   ├── rag_chain.py        # Core RAG pipeline — hybrid retrieval → LLM
│   │   └── build_vector_db.py  # One-time Pinecone indexing script
│   ├── processing/
│   │   ├── prepare_data.py     # Text chunking & BM25 corpus builder
│   │   └── fetch_adilet.py     # Adilet.zan.kz scraper (raw law text)
│   ├── utils/
│   │   ├── latency.py          # Request-level latency profiling decorator
│   │   ├── benchmark.py        # RAGAS / custom benchmark suite
│   │   ├── test_retrieval.py   # Manual retrieval quality auditing
│   │   └── verify_langchain.py # LangChain chain smoke test
│   ├── app.py                  # Streamlit standalone UI (optional)
│   ├── main.py                 # Uvicorn entry point
│   └── requirements.txt
│
├── backend/
│   └── legally/                # Go / Gin Backend (Orchestrator)
│       ├── api/
│       │   ├── routes.go       # Route registration (Gin)
│       │   └── controllers/    # Handler functions (auth, chat, admin…)
│       ├── services/           # Business logic layer
│       │   ├── chat_service.go
│       │   └── analysis_service.go
│       ├── models/             # MongoDB document structs
│       ├── middleware/         # JWT auth middleware
│       └── main.go
│
├── frontend/
│   └── legally-app/            # React 18 SPA
│       ├── src/
│       │   ├── components/     # ChatSection, UploadSection, ResultSection…
│       │   ├── pages/          # Login, Dashboard, Admin…
│       │   └── services/       # Axios API clients
│       └── package.json
│
├── documents/                  # Raw law text files from Adilet
├── .env                        # Secret keys (never commit)
├── .env.example                # Template for .env
└── README.md
```

---

## AI Engine Deep-Dive (Python / FastAPI)

### RAG Pipeline Architecture

The retrieval pipeline is multi-stage for maximum accuracy:

```
User Query
    │
    ▼
[Query Augmentation] ─── Key legal terms, article ranges, code names appended
    │
    ▼
[Hybrid Retriever: 70% Pinecone + 30% BM25]
    │   • Pinecone: semantic vector similarity (intfloat/multilingual-e5-large, 1024-dim)
    │   • BM25: exact keyword match with Snowball Russian stemming
    │
    ▼
[_HeuristicRetriever] ─── Criminal query detection, article range narrowing
    │
    ▼
[_LawAwareRetriever] ──── Force-supplements results for ≤10 criminal law docs
    │
    ▼
[BGE-M3 Reranker] ─────── BAAI/bge-reranker-v2-m3, FP16 scores, top-8 select
    │   (falls back to unranked if disabled or unavailable)
    │
    ▼
[_TrimRetriever] ──────── Truncate to max 8 docs × 1800 chars each
    │
    ▼
[Groq LLM: llama-3.1-8b-instant] ─── Strict prompt: cite only from context
    │
    ▼
Answer + Source Documents (JSON)
```

### Key Design Decisions

| Decision | Reason |
|---|---|
| **Lazy Pinecone initialization** | `PineconeVectorStore` connects only on first request, preventing startup crash on missing keys |
| **Pydantic `BaseSettings`** | Fails loudly with a clear message if required env vars are missing — not deep inside LangChain |
| **Prefix embeddings** (`query:` / `passage:`) | Required by `multilingual-e5-large` for correct cosine similarity |
| **Law-aware retrieval layers** | Prevent the LLM from hallucinating articles from the wrong code (e.g. returning civil law for a criminal query) |
| **Dynamic prompt selection** | Detects criminal, article-range, or general query and routes to the optimal few-shot prompt template |
| **Latency tracking** | Per-stage timing via `@measure_latency` decorator, returned in API trace reports |

### Configuration Reference (`ai_service/core/config.py`)

All values are loaded via **Pydantic `EngineSettings`** from the root `.env` file.

| Variable | Type | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | `str` | **required** | Pinecone cloud API key |
| `PINECONE_INDEX_NAME` | `str` | `legally-index` | Pinecone index name |
| `PINECONE_NAMESPACE` | `str` | `default` | Vector namespace |
| `GROQ_API_KEY` | `str` | **required** | Groq cloud LLM key |
| `HF_TOKEN` | `str?` | `None` | HuggingFace token (avoids rate limits) |
| `LEGAL_RAG_LLM` | `str` | `llama-3.1-8b-instant` | Groq model name |
| `LEGAL_RAG_LLM_BACKEND` | `str` | `groq` | `groq` or `ollama` |
| `LEGAL_RAG_USE_RERANKER` | `bool` | `1` | Enable BGE-M3 reranker |
| `LEGAL_RAG_CONTEXT_MAX_DOCS` | `int` | `5` | Max docs sent to LLM |
| `LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC` | `int` | `1200` | Max chars per doc |
| `HF_HUB_OFFLINE` | `bool` | `0` | Set `1` for cached-model offline mode (faster boot) |
| `LEGAL_RAG_HF_LOCAL_ONLY` | `bool` | `0` | Force local HuggingFace cache only |

### Retriever Filters (Advanced)

To narrow retrieval to a specific code/article (useful for debugging or testing):

```bash
# Only retrieve from the Criminal Code, Article 136
LEGAL_RAG_FILTER_CODE_RU="Уголовный кодекс РК"
LEGAL_RAG_FILTER_ARTICLE_NUMBER="136"
```

### Module Reference

| Module | Purpose |
|---|---|
| `ai_service/api/api.py` | FastAPI app, endpoint handlers, numpy-type serialization |
| `ai_service/core/config.py` | Typed Pydantic settings, BASE_DIR resolution, HF hub configuration |
| `ai_service/retrieval/rag_chain.py` | Full RAG pipeline: embeddings, vector store lazy init, BM25, reranker, LLM, QA chains |
| `ai_service/retrieval/build_vector_db.py` | Indexes legal documents into Pinecone (run once) |
| `ai_service/processing/prepare_data.py` | Chunks law text for BM25 retriever, exports `chunks_for_bm25.pkl` |
| `ai_service/processing/fetch_adilet.py` | Scrapes current law text from `adilet.zan.kz` |
| `ai_service/utils/latency.py` | `@measure_latency` decorator, per-request timing context |
| `ai_service/utils/benchmark.py` | Full RAGAS-style benchmark suite with faithfulness, recall, precision metrics |

---

## Backend Deep-Dive (Go / Gin)

### Service Architecture

```
MongoDB Atlas
     │
     ▼                                        Python AI Engine
┌─────────────────────────────────┐                │
│  Go / Gin (Port 8080)           │◄───REST(8000)──┤
│                                 │                │
│  ┌──────────────────────────┐   │   /api/v1/internal-chat
│  │  JWT Middleware           │   │   /api/v1/analyze
│  │  (Auth on all /api/…)    │   │   /api/v1/stats
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  chat_service.go         │◄──┼── Orchestrates multi-turn chat sessions
│  │  analysis_service.go     │◄──┼── Uploads PDF, calls /analyze, stores result
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

### Route Map

#### Public (No Auth)
| Method | Endpoint | Controller | Description |
|---|---|---|---|
| `POST` | `/api/register` | `Register` | Create new user account |
| `POST` | `/api/login` | `Login` | Login with email + password, returns JWT pair |
| `POST` | `/api/refresh` | `Refresh` | Rotate access token using refresh token |
| `GET` | `/api/validate-token` | `ValidateToken` | Check token validity |
| `GET` | `/api/laws` | `GetRelevantLaws` | List indexed legal documents |
| `GET` | `/api/stats` | `GetSystemStats` | Returns vector count, active models |
| `GET` | `/health` | inline | Health check |

#### Private (JWT Required)
| Method | Endpoint | Controller | Description |
|---|---|---|---|
| `GET` | `/api/user` | `GetUser` | Get current user profile |
| `POST` | `/api/analyze` | `AnalyzeDocuments` | Upload PDF → AI analysis → stored result |
| `GET` | `/api/history` | `GetHistory` | User's past PDF analysis results |
| `POST` | `/api/chat` | `HandleChat` | Send message, get RAG answer |
| `GET` | `/api/chat/history` | `GetChatHistory` | Retrieve current session history |
| `DELETE` | `/api/chat/history` | `ClearChatHistory` | Delete current session |
| `GET` | `/api/chat/export` | `ExportChatHistory` | Export session as text |
| `POST` | `/api/logout` | `Logout` | Invalidate session |
| `POST` | `/api/analysis/cancel` | `CancelAnalysis` | Cancel in-progress analysis |
| `POST` | `/api/cache/clear` | `ClearFileCache` | Clear server-side file cache |

#### Admin (Admin Role)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/admin/users` | List all registered users |
| `POST` | `/api/admin/users` | Create user account manually |
| `DELETE` | `/api/admin/users/:id` | Remove user account |
| `POST` | `/api/admin/users/role` | Change a user's role |
| `GET/POST` | `/api/admin/tasks` | List / create HITL evaluation tasks |
| `GET/PUT/DELETE` | `/api/admin/tasks/:id` | Get / update / delete specific task |
| `POST` | `/api/admin/tasks/assign` | Assign task to a reviewer |
| `POST` | `/api/admin/tasks/upload/generate` | Auto-generate eval questions from AI |
| `POST` | `/api/admin/tasks/upload/ready` | Mark generated questions as ready |
| `GET` | `/api/admin/eval/parsed` | Fetch all parsed evaluation questions |
| `GET` | `/api/admin/eval/rated` | Fetch all rated results |
| `GET` | `/api/admin/eval/export` | Export rated results as Excel |

#### Reviewer (Professor / Student roles)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/eval/my-tasks` | Get tasks assigned to this reviewer |
| `POST` | `/api/eval/submit` | Submit evaluation scores and feedback |

---

## Frontend Deep-Dive (React)

### Key Components

| Component | Location | Purpose |
|---|---|---|
| `ChatSection.js` | `src/components/` | Main chat interface — sends query, renders markdown, shows source citations |
| `UploadSection.js` | `src/components/` | PDF drag-drop upload, triggers analysis pipeline, progress feedback |
| `ResultSection.js` | `src/components/` | Renders analysis results with risk indicators and article references |
| `LoginPage.js` | `src/pages/` | JWT login form with refresh token management |
| `AdminDashboard.js` | `src/pages/` | User management, HITL task creation and assignment |
| `EvaluationPage.js` | `src/pages/` | Reviewer interface for rating AI responses |

### Auth Flow

```
User logs in → POST /api/login
    → receives { access_token, refresh_token }
    → access_token stored in memory (not localStorage for XSS safety)
    → refresh_token stored in httpOnly cookie
    → On 401 → auto-retry with POST /api/refresh
    → On refresh failure → redirect to login
```

---

## Legal Corpus Coverage

Legally indexes **19 core laws** sourced directly from [adilet.zan.kz](https://adilet.zan.kz):

| # | File | Law |
|---|---|---|
| 1 | `constitution.txt` | Constitution of the Republic of Kazakhstan |
| 2 | `civil_code.txt` | Civil Code RK — General Part |
| 3 | `civil_code2.txt` | Civil Code RK — Special Part |
| 4 | `labor_code.txt` | Labour Code RK |
| 5 | `tax_code.txt` | Tax Code RK |
| 6 | `code_of_administrative_offenses.txt` | Code of Administrative Offences RK (КоАП) |
| 7 | `criminal_code.txt` | Criminal Code RK (УК РК) |
| 8 | `code_on_marriage_and_family.txt` | Code on Marriage and Family RK |
| 9 | `code_on_public_health.txt` | Code on Public Health RK |
| 10 | `entrepreneurial_code.txt` | Entrepreneurial Code RK |
| 11 | `code_on_administrative_procedures.txt` | Code on Administrative Procedures RK |
| 12 | `social_code.txt` | Social Code RK |
| 13 | `civil_procedure_code.txt` | Civil Procedure Code RK (ГПК РК) |
| 14 | `criminal_procedure_code.txt` | Criminal Procedure Code RK (УПК РК) |
| 15 | `law_on_public_procurement.txt` | Law on Public Procurement |
| 16 | `law_on_anticorruption.txt` | Law on Countering Corruption |
| 17 | `law_on_enforcement.txt` | Law on Enforcement Proceedings |
| 18 | `law_on_personal_data.txt` | Law on Personal Data |
| 19 | `law_on_ai.txt` | Law on Artificial Intelligence (2025) |

> **Bilingual support:** All queries in Russian and Kazakh are processed. Query augmentation adds equivalent legal terms in the other language automatically.

---

## Prerequisites

| Tool | Minimum Version | Install command |
|---|---|---|
| Python | 3.12 | `python --version` |
| Go | 1.20+ | `go version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |

**External Cloud Services (API keys required):**

| Service | Used For | URL |
|---|---|---|
| [Pinecone](https://app.pinecone.io) | Vector search index | Free tier available |
| [Groq](https://console.groq.com) | LLM inference (llama-3.x) | Free tier available |
| [HuggingFace](https://huggingface.co) | Embedding model download | Free token |
| [MongoDB Atlas](https://cloud.mongodb.com) | User/session data | Free 512MB cluster |

---

## Environment Configuration

Copy `.env.example` to `.env` in the project root and fill in all values:

```bash
cp .env.example .env
```

```env
# ── AI & Vector Store ──────────────────────────────────────────────────────
PINECONE_API_KEY="pcsk_your_key_here"
PINECONE_INDEX_NAME="legally-index"
PINECONE_NAMESPACE="default"

GROQ_API_KEY="gsk_your_groq_key_here"

# Optional but strongly recommended — avoids anonymous HF Hub rate limits
HF_TOKEN="hf_your_token_here"

# ── Performance Toggles ────────────────────────────────────────────────────
# After first model download, set both to 1 for instant offline startup:
HF_HUB_OFFLINE="0"
LEGAL_RAG_HF_LOCAL_ONLY="0"

# Disable reranker (useful if FlagEmbedding / transformers version mismatch):
# LEGAL_RAG_USE_RERANKER=0

# ── Database ───────────────────────────────────────────────────────────────
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/?appName=Cluster0"
DB_NAME="legally_bot"

# ── Go Backend Auth ────────────────────────────────────────────────────────
JWT_SECRET="your_long_random_jwt_secret_here"
ADMIN_IDS="991315506"
```

> ⚠️ **Never commit `.env` to version control.** It is listed in `.gitignore`.

---

## Running the Application

Run each service **in a separate terminal**, in this order.

### Step 1 — AI Engine (Python / FastAPI)

```bash
# Create and activate the virtual environment (first time only)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the AI engine
./venv/bin/uvicorn ai_service.api.api:app --reload --port 8000
```

**Healthy startup output:**
```
Pinecone подключён: legally-index
LLM: Groq (model=llama-3.1-8b-instant)
INFO: Application startup complete.
```

**Common startup warnings (safe to ignore):**

| Warning | Meaning | Action |
|---|---|---|
| `The following layers were not sharded` | Harmless architecture metadata from sentence-transformers | None |
| `embeddings.position_ids \| UNEXPECTED` | Known architecture mismatch in multilingual-e5-large | None |
| `Sending unauthenticated requests to HF Hub` | No HF_TOKEN set | Add `HF_TOKEN` to `.env` |
| `BM25 не запустился: Нет чанков` | BM25 pickle not built yet | Run `build_vector_db.py` (see below) |
| `Reranker BGE-M3 не запустился` | FlagEmbedding/transformers version mismatch | Set `LEGAL_RAG_USE_RERANKER=0` or upgrade |

### Step 2 — Go Backend

```bash
cd backend/legally
go run main.go
```

**Healthy startup output:**
```
✅ MongoDB подключена
✅ Сервер запущен на http://localhost:8080
```

### Step 3 — React Frontend

```bash
cd frontend/legally-app
npm install        # first time only
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Building the Vector Database (First-Time Setup)

Before the hybrid retriever can work at full capacity you need to:

**1. Fetch the latest law texts from Adilet** *(optional — files may already exist in `documents/`)*:
```bash
./venv/bin/python -m ai_service.processing.fetch_adilet
```

**2. Chunk and index into Pinecone + build BM25 corpus:**
```bash
./venv/bin/python ai_service/retrieval/build_vector_db.py
```

**Expected output:**
```
[INFO] Connecting to Pinecone index: legally-index
[INFO] Loading 19 legal documents from /documents/...
[INFO] Chunking documents (chunk_size=1000, overlap=200)...
[INFO] Total chunks: ~4,200
[INFO] Uploading vectors to Pinecone...
[INFO] Saving chunks_for_bm25.pkl for BM25 retriever...
[SUCCESS] Vector database ready. Total vectors: 4200
```

> ✅ Only needs to run once unless the legal corpus is updated.

---

## API Reference — AI Engine (Port 8000)

### `POST /api/v1/internal-chat`

Called by the Go backend to run a RAG query.

**Request:**
```json
{
  "query": "Что грозит за подмену ребёнка?",
  "history": [
    { "role": "user", "content": "Привет" },
    { "role": "assistant", "content": "Здравствуйте!" }
  ]
}
```

**Response:**
```json
{
  "result": "Это не официальная юридическая консультация...\nСогласно ст. 136 УК РК...",
  "source_documents": [
    {
      "page_content": "Статья 136. Подмена ребёнка...",
      "metadata": {
        "source": "criminal_code.txt",
        "article_number": "136",
        "code_ru": "Уголовный кодекс РК"
      }
    }
  ],
  "trace_report": {
    "metadata": { "id": "trace_1234", "timestamp": "2026-03-12T00:00:00.000Z" },
    "metrics_ms": {
      "python_rag_total": 6540,
      "breakdown": {
        "embedding": 120,
        "vector_search": 380,
        "llm_inference": 4900
      }
    }
  }
}
```

### `POST /api/v1/analyze`

Analyze a legal text excerpt.

**Request:**
```json
{ "text": "Договор купли-продажи квартиры..." }
```

**Response:**
```json
{ "result": "Анализ: контракт содержит..." }
```

### `GET /api/v1/stats`

Returns index metadata.

**Response:**
```json
{
  "total_vectors": 4200,
  "index_dimension": 1024,
  "models": {
    "embedding": "multilingual-e5-large",
    "reranker": "BAAI/bge-reranker-v2-m3"
  }
}
```

### `POST /api/v1/generate-eval-data`

Generate evaluation dataset entries (admin use).

```json
{ "query": "Каковы права работника при увольнении?" }
```

Returns: `{ "answer", "chunks", "articles" }`

---

## User Roles & Permissions

| Role | Access Level | Capabilities |
|---|---|---|
| `user` | Standard | Upload PDFs, use chat, view own history |
| `admin` | Full | All user capabilities + user management + HITL admin |
| `professor` | Reviewer | HITL evaluation tasks (rating AI responses on quality) |
| `student` | Reviewer (limited) | HITL evaluation tasks (subset of task types) |

---

## Benchmarking & Evaluation

### Automated Retrieval Benchmark

Run the full benchmark suite against the test question set:

```bash
./venv/bin/python -m ai_service.utils.benchmark
```

Outputs an Excel report to `benchmark_results/` with per-question metrics:
- **Faithfulness** — Is the answer supported by the retrieved context?
- **Context Recall** — Were the correct articles retrieved?
- **Context Precision** — Were irrelevant docs included?
- **Answer Relevance** — Does the answer address the question?

### Manual Retrieval Audit

```bash
./venv/bin/python ai_service/utils/test_retrieval.py
```

### Verify LangChain Chain

```bash
./venv/bin/python ai_service/utils/verify_langchain.py
```

---

## Troubleshooting

### Startup & Runtime

| Symptom | Cause | Fix |
|---|---|---|
| `[CRITICAL ERROR] Missing Configuration: PINECONE_API_KEY` | `.env` not found or key missing | Check `BASE_DIR` in `config.py`, ensure `.env` is in project root |
| `SystemExit: Задайте GROQ_API_KEY` | Pydantic loaded key but LLM code couldn't read it | Ensure `config.GROQ_API_KEY` is used (not `os.environ.get`) |
| `Error loading ASGI app. Could not import module "api"` | Wrong uvicorn command (old path) | Use `./venv/bin/uvicorn ai_service.api.api:app` |
| `ModuleNotFoundError: No module named 'fastapi'` | System uvicorn used instead of venv | Always use `./venv/bin/uvicorn` |
| `is_torch_fx_available` import error | FlagEmbedding/transformers version clash | Set `LEGAL_RAG_USE_RERANKER=0` or run `pip install --upgrade transformers FlagEmbedding` |
| `ValueError: Pinecone API key must be provided` | Old code path before lazy loading | Confirm `rag_chain.py` uses `get_vector_store()` function, not module-level init |

### Chat Quality

| Symptom | Cause | Fix |
|---|---|---|
| Empty / hallucinated answers | BM25 not initialized (no chunks pickle) | Run `build_vector_db.py` to generate `chunks_for_bm25.pkl` |
| Wrong law code returned | Retriever not filtering by code | Enable `LEGAL_RAG_FILTER_CODE_RU` for single-code testing |
| Slow responses (>10s) | LLM inference bottleneck | Switch to a smaller Groq model (`llama-3.1-8b-instant`) or use Ollama locally |
| "Информация не найдена" on valid queries | Document not indexed | Check `documents/` folder and rerun `build_vector_db.py` |

### Frontend / Backend

| Symptom | Cause | Fix |
|---|---|---|
| Login fails silently | Go backend not running | Start `go run main.go` first |
| Chat returns empty | AI engine not running or Pinecone timeout | Start `uvicorn`, check Pinecone key |
| PDF upload hangs | AI engine unreachable from Go | Verify both services are on correct ports; check `AI_ENGINE_URL` in Go config |
| CORS errors in browser | Frontend domain not whitelisted | Update CORS origins in Go backend config |

---

## Security & Compliance

### AI Transparency (Law on Artificial Intelligence, RK 2025)

Every response includes a mandatory disclosure:

> *"Это не официальная юридическая консультация. Информация только из базы."*

This satisfies Article X of Kazakhstan's AI Transparency Law requiring AI-generated legal content to be clearly labelled.

Sources are always cited. The system **refuses to fabricate** article numbers, sanctions, or dates not present in the indexed corpus.

### Data Security

- All API keys stored in `.env` — never in code or git history
- JWT access tokens expire in 15 minutes; refresh tokens rotated on use
- PDF uploads processed in-memory; no files persisted server-side beyond the analysis session
- MongoDB Atlas provides encryption at rest + in transit
- Pinecone data encrypted at rest (managed by Pinecone cloud)

### Rate Limiting Recommendations (Production)

- Add Nginx/Caddy reverse proxy with rate limiting on `/api/chat` (e.g., 10 req/min per user)
- Enable Groq's account-level rate limiting to control LLM costs
- Use HuggingFace token and set `HF_HUB_OFFLINE=1` after initial model download to prevent quota exhaustion

---

*Legally is an AI assistant. Always verify legal conclusions with a qualified lawyer.*
