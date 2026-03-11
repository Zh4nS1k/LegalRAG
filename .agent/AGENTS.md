---
description: Legally AI Agent — Roles, Workflows, and Behaviours
---

# Legally — Agent & AI Workflow Reference

This file documents all AI agents, ML pipelines, and workflow extension points in the Legally platform. It is the canonical reference for developers building on or maintaining the AI layer.

---

## Platform Architecture

```
React Frontend (3000)
      │ REST + JWT
      ▼
Go / Gin Backend (8080)   ←── MongoDB Atlas
      │ Internal REST (LAN)
      ▼
Python / FastAPI AI Engine (8000)
      ├── Pinecone Vector Store
      ├── Groq LLM (or Ollama)
      └── HuggingFace Embeddings (multilingual-e5-large)
```

The Go backend is the **public-facing orchestrator**. It handles auth, sessions, PDF parsing, and chat history. The Python engine is **internal-only** and must not be exposed publicly.

---

## Agent 1 — RAG Chat Agent

### Purpose
Answer legal questions grounded in the official Kazakhstani legal corpus with exact article citations. Supports Russian and Kazakh language queries.

### Trigger
```
POST /api/chat  (Go backend, authenticated)
    └─► POST /api/v1/internal-chat  (Python engine, internal)
```

### Full Pipeline

```
User Question
    │
    ▼
[1] Query Augmentation
    └── Appends legal synonyms, article ranges, equivalent Kazakh/Russian terms
    │
    ▼
[2] Hybrid Retrieval  (Pinecone 70% + BM25 30%)
    ├── Pinecone: semantic similarity (multilingual-e5-large, 1024-dim, cosine)
    │   Prefix: "query: {text}" for queries, "passage: {text}" for docs
    └── BM25: keyword match with Snowball Russian stemming (from chunks_for_bm25.pkl)
    │
    ▼
[3] Heuristic Filter Layer  (_HeuristicRetriever)
    └── Detects criminal law queries → enforces УК РК filter
        Detects article ranges → narrows to that range
        Detects topic focus → re-ranks by known article numbers
    │
    ▼
[4] Law-Aware Supplement Layer  (_LawAwareRetriever)
    └── If criminal query has <10 results → pulls extra from УК РК variants
        If смягчающие/отягчающие query → adds circumstance articles
    │
    ▼
[5] BGE-M3 Reranker  (BAAI/bge-reranker-v2-m3, FP16)
    └── Computes cross-encoder scores for all (query, doc) pairs
        Selects top-N by score (default: top-8)
    │
    ▼
[6] Context Trimmer  (_TrimRetriever)
    └── Max 8 docs × 1800 chars each → prevents LLM context overflow
    │
    ▼
[7] Prompt Selection
    ├── RANGE_PROMPT   — if query contains article range (e.g. ст. 120–135)
    ├── CRIMINAL_PROMPT — if query is about criminal law
    └── UNIVERSAL_PROMPT — all other queries (civil, tax, labour, family…)
    │
    ▼
[8] LLM Generation  (Groq: llama-3.1-8b-instant, or Ollama)
    └── Strictly citation-only response. Declines to speculate on missing context.
    │
    ▼
[9] Response
    └── { answer, source_documents, trace_report }
```

### Key Files
| File | Role |
|---|---|
| `ai_service/retrieval/rag_chain.py` | Full pipeline — retrievers, reranker, prompts, `invoke_qa()` |
| `ai_service/core/config.py` | Typed Pydantic config — loads from `.env` |
| `ai_service/api/api.py` | FastAPI endpoint handlers |

### Configuration Variables
| Variable | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | **required** | Pinecone authentication |
| `PINECONE_INDEX_NAME` | `legally-index` | Vector index name |
| `PINECONE_NAMESPACE` | `default` | Vector namespace |
| `GROQ_API_KEY` | **required** | Groq LLM authentication |
| `HF_TOKEN` | optional | HuggingFace token (prevents rate-limiting) |
| `LEGAL_RAG_LLM` | `llama-3.1-8b-instant` | LLM model name |
| `LEGAL_RAG_LLM_BACKEND` | `groq` | `groq` or `ollama` |
| `LEGAL_RAG_USE_RERANKER` | `1` | Enable BGE-M3 reranker |
| `LEGAL_RAG_CONTEXT_MAX_DOCS` | `5` | Max docs sent to LLM |
| `LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC` | `1200` | Max chars per doc |
| `HF_HUB_OFFLINE` | `0` | Set `1` to use cached model only |
| `LEGAL_RAG_HF_LOCAL_ONLY` | `0` | Same — venv-level override |

### Prompt Design Principles
- Always responds in the **same language** as the user's question (Russian/Kazakh)
- Never fabricates article numbers, sanctions, or dates not present in context
- Always cites: article number, code name, and source file
- Mandatory disclaimer: *"Это не официальная юридическая консультация."*
- Falls back to: *"Информация не найдена в доступных текстах законов."* when context lacks the answer

---

## Agent 2 — Contract Analysis Agent

### Purpose
Analyse uploaded PDF contracts for legal risks, clause compliance, document type classification, and missing clause detection — grounded in RK law.

### Trigger
```
POST /api/analyze  (Go backend, multipart PDF, authenticated)
    └─► Go analysis_service.go extracts text
        └─► POST /api/v1/analyze  (Python engine, plain text)
```

### Pipeline
```
PDF Upload
    │
    ▼
[1] Text Extraction  (Go — analysis_service.go)
    └── Reads raw text from uploaded PDF binary
    │
    ▼
[2] Type Classification  (LLM pass 1)
    └── Determines document type: Lease / Employment / Purchase / Service / NDA / Other
    │
    ▼
[3] Multi-Dimensional Analysis  (LLM pass 2)
    ├── Risk Assessment     — High / Medium / Low per clause
    ├── Legal Compliance    — RK code references, missing mandatory terms
    ├── Missing Clauses     — compares against standard clause checklist
    └── Executive Summary   — plain-language summary for non-lawyers
    │
    ▼
[4] Structured JSON Response
    └── { analysis, document_type, timestamp, filename }
    │
    ▼
[5] Persistence  (Go saves to MongoDB)
    └── Available in /api/history for the user
```

### Key Files
| File | Role |
|---|---|
| `ai_service/api/api.py` | `POST /api/v1/analyze` endpoint |
| `backend/legally/services/analysis_service.go` | Go orchestrator — PDF parsing + AI call |

---

## Agent 3 — HITL Evaluation Agent

Legally includes a Human-In-The-Loop system for rating AI response quality.

### Roles & Dashboards
| Role | URL | Responsibilities |
|---|---|---|
| `admin` | `/admin/eval` | Create tasks, assign to reviewers, export results as Excel |
| `professor` | `/reviewer/eval` | Rate AI answers (1–5), provide written feedback |
| `student` | `/reviewer/eval` | Rate AI answers (limited task assignment) |

### Workflow
```
Admin creates evaluation task
    └── question + AI-generated answer + source docs
          │
          ▼
Admin assigns to professor or student
          │
          ▼
Reviewer reads question + AI answer → submits score (1–5) + comment
          │
          ▼
Admin views aggregated results → exports CSV/Excel for analysis
```

### API Routes
| Method | Endpoint | Actor | Description |
|---|---|---|---|
| `GET/POST` | `/api/admin/tasks` | admin | List / create eval tasks |
| `PUT/DELETE` | `/api/admin/tasks/:id` | admin | Edit / delete a task |
| `POST` | `/api/admin/tasks/assign` | admin | Assign task to reviewer |
| `POST` | `/api/admin/tasks/upload/generate` | admin | Auto-generate questions from AI |
| `GET` | `/api/admin/eval/parsed` | admin | View parsed questions |
| `GET` | `/api/admin/eval/rated` | admin | View crowd-rated results |
| `GET` | `/api/admin/eval/export` | admin | Download results as Excel |
| `GET` | `/api/eval/my-tasks` | reviewer | Get assigned tasks |
| `POST` | `/api/eval/submit` | reviewer | Submit rating + feedback |

---

## Adding New Laws to the Corpus

1. Add the new law as a `.txt` file to `documents/`
2. Re-index into Pinecone:
   ```bash
   ./venv/bin/python ai_service/retrieval/build_vector_db.py
   ```
3. Rebuild the BM25 corpus:
   ```bash
   ./venv/bin/python -m ai_service.processing.prepare_data
   ```
4. Confirm vectors appear in Pinecone console
5. Restart the AI engine — the lazy Pinecone loader picks up new vectors automatically

---

## Extending the RAG Pipeline

### Change the LLM
Open `ai_service/core/config.py` → update `LLM_MODEL` default, or set `LEGAL_RAG_LLM` in `.env`.

### Change the Embedding Model
1. Open `ai_service/retrieval/rag_chain.py` — locate `_make_embeddings()`
2. Replace `config.EMBEDDING_MODEL` value in `core/config.py`
3. **Rebuild the Pinecone index** — embedding dimensions must match

### Disable the Reranker
```env
LEGAL_RAG_USE_RERANKER=0
```

### Add a New Retrieval Filter
```env
LEGAL_RAG_FILTER_CODE_RU="Трудовой кодекс РК"
LEGAL_RAG_FILTER_ARTICLE_NUMBER="52"
```

This pins all retrieval to that specific code/article (useful for domain-specific deployments).

### Smoke Testing After Changes
```bash
./venv/bin/python ai_service/utils/verify_langchain.py  # chain loads correctly
./venv/bin/python ai_service/utils/test_retrieval.py    # retrieval quality audit
./venv/bin/python -m ai_service.utils.benchmark          # full RAGAS benchmark
```

---

## Latency Profiling

Every response includes a `trace_report` JSON field:
```json
{
  "metrics_ms": {
    "python_rag_total": 6540,
    "breakdown": {
      "embedding": 120,
      "vector_search": 380,
      "llm_inference": 4900,
      "prompt_template_build": 2
    }
  }
}
```

Implemented via `@measure_latency` decorator in `ai_service/utils/latency.py`.
