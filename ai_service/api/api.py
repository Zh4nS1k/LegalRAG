import logging
import os
import time
import uvicorn
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from ai_service.utils.latency import metrics_ctx

logger = logging.getLogger("ai_service.api")

# Configure diagnostic logging (granular step-by-step for tracing hangs/timeouts)
_log_level = os.environ.get("LEGAL_RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Reduce noise from third-party libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(title="Legally RAG API", version="1.0")


@app.on_event("startup")
async def _warmup_rag():
    """Load all RAG components before accepting connections. Server binds after warmup (~2–3 min)."""
    logger.info("[START] Model Initialization")
    t0 = time.perf_counter()
    try:
        # Level 1: Hooks - Absolute Guarantee
        from ai_service.lifecycle_hooks import pre_flight_check
        pre_flight_check()

        from ai_service.retrieval import rag_chain

        rag_chain.get_embeddings()
        rag_chain.get_vector_store()
        rag_chain.get_retriever()
        rag_chain.get_llm()
        elapsed = time.perf_counter() - t0
        logger.info("[SUCCESS] Model Initialization (%.2fs)", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("[FAIL] Model Initialization (%.2fs): %s", elapsed, e, exc_info=True)
        raise


@app.get("/health")
async def health():
    """Health check: 200 when ready (server only binds after warmup)."""
    return {"status": "ok"}


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class SourceDocument(BaseModel):
    page_content: str
    metadata: dict

class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    result: str

class ChatResponse(BaseModel):
    result: str
    source_documents: List[dict]
    trace_report: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    missing_fields: Optional[List[str]] = None
    clarifying_questions: Optional[List[str]] = None

import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

@app.post("/api/v1/internal-chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    logger.info("[START] Incoming Request Parsing")
    metrics_ctx.set({})
    x_trace_id = request.headers.get("X-Trace-ID", f"trace_{int(time.time())}")
    start_time = time.perf_counter()
    try:
        _query = body.query
        _history = body.history or []
        logger.info("[SUCCESS] Request Parsed (query_len=%d, history_len=%d)", len(_query), len(_history))
    except Exception as e:
        logger.error("Request parsing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

    try:
        from ai_service.retrieval import detective_mode

        # Detective Mode: completeness → agentic (HyDE, Censor, Self-RAG, CoVe) → causality/skeptic/flip → confidence
        response = detective_mode.invoke_detective_qa(
            body.query,
            history=body.history,
            trace_id=x_trace_id,
        )
        result = response.get("result", "")
        source_docs = []
        for doc in response.get("source_documents", []):
            if hasattr(doc, "metadata"):
                metadata = convert_numpy_types(doc.metadata)
                source_docs.append({"page_content": doc.page_content, "metadata": metadata})
            else:
                source_docs.append(convert_numpy_types(doc))

        trace_report = response.get("trace_report") or {}
        trace_report.setdefault("metadata", {})["id"] = x_trace_id
        trace_report["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
        ms = trace_report.get("metrics_ms") or {}
        ms["python_rag_total"] = int((time.perf_counter() - start_time) * 1000)
        if metrics_ctx.get():
            ms["breakdown"] = metrics_ctx.get()
        trace_report["metrics_ms"] = ms

        return ChatResponse(
            result=result,
            source_documents=source_docs,
            trace_report=trace_report,
            confidence_score=response.get("confidence_score", 0.0),
            missing_fields=response.get("missing_fields") or [],
            clarifying_questions=response.get("clarifying_questions"),
        )
    except Exception as e:
        logger.error("Step failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        from ai_service.retrieval import rag_chain

        result = rag_chain.analyze_text(request.text)
        return AnalysisResponse(result=result)
    except Exception as e:
        logger.error("Analysis step failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats():
    try:
        from ai_service.retrieval import rag_chain
        # Get stats from Pinecone index via LangChain vectorstore
        # Note: This depends on the specific vectorstore implementation
        # For Pinecone, we can access the index directly
        stats = {
            "total_vectors": 0,
            "index_dimension": 0,
            "models": {
                "embedding": "multilingual-e5-large",
                "reranker": "BAAI/bge-reranker-v2-m3" if rag_chain.config.USE_RERANKER else "None"
            }
        }
        
        try:
            vs = rag_chain.get_vector_store()
            if hasattr(vs, "_index"):
                index_stats = vs._index.describe_index_stats()
                stats["total_vectors"] = index_stats.get("total_vector_count", 0)
                stats["index_dimension"] = index_stats.get("dimension", 0)
        except Exception as e:
            logger.error("Failed to get Pinecone stats: %s", e, exc_info=True)

        return stats
    except Exception as e:
        logger.error("Stats fetch failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate-eval-data")
async def generate_eval_data(request: ChatRequest):
    try:
        from ai_service.retrieval import rag_chain
        # We reuse ChatRequest (query: str) for simpler reuse
        response = rag_chain.invoke_qa(request.query)
        
        result = response.get("result", "")
        chunks = [doc.page_content for doc in response.get("source_documents", [])]
        articles = []
        for doc in response.get("source_documents", []):
            code = doc.metadata.get("code_ru", "Неизвестно")
            article = doc.metadata.get("article_number", "")
            articles.append(f"{code} ст.{article}" if article else code)

        return {
            "answer": result,
            "chunks": chunks,
            "articles": articles
        }
    except Exception as e:
        logger.error("Generate eval data failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
