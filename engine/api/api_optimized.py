"""
Optimized API endpoints with caching and performance improvements.
This is a drop-in replacement for the original api.py with optimizations.
"""

import asyncio
import json
import logging
import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engine.core.logging_config import configure_logging
from engine.utils.latency import metrics_ctx
from engine.retrieval import intent_router
from engine.utils.query_cache import cached_rag_response, get_cache_stats

logger = logging.getLogger("engine.api.optimized")

# Configure diagnostic logging
_log_level = os.environ.get("LEGAL_RAG_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
configure_logging()
# Reduce noise from third-party libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(title="Legally RAG API (Optimized)", version="2.0")


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []
    stream: bool = False


class ChatResponse(BaseModel):
    result: str
    source_documents: List[dict]
    trace_report: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    cached: bool = False  # New field to indicate if response was from cache


# Cache statistics endpoint
@app.get("/api/v1/cache-stats")
async def get_cache_stats_endpoint():
    """Get cache statistics."""
    try:
        stats = get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/api/v1/clear-cache")
async def clear_cache_endpoint():
    """Clear all cache entries."""
    try:
        from engine.utils.query_cache import clear_cache
        count = clear_cache()
        return {"cleared_entries": count, "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        return {"error": str(e)}


def _is_simple_query(query: str) -> bool:
    """Check if query can use fast path (no reranking, minimal context)."""
    import re

    # Patterns for simple queries
    simple_patterns = [
        r'стать[ьяи]\s+\d+',          # Article lookup
        r'ст\.\s*\d+',                # Article shorthand
        r'что такое\s+[\w\s]+\?',     # Definition requests
        r'определение\s+[\w\s]+',     # Definition requests
        r'сколько\s+[\w\s]+\?',       # Quantity questions
    ]

    query_lower = query.lower()

    # Check patterns
    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            return True

    # Check length
    if len(query) < 30 and '?' in query:
        return True

    return False


def _get_optimized_config_for_query(query: str, intent: str) -> Dict[str, Any]:
    """Get optimized configuration based on query type."""
    config = {
        'use_reranker': True,
        'context_max_docs': 3,
        'retriever_wide_k': 16,
    }

    if _is_simple_query(query):
        # Fast path for simple queries
        config.update({
            'use_reranker': False,
            'context_max_docs': 2,
            'retriever_wide_k': 10,
        })

    elif intent == intent_router.CASE_SPECIFIC:
        # More context for case-specific queries
        config.update({
            'context_max_docs': 4,
            'retriever_wide_k': 20,
        })

    return config


@cached_rag_response
def get_cached_rag_response(query: str, intent: str, **kwargs) -> Dict[str, Any]:
    """Wrapper function for caching (decorator handles caching)."""
    from engine.retrieval import detective_mode, rag_chain, sherlock_engine

    # Get optimized config for this query
    optimized_config = _get_optimized_config_for_query(query, intent)

    # TODO: Pass optimized config to rag_chain
    # For now, use standard invocation

    if intent == intent_router.SOCIAL:
        response = rag_chain.invoke_qa(query, history=[], intent=intent)
    elif intent == intent_router.GENERAL_LEGAL:
        response = rag_chain.invoke_qa(query, history=[], intent=intent)
    elif intent == intent_router.PROCEDURAL:
        response = rag_chain.invoke_qa(query, history=[], intent=intent)
    else:
        response = detective_mode.invoke_detective_qa(query, history=[])

    return response


@app.post("/api/v1/internal-chat-fast", response_model=ChatResponse)
async def chat_fast(request: Request, body: ChatRequest):
    """Optimized chat endpoint with caching and fast paths."""
    logger.info(f"🚀 [FAST] Incoming Request: {body.query[:50]}...")

    metrics_ctx.set({})
    x_trace_id = request.headers.get("X-Trace-ID", f"trace_{int(time.time())}")
    start_time = time.perf_counter()

    try:
        # Intent classification
        routing_decision = intent_router.classify_intent_with_confidence(
            body.query, body.history or []
        )
        intent = routing_decision.intent

        # Check cache first (unless streaming)
        cache_key = None
        cached_response = None

        if not body.stream:
            from engine.utils.query_cache import generate_cache_key, cache_get
            cache_key = generate_cache_key(body.query, intent)
            cached_response = cache_get(cache_key)

        if cached_response:
            logger.info(f"✅ [CACHE HIT] for query: {body.query[:50]}...")

            # Update trace report with cache info
            trace_report = cached_response.get('trace_report') or {}
            trace_report.setdefault('metadata', {})['id'] = x_trace_id
            trace_report['metadata']['timestamp'] = datetime.now(timezone.utc).isoformat()
            trace_report['metadata']['routing'] = routing_decision.as_dict()
            trace_report['metadata']['cached'] = True

            ms = trace_report.get('metrics_ms') or {}
            ms['python_rag_total'] = int((time.perf_counter() - start_time) * 1000)
            if metrics_ctx.get():
                ms['breakdown'] = metrics_ctx.get()
            ms['breakdown']['cache_lookup'] = ms['python_rag_total']
            trace_report['metrics_ms'] = ms

            return ChatResponse(
                result=cached_response.get('result', ''),
                source_documents=cached_response.get('source_documents', []),
                trace_report=trace_report,
                confidence_score=cached_response.get('confidence_score', 0.0),
                cached=True
            )

        # Not in cache, process normally
        logger.info(f"🔄 [CACHE MISS] Processing query: {body.query[:50]}...")

        # Use optimized response function
        response = get_cached_rag_response(body.query, intent)

        # Prepare response
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
        trace_report["metadata"]["routing"] = routing_decision.as_dict()
        trace_report["metadata"]["cached"] = False
        trace_report["metadata"]["optimized"] = True

        ms = trace_report.get("metrics_ms") or {}
        ms["python_rag_total"] = int((time.perf_counter() - start_time) * 1000)
        if metrics_ctx.get():
            ms["breakdown"] = metrics_ctx.get()
        trace_report["metrics_ms"] = ms

        return ChatResponse(
            result=response.get("result", ""),
            source_documents=source_docs,
            trace_report=trace_report,
            confidence_score=response.get("confidence_score", 0.0),
            cached=False
        )

    except Exception as e:
        logger.error(f"Chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Keep original endpoints for compatibility
@app.post("/api/v1/internal-chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Original endpoint (redirects to optimized version)."""
    return await chat_fast(request, body)


@app.on_event("startup")
async def _warmup_rag():
    """Optimized warmup - load only essential components."""
    logger.info("🚀 [FAST START] Model Initialization")
    t0 = time.perf_counter()

    try:
        # Load only embeddings and vector store initially
        from engine.retrieval import rag_chain
        from engine.lifecycle_hooks import pre_flight_check

        pre_flight_check()

        # Load embeddings (essential for any query)
        rag_chain.get_embeddings()

        # Load vector store (essential for retrieval)
        rag_chain.get_vector_store()

        # LLM and retriever will load lazily on first request
        elapsed = time.perf_counter() - t0
        logger.info(f"✅ [FAST START] Model Initialization ({elapsed:.2f}s)")

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"❌ [FAST START FAIL] ({elapsed:.2f}s): {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=30)