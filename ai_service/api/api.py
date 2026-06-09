import asyncio
import json
import logging
import os
import time
import uvicorn
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ai_service.core.logging_config import configure_logging
from ai_service.utils.latency import metrics_ctx
from ai_service.retrieval import intent_router

logger = logging.getLogger("ai_service.api")


def _is_connection_failure(exc: Exception) -> bool:
    name = exc.__class__.__name__
    return isinstance(exc, ConnectionError) or name.endswith("ConnectionError") or name.endswith("ConnectError")


def _raise_http_error(
    message: str,
    exc: Exception,
    *,
    status_code: int = 500,
    detail: str = "Internal server error",
) -> None:
    logger.error(message, exc_info=True)
    if _is_connection_failure(exc):
        from ai_service.retrieval.rag_chain import reset_instances

        reset_instances()
    raise HTTPException(status_code=status_code, detail=detail)

# Configure diagnostic logging (granular step-by-step for tracing hangs/timeouts)
_log_level = os.environ.get("LEGAL_RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
configure_logging()
# Reduce noise from third-party libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(title="Legally RAG API", version="1.0")


@app.on_event("startup")
async def _warmup_rag():
    """Load all RAG components before accepting connections. Server binds after warmup (~2–3 min)."""
    logger.info("🚀 [START] Model Initialization")
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
        logger.info("✅ [SUCCESS] Model Initialization (%.2fs)", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("❌ [FAIL] Model Initialization (%.2fs): %s", elapsed, e, exc_info=True)
        raise


@app.on_event("shutdown")
async def _graceful_shutdown():
    logger.info("🛑 Shutdown signal received, allowing in-flight requests to finish")
    await asyncio.sleep(2)
    logger.info("🛑 Shutdown complete")


@app.get("/health")
async def health():
    """Readiness check: verifies the vector store can be reached."""
    checks: dict[str, Any] = {}
    try:
        from ai_service.retrieval import rag_chain

        vector_store = rag_chain.get_vector_store()
        if hasattr(vector_store, "_index"):
            vector_store._index.describe_index_stats()
        checks["pinecone"] = "ok"
    except Exception as e:
        checks["pinecone"] = f"error: {e}"

    try:
        from ai_service.retrieval import rag_chain

        rag_chain.get_llm()
        checks["llm"] = "ok"
    except Exception as e:
        checks["llm"] = f"error: {e}"

    try:
        from ai_service.retrieval import rag_chain

        checks["circuit_breakers"] = rag_chain.get_breaker_states()
    except Exception as e:
        checks["circuit_breakers"] = {"error": str(e)}

    all_ok = checks.get("pinecone") == "ok" and checks.get("llm") == "ok"
    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )


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
    deductive_block: Optional[Dict[str, Any]] = None
    deductive_output: Optional[str] = None


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


async def _run_chat_pipeline(
    body: ChatRequest,
    intent: str,
    routing_decision: dict[str, Any],
    x_trace_id: str,
    start_time: float,
) -> ChatResponse:
    from ai_service.retrieval import detective_mode, rag_chain, sherlock_engine

    sherlock_res = None
    if intent == intent_router.SOCIAL:
        response = await asyncio.to_thread(
            rag_chain.invoke_qa, body.query, history=body.history, intent=intent
        )
    elif intent == intent_router.GENERAL_LEGAL:
        response = await asyncio.to_thread(
            rag_chain.invoke_qa, body.query, history=body.history, intent=intent
        )
        sherlock = sherlock_engine.SherlockEngine()
        sherlock_res = await sherlock.run_sherlock_loop(body.query)
    elif intent == intent_router.PROCEDURAL:
        response = await asyncio.to_thread(
            rag_chain.invoke_qa, body.query, history=body.history, intent=intent
        )
    else:
        response = await detective_mode.invoke_detective_qa(
            body.query,
            history=body.history,
            trace_id=x_trace_id,
        )
        sherlock = sherlock_engine.SherlockEngine()
        sherlock_res = await sherlock.run_sherlock_loop(body.query)

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
    trace_report["metadata"]["routing"] = routing_decision
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
        deductive_output=(sherlock_res.get("deductive_output") if sherlock_res else None),
    )


@app.post("/api/v1/internal-chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    logger.info("🚀 [START] Incoming Request Parsing")
    metrics_ctx.set({})
    x_trace_id = request.headers.get("X-Trace-ID", f"trace_{int(time.time())}")
    start_time = time.perf_counter()
    try:
        _query = body.query
        _history = body.history or []
        routing_decision = intent_router.classify_intent_with_confidence(_query, _history)
        intent = routing_decision.intent
        logger.info(
            "✅ [SUCCESS] Request Parsed (query_len=%d, history_len=%d, intent=%s, confidence=%.3f)",
            len(_query),
            len(_history),
            intent,
            routing_decision.confidence,
        )
    except Exception as e:
        _raise_http_error(
            "Request parsing failed",
            e,
            status_code=400,
            detail="Invalid request",
        )

    try:
        return await _run_chat_pipeline(
            body,
            intent,
            routing_decision.as_dict(),
            x_trace_id,
            start_time,
        )
    except Exception as e:
        _raise_http_error("Chat request failed", e)


@app.post("/api/v1/chat-stream")
async def chat_stream(request: Request, body: ChatRequest):
    logger.info("🚀 [START] Incoming Stream Request Parsing")
    metrics_ctx.set({})
    x_trace_id = request.headers.get("X-Trace-ID", f"trace_{int(time.time())}")
    try:
        routing_decision = intent_router.classify_intent_with_confidence(
            body.query, body.history or []
        )
        intent = routing_decision.intent
    except Exception as e:
        _raise_http_error(
            "Stream request parsing failed",
            e,
            status_code=400,
            detail="Invalid request",
        )

    async def generate():
        started_at = time.perf_counter()
        from ai_service.retrieval import rag_chain

        yield f"event: status\ndata: {json.dumps({'status': 'started', 'intent': intent}, ensure_ascii=False)}\n\n"
        try:
            yield (
                "event: status\ndata: "
                + json.dumps({"status": "retrieving_context"}, ensure_ascii=False)
                + "\n\n"
            )
            stream_payload = rag_chain.build_streaming_qa_prompt(
                body.query,
                history=body.history,
                intent=intent,
            )
            chunks: list[str] = []
            last_ping = time.perf_counter()
            async for chunk in stream_payload["llm"].astream(stream_payload["prompt_text"]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    chunks.append(token)
                    yield (
                        "event: token\ndata: "
                        + json.dumps({"token": token}, ensure_ascii=False)
                        + "\n\n"
                    )
                if time.perf_counter() - last_ping >= 1.0:
                    last_ping = time.perf_counter()
                    yield (
                        "event: ping\ndata: "
                        + json.dumps(
                            {
                                "status": "working",
                                "elapsed_ms": int(
                                    (time.perf_counter() - started_at) * 1000
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
            result = "".join(chunks).strip()
            yield (
                "event: final\ndata: "
                + json.dumps(
                    convert_numpy_types(
                        {
                            "result": result,
                            "source_documents": [
                                {
                                    "page_content": doc.page_content,
                                    "metadata": convert_numpy_types(doc.metadata),
                                }
                                for doc in stream_payload["source_documents"]
                            ],
                            "trace_report": {
                                "metadata": {
                                    "id": x_trace_id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "routing": routing_decision.as_dict(),
                                },
                                "metrics_ms": {
                                    "python_rag_total": int(
                                        (time.perf_counter() - started_at) * 1000
                                    )
                                },
                            },
                        }
                    ),
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        except Exception as e:
            logger.error("Chat stream failed: %s", e, exc_info=True)
            yield (
                "event: error\ndata: "
                + json.dumps(
                    {"detail": "Internal server error"},
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        from ai_service.retrieval import rag_chain

        result = rag_chain.analyze_text(request.text)
        return AnalysisResponse(result=result)
    except Exception as e:
        _raise_http_error("Analysis request failed", e)


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
                "reranker": (
                    "BAAI/bge-reranker-v2-m3"
                    if rag_chain.config.USE_RERANKER
                    else "None"
                ),
            },
        }

        try:
            vs = rag_chain.get_vector_store()
            if hasattr(vs, "_index"):
                index_stats = vs._index.describe_index_stats()
                stats["total_vectors"] = index_stats.get("total_vector_count", 0)
                stats["index_dimension"] = index_stats.get("dimension", 0)
        except Exception as e:
            logger.error("Failed to get Pinecone stats: %s", e, exc_info=True)
            if _is_connection_failure(e):
                from ai_service.retrieval.rag_chain import reset_instances

                reset_instances()

        return stats
    except Exception as e:
        _raise_http_error("Stats request failed", e)


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

        return {"answer": result, "chunks": chunks, "articles": articles}
    except Exception as e:
        _raise_http_error("Generate eval data failed", e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=30)
