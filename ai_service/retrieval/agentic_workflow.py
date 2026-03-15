# agentic_workflow.py — Board of Directors: Linguist-Analyst, Censor, Self-RAG, CoVe
# Trace_id is preserved through all modules for Go/Python boundary.

import logging
import re
import time
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from ai_service.core import config
from ai_service.retrieval import rag_chain

logger = logging.getLogger("ai_service.agentic")

# Config
TOP_K_CANDIDATES = getattr(config, "AGENTIC_TOP_K_CANDIDATES", 50)
RERANKER_TOP_N = getattr(config, "AGENTIC_RERANKER_TOP_N", 5)
CONFIDENCE_THRESHOLD = getattr(config, "AGENTIC_RERANKER_CONFIDENCE_THRESHOLD", 0.35)
N_VARIATIONS = getattr(config, "AGENTIC_QUERY_VARIATIONS", 4)
COVE_ENABLED = getattr(config, "AGENTIC_COVE_ENABLED", True)

NOT_FOUND_MSG = "Информация не найдена в доступных текстах законов."

_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        logger.info("[START] Reranker model initialization (BAAI/bge-reranker-v2-m3)")
        t0 = time.perf_counter()
        try:
            config.configure_hf_hub()
            from FlagEmbedding import FlagReranker
            _reranker_model = FlagReranker(
                getattr(config, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                use_fp16=True,
            )
            logger.info("[SUCCESS] Reranker initialized (%.2fs)", time.perf_counter() - t0)
        except Exception as e:
            logger.error("Reranker initialization failed: %s", e, exc_info=True)
            raise
    return _reranker_model


# ─── 1. Linguist-Analyst: HyDE + Query expansion (RU/KZ) ─────────────────────

def _linguist_hyde(query: str, trace_id: str) -> Tuple[str, dict]:
    """Generate hypothetical legal answer (HyDE) for better retrieval. Returns (hypothetical_doc, metrics)."""
    t0 = time.perf_counter()
    llm = rag_chain.get_llm()
    prompt = (
        "Ты — эксперт по законодательству РК. Дай краткий гипотетический ответ (2–3 предложения), "
        "как могла бы звучать формулировка из закона или судебной практики по вопросу. "
        "Пиши только суть, без вводных слов. Вопрос: {query}"
    )
    try:
        resp = llm.invoke(prompt.format(query=query))
        hyde_doc = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        hyde_doc = ""
        logger.error("[%s] HyDE (Groq/Ollama) failed: %s", trace_id, e, exc_info=True)
    ms = round((time.perf_counter() - t0) * 1000)
    return hyde_doc.strip() or "", {"linguist_hyde_ms": ms}


def _linguist_expand(query: str, trace_id: str) -> Tuple[List[str], dict]:
    """Generate 3–5 query variations in Russian and Kazakh. Returns (list of query strings, metrics)."""
    t0 = time.perf_counter()
    n = min(5, max(3, N_VARIATIONS))
    llm = rag_chain.get_llm()
    prompt = (
        f"Сгенерируй ровно {n} коротких перефразировок следующего юридического вопроса: "
        "половину на русском, половину на казахском. Каждый вариант — одна строка, без нумерации. "
        "Вопрос: {query}"
    )
    try:
        resp = llm.invoke(prompt.format(query=query))
        text = resp.content if hasattr(resp, "content") else str(resp)
        lines = [s.strip() for s in text.splitlines() if s.strip()][: n + 2]
    except Exception as e:
        lines = []
        logger.error("[%s] Query expansion (Groq/Ollama) failed: %s", trace_id, e, exc_info=True)
    ms = round((time.perf_counter() - t0) * 1000)
    queries = [query] + lines  # original + variations
    return queries, {"linguist_expand_ms": ms}


# ─── 2. Multi-query retrieval (HyDE + variations) → Top-50 ────────────────────

def _retrieve_candidates(queries: List[str], hyde_doc: str, trace_id: str) -> Tuple[List[Document], dict]:
    """Run vector search for each query (and HyDE doc if present). Merge and dedupe to up to TOP_K_CANDIDATES."""
    t0 = time.perf_counter()
    store = rag_chain.get_vector_store()
    embeddings = rag_chain.get_embeddings()
    # Pinecone filter (same as rag_chain)
    search_kwargs: dict = {"k": min(20, TOP_K_CANDIDATES // max(1, len(queries) + 1))}
    if hasattr(rag_chain, "_vector_kwargs") and rag_chain._vector_kwargs.get("filter"):
        search_kwargs["filter"] = rag_chain._vector_kwargs["filter"]

    seen_keys: set = set()
    merged: List[Document] = []
    search_queries = list(queries)
    if hyde_doc:
        # HyDE: use hypothetical answer as a "query" for embedding (passage-style for consistency with index)
        search_queries.append(hyde_doc)

    for q in search_queries:
        try:
            docs = store.similarity_search(q, **search_kwargs)
            for d in docs:
                key = (d.metadata.get("source"), d.metadata.get("article_number"), d.page_content[:100])
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(d)
                    if len(merged) >= TOP_K_CANDIDATES:
                        break
        except Exception as e:
            logger.error("[%s] Pinecone retrieval for query failed: %s", trace_id, e, exc_info=True)
        if len(merged) >= TOP_K_CANDIDATES:
            break

    ms = round((time.perf_counter() - t0) * 1000)
    return merged[:TOP_K_CANDIDATES], {"retrieval_candidates_ms": ms, "n_candidates": len(merged)}


# ─── 3. The Censor: Rerank with bge-reranker-v2-m3 → Top-5 + scores ──────────

def _censor_rerank(query: str, candidates: List[Document], trace_id: str) -> Tuple[List[Document], List[float], dict]:
    """Rerank candidates with BGE-M3; return top RERANKER_TOP_N docs and their scores."""
    t0 = time.perf_counter()
    if not candidates:
        return [], [], {"censor_rerank_ms": 0}
    model = _get_reranker()
    pairs = [[query, d.page_content] for d in candidates]
    scores = model.compute_score(pairs)
    if isinstance(scores, float):
        scores = [scores]
    for i, doc in enumerate(candidates):
        doc.metadata["relevance_score"] = scores[i] if i < len(scores) else 0.0
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in scored[:RERANKER_TOP_N]]
    top_scores = [s for _, s in scored[:RERANKER_TOP_N]]
    ms = round((time.perf_counter() - t0) * 1000)
    return top_docs, top_scores, {"censor_rerank_ms": ms}


# ─── 4. Self-RAG: Reject if confidence below threshold ───────────────────────

def _self_rag_gate(top_scores: List[float], trace_id: str) -> bool:
    """Return True if context is accepted (max score >= threshold)."""
    if not top_scores:
        return False
    best = max(top_scores)
    return best >= CONFIDENCE_THRESHOLD


# ─── 5. CoVe: Chain of Verification ──────────────────────────────────────────

def _extract_cited_articles(response: str) -> List[Tuple[str, str]]:
    """Extract (article_number, code_ru) from response text (e.g. 'статья 136 УК', 'ст. 122')."""
    cited = []
    # Patterns: статья 136, ст. 136, бап 5, Article 136; optionally followed by code name
    for m in re.finditer(
        r"(?:статья|ст\.|бап|Article)\s*(\d+[а-яА-Яa-zA-Z\-]?)\s*(?:УК|ГК|КоАП|РК|кодекс|закон)?",
        response,
        re.IGNORECASE,
    ):
        cited.append((m.group(1), ""))
    return cited[:10]


def _cove_verify(response: str, source_docs: List[Document], trace_id: str) -> Tuple[str, bool, dict]:
    """Verify that key statements in the response match the cited articles in context. Returns (verified_response, all_ok, metrics)."""
    t0 = time.perf_counter()
    if not COVE_ENABLED or not source_docs:
        return response, True, {"cove_ms": 0}
    cited = _extract_cited_articles(response)
    if not cited:
        return response, True, {"cove_ms": 0}
    # Build context snippet from source_docs for cited articles
    context_parts = []
    for d in source_docs[:5]:
        art = d.metadata.get("article_number", "")
        code = d.metadata.get("code_ru", "")
        context_parts.append(f"[{code} ст.{art}]\n{d.page_content[:800]}")
    context_str = "\n\n".join(context_parts)
    llm = rag_chain.get_llm()
    prompt = (
        "По законодательству РК. Контекст из НПА:\n{context}\n\n"
        "Ответ ассистента:\n{response}\n\n"
        "Вопрос: Соответствует ли ответ приведённому контексту? Ответь одним словом: ДА или НЕТ."
    )
    try:
        resp = llm.invoke(prompt.format(context=context_str, response=response[:2000]))
        ans = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
        all_ok = "НЕТ" not in ans[:10]
        if not all_ok:
            response = NOT_FOUND_MSG
    except Exception as e:
        logger.error("[%s] CoVe verification failed: %s", trace_id, e, exc_info=True)
        all_ok = True
    ms = round((time.perf_counter() - t0) * 1000)
    return response, all_ok, {"cove_ms": ms}


# ─── Main pipeline ───────────────────────────────────────────────────────────

def invoke_agentic_qa(
    query: str,
    history: Optional[List[dict]] = None,
    trace_id: Optional[str] = None,
) -> dict:
    """
    Board of Directors pipeline: Linguist (HyDE + expansion) → Retrieval → Censor (rerank 50→5)
    → Self-RAG gate → QA (if accepted) → CoVe. trace_id preserved in returned trace_report.
    """
    trace_id = trace_id or f"trace_{int(time.time())}"
    metrics: dict = {}
    t_total = time.perf_counter()

    # 1. Linguist-Analyst
    hyde_doc, m1 = _linguist_hyde(query, trace_id)
    metrics.update(m1)
    queries, m2 = _linguist_expand(query, trace_id)
    metrics.update(m2)

    # 2. Multi-query retrieval → candidates (up to 50)
    candidates, m3 = _retrieve_candidates(queries, hyde_doc, trace_id)
    metrics.update(m3)

    # 3. The Censor: rerank to top 5 with scores
    top_docs, top_scores, m4 = _censor_rerank(query, candidates, trace_id)
    metrics.update(m4)

    # 4. Self-RAG: reject if confidence too low
    if not _self_rag_gate(top_scores, trace_id):
        metrics["self_rag_rejected"] = True
        metrics["best_reranker_score"] = max(top_scores) if top_scores else 0.0
        return {
            "result": NOT_FOUND_MSG,
            "source_documents": [],
            "trace_report": {
                "metadata": {"id": trace_id},
                "metrics_ms": metrics,
                "agentic": {"self_rag_rejected": True},
            },
        }

    # Enrich with parent context (Chapter/Article breadcrumb)
    top_docs = rag_chain._enrich_with_parent_context(top_docs)
    # Trim length for LLM
    max_chars = getattr(config, "CONTEXT_MAX_CHARS_PER_DOC", 1800)
    trimmed = []
    for d in top_docs:
        content = d.page_content
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[...текст обрезан...]"
        trimmed.append(Document(page_content=content, metadata=d.metadata))
    top_docs = trimmed

    # 5. QA with top 5 context
    t_qa = time.perf_counter()
    qa_result = rag_chain.invoke_qa_with_context(query, top_docs, history=history)
    metrics["qa_ms"] = round((time.perf_counter() - t_qa) * 1000)
    result = qa_result.get("result", "")
    source_documents = qa_result.get("source_documents", [])

    # 6. CoVe verification
    result, cove_ok, m5 = _cove_verify(result, source_documents, trace_id)
    metrics.update(m5)
    if not cove_ok:
        metrics["cove_replaced"] = True

    metrics["python_rag_total"] = round((time.perf_counter() - t_total) * 1000)
    metrics["agentic"] = {
        "self_rag_rejected": False,
        "n_candidates": metrics.get("n_candidates", 0),
        "reranker_top_n": RERANKER_TOP_N,
        "best_reranker_score": max(top_scores) if top_scores else 0.0,
    }

    return {
        "result": result,
        "source_documents": source_documents,
        "trace_report": {
            "metadata": {"id": trace_id},
            "metrics_ms": metrics,
        },
    }
