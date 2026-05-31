# agentic_workflow.py — Board of Directors: Linguist-Analyst, Censor, Self-RAG, CoVe
# Trace_id is preserved through all modules for Go/Python boundary.

import asyncio
import json
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
CRAG_ENABLED = getattr(config, "AGENTIC_CRAG_ENABLED", True)
CRAG_MAX_ROUNDS = int(getattr(config, "AGENTIC_CRAG_MAX_ROUNDS", 2))
CRAG_TOP_K = int(getattr(config, "AGENTIC_CRAG_TOP_K", 10))
CRAG_REWRITE_TOP_K = int(getattr(config, "AGENTIC_CRAG_REWRITE_TOP_K", 20))
CRAG_MIN_CONTEXT_SCORE = float(getattr(config, "AGENTIC_CRAG_MIN_CONTEXT_SCORE", 0.58))
CRAG_LLM_EVALUATOR = getattr(config, "AGENTIC_CRAG_LLM_EVALUATOR", False)
CRAG_DECOMPOSE_QUERY = getattr(config, "AGENTIC_CRAG_DECOMPOSE_QUERY", True)

NOT_FOUND_MSG = "Информация не найдена в доступных текстах законов."

_reranker_model = None


def _escape_format_braces(text: str) -> str:
    return (text or "").replace("{", "{{").replace("}", "}}")


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
            logger.info(
                "[SUCCESS] Reranker initialized (%.2fs)", time.perf_counter() - t0
            )
        except Exception as e:
            logger.error("Reranker initialization failed: %s", e, exc_info=True)
            raise
    return _reranker_model


def _doc_signature(doc: Document) -> tuple[str, str, str]:
    meta = doc.metadata or {}
    return (
        str(meta.get("code_ru") or "").strip(),
        str(meta.get("article_number") or "").strip(),
        str(meta.get("revision_date") or "").strip(),
    )


def _normalize_query_terms(query: str) -> list[str]:
    text = (query or "").lower()
    terms = [t for t in re.findall(r"[a-zа-яёқіңғүұһәө0-9\-]+", text) if len(t) > 2]
    return list(dict.fromkeys(terms))


def _extract_date_like_terms(query: str) -> list[str]:
    q = (query or "").lower()
    matches = re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", q)
    matches.extend(re.findall(r"\b20\d{2}\b", q))
    return list(dict.fromkeys(matches))


def _is_compound_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in (" и ", " или ", ";", "\n", "заодно", "а также", "plus"))


def _evaluate_context_quality(query: str, top_docs: List[Document], top_scores: List[float]) -> dict[str, Any]:
    """Deterministic CRAG evaluator for legal retrieval quality."""
    if not top_docs:
        return {
            "action": "rewrite",
            "context_score": 0.0,
            "missing_reasons": ["no_documents"],
            "contradictions": [],
            "need_more": True,
        }

    target_codes = set(rag_chain._detect_target_codes(query))
    query_articles = set(rag_chain._focus_articles_from_query(query))
    query_terms = _normalize_query_terms(query)
    date_terms = _extract_date_like_terms(query)
    best_score = max(top_scores) if top_scores else 0.0

    code_hits = 0
    article_hits = 0
    revision_hits = 0
    status_hits = 0
    source_codes: list[str] = []
    revision_dates: set[str] = set()
    noise_docs = 0

    for doc in top_docs[:CRAG_TOP_K]:
        meta = doc.metadata or {}
        code_ru = str(meta.get("code_ru") or "").strip()
        article_number = str(meta.get("article_number") or "").strip()
        revision_date = str(meta.get("revision_date") or "").strip()
        status = str(meta.get("status") or "").strip().lower()
        content = f"{code_ru} {meta.get('article_title', '')} {meta.get('chapter_title', '')} {doc.page_content[:900]}".lower()

        if code_ru:
            source_codes.append(code_ru)
        if code_ru and target_codes and code_ru in target_codes:
            code_hits += 1
        if article_number and article_number in query_articles:
            article_hits += 1
        if revision_date:
            revision_hits += 1
            revision_dates.add(revision_date)
        if status:
            status_hits += 1
        if rag_chain._is_noisy_legal_chunk(doc):
            noise_docs += 1

    contradictions: list[str] = []
    if len(revision_dates) > 1 and date_terms:
        contradictions.append("multiple_revision_dates")
    if len(set(source_codes)) > 2 and target_codes:
        contradictions.append("mixed_target_codes")

    missing_reasons: list[str] = []
    if target_codes and code_hits == 0:
        missing_reasons.append("target_code_missing")
    if query_articles and article_hits == 0:
        missing_reasons.append("target_article_missing")
    if date_terms and revision_hits == 0:
        missing_reasons.append("missing_revision_date")
    if any(token in (query or "").lower() for token in ("действует", "утратил", "status", "статус")) and status_hits == 0:
        missing_reasons.append("missing_status")
    if noise_docs >= max(2, len(top_docs) // 2):
        missing_reasons.append("too_much_noise")

    coverage = 0.0
    if target_codes:
        coverage += 0.35 if code_hits else 0.0
    if query_articles:
        coverage += 0.25 if article_hits else 0.0
    if date_terms:
        coverage += 0.15 if revision_hits else 0.0
    coverage += 0.10 if status_hits else 0.0
    coverage += 0.15 if best_score >= CONFIDENCE_THRESHOLD else 0.0
    coverage -= 0.15 if contradictions else 0.0
    coverage -= 0.10 if noise_docs else 0.0
    if coverage < 0.0:
        coverage = 0.0
    if coverage > 1.0:
        coverage = 1.0

    if not missing_reasons and not contradictions and coverage >= CRAG_MIN_CONTEXT_SCORE:
        action = "accept"
    elif contradictions:
        action = "search_more"
    elif _is_compound_query(query):
        action = "decompose"
    else:
        action = "rewrite"

    return {
        "action": action,
        "context_score": round(coverage, 2),
        "missing_reasons": missing_reasons,
        "contradictions": contradictions,
        "need_more": action != "accept",
    }


def _build_corrective_queries(
    query: str,
    top_docs: List[Document],
    evaluator: dict[str, Any],
    trace_id: str,
) -> list[str]:
    variants: list[str] = []
    docs_by_code: dict[str, list[Document]] = {}
    for doc in top_docs:
        code_ru = str((doc.metadata or {}).get("code_ru") or "").strip()
        if code_ru:
            docs_by_code.setdefault(code_ru, []).append(doc)

    # Query rewrite by legal signal gaps
    if "target_code_missing" in evaluator.get("missing_reasons", []):
        variants.append(rag_chain._rewrite_query_for_retrieval(query))
    if "target_article_missing" in evaluator.get("missing_reasons", []):
        variants.append(rag_chain._rewrite_query_for_retrieval(f"{query} статья / бап / article"))
    if "missing_revision_date" in evaluator.get("missing_reasons", []):
        variants.append(rag_chain._rewrite_query_for_retrieval(f"{query} действующая редакция дата"))
    if "missing_status" in evaluator.get("missing_reasons", []):
        variants.append(rag_chain._rewrite_query_for_retrieval(f"{query} действует утратил силу"))

    # Decompose compound requests into subqueries.
    if evaluator.get("action") == "decompose" and CRAG_DECOMPOSE_QUERY:
        parts = [p.strip(" .;") for p in re.split(r"\s+(?:и|или|а также)\s+|;", query) if len(p.strip(" .;")) > 8]
        variants.extend(parts[:4])

    # If we have candidate codes, narrow by the most relevant legal source.
    if docs_by_code:
        ranked_codes = sorted(
            docs_by_code.items(),
            key=lambda item: (
                -sum(1 for d in item[1] if _doc_signature(d)[1] and _doc_signature(d)[1] in rag_chain._focus_articles_from_query(query)),
                -sum(1 for d in item[1] if _doc_signature(d)[2]),
                item[0],
            ),
        )
        for code_ru, _ in ranked_codes[:2]:
            variants.append(f"{query} {code_ru}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for item in [query, *variants]:
        normalized = " ".join(str(item or "").split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    logger.info("[%s] CRAG corrective queries: %s", trace_id, out[:6])
    return out[:6]


async def _corrective_retrieve(
    query: str,
    history: Optional[List[dict]],
    trace_id: str,
    base_queries: list[str],
    hyde_doc: str,
) -> tuple[list[Document], list[float], dict]:
    """Run CRAG loop: evaluate retrieved context, then rewrite/decompose/search more."""
    metrics: dict[str, Any] = {"crag_enabled": True, "crag_rounds": 0, "crag_last_action": "rewrite"}
    candidates, candidate_scores, base_metrics = await _retrieve_candidates(base_queries, hyde_doc, trace_id)
    metrics.update(base_metrics)

    for round_idx in range(int(CRAG_MAX_ROUNDS)):
        top_docs, top_scores, rerank_metrics = await _censor_rerank(query, candidates, candidate_scores, trace_id)
        metrics.update(rerank_metrics)
        evaluator = _evaluate_context_quality(query, top_docs, top_scores)
        metrics[f"crag_round_{round_idx + 1}"] = evaluator
        metrics["crag_last_action"] = evaluator.get("action", "rewrite")
        metrics["crag_context_score"] = evaluator.get("context_score", 0.0)
        metrics["crag_missing_reasons"] = evaluator.get("missing_reasons", [])
        metrics["crag_contradictions"] = evaluator.get("contradictions", [])
        if evaluator["action"] == "accept":
            metrics["crag_rounds"] = round_idx + 1
            return top_docs, top_scores, metrics

        corrective_queries = _build_corrective_queries(query, top_docs, evaluator, trace_id)
        if len(corrective_queries) <= 1:
            metrics["crag_rounds"] = round_idx + 1
            break

        # Additional retrieval with larger k on rewritten / decomposed queries.
        search_queries = corrective_queries[1:]
        additional_candidates, additional_scores, add_metrics = await _retrieve_candidates(
            search_queries, hyde_doc, trace_id
        )
        metrics.setdefault("crag_additional_retrieval_ms", 0)
        metrics["crag_additional_retrieval_ms"] += add_metrics.get("retrieval_candidates_ms", 0)
        candidates.extend(additional_candidates)
        candidate_scores.extend(additional_scores)

        # Dedupe candidates and keep the strongest score for each doc.
        merged: dict[tuple[str, str], tuple[Document, float]] = {}
        pair_count = min(len(candidates), len(candidate_scores))
        for idx in range(pair_count):
            doc = candidates[idx]
            score = candidate_scores[idx]
            key = (
                str((doc.metadata or {}).get("source") or ""),
                str((doc.metadata or {}).get("article_number") or ""),
                doc.page_content[:120],
            )
            prev = merged.get(key)
            if prev is None or score > prev[1]:
                merged[key] = (doc, score)
        if merged:
            candidates = [doc for doc, _ in merged.values()]
            candidate_scores = [score for _, score in merged.values()]
        candidates = candidates[:TOP_K_CANDIDATES]
        candidate_scores = candidate_scores[:TOP_K_CANDIDATES]

    top_docs, top_scores, rerank_metrics = await _censor_rerank(query, candidates, candidate_scores, trace_id)
    metrics.update(rerank_metrics)
    metrics["crag_rounds"] = int(metrics.get("crag_rounds", 0)) + 1
    return top_docs, top_scores, metrics


# ─── 1. Linguist-Analyst: HyDE + Query expansion (RU/KZ) ─────────────────────


async def _linguist_hyde(query: str, trace_id: str) -> Tuple[str, dict]:
    """Generate hypothetical legal answer (HyDE) for better retrieval. Returns (hypothetical_doc, metrics)."""
    t0 = time.perf_counter()
    llm = rag_chain.get_llm()
    prompt = (
        "Ты — эксперт по законодательству РК. Дай краткий гипотетический ответ (2–3 предложения), "
        "как могла бы звучать формулировка из закона или судебной практики по вопросу. "
        'Пиши только JSON: {{"hypothesis": "твой ответ здесь"}}. Вопрос: {query}'
    )
    try:
        resp = await asyncio.to_thread(llm.invoke, prompt.format(query=query))
        text = resp.content if hasattr(resp, "content") else str(resp)
        # Robust JSON extraction: look for {...} in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                data = json.loads(json_text)
                hyde_doc = data.get("hypothesis", "").strip()
            except json.JSONDecodeError:
                # Fallback: if { } present but not valid JSON, try to extract value manually
                # or just use the whole text if it's short
                hyde_doc = text.strip() if len(text) < 300 else ""
        else:
            # Fallback: no { } found, use the whole response if it looks reasonable
            hyde_doc = text.strip() if len(text) < 300 else ""
    except Exception as e:
        hyde_doc = ""
        logger.error("[%s] HyDE (Groq/Ollama) failed: %s", trace_id, e, exc_info=True)
    ms = round((time.perf_counter() - t0) * 1000)
    return hyde_doc.strip() or "", {"linguist_hyde_ms": ms}


async def _linguist_expand(query: str, trace_id: str) -> Tuple[List[str], dict]:
    """Generate 3–5 query variations in Russian and Kazakh. Returns (list of query strings, metrics)."""
    t0 = time.perf_counter()
    n = min(5, max(3, N_VARIATIONS))
    llm = rag_chain.get_llm()
    prompt = (
        f"Сгенерируй ровно {n} коротких перефразировок следующего юридического вопроса: "
        "половину на русском, половину на казахском. Каждый вариант — одна строка, без нумерации. "
        "Respond with JSON: {{'variations': ['вар1', 'вар2', ...]}} Вопрос: {query}"
    )
    try:
        resp = await asyncio.to_thread(llm.invoke, prompt.format(query=query))
        text = resp.content if hasattr(resp, "content") else str(resp)
        # Robust JSON extraction
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                data = json.loads(json_text)
                lines = data.get("variations", [])
            except json.JSONDecodeError:
                # If JSON fails, try to split by lines as a secondary fallback
                lines = [
                    l.strip()
                    for l in text.split("\n")
                    if l.strip() and "{" not in l and "}" not in l
                ]
        else:
            # Fallback for plain text variations
            lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Filter and limit
        lines = [l for l in lines if l and l != query][:n]
    except Exception as e:
        lines = []
        logger.error(
            "[%s] Query expansion (Groq/Ollama) failed: %s", trace_id, e, exc_info=True
        )
    ms = round((time.perf_counter() - t0) * 1000)
    queries = [query] + lines  # original + variations
    return queries, {"linguist_expand_ms": ms}


# ─── 2. Multi-query retrieval (HyDE + variations) → Top-50 ────────────────────


async def _retrieve_candidates(
    queries: List[str], hyde_doc: str, trace_id: str
) -> Tuple[List[Document], dict]:
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
    scores: List[float] = []
    search_queries = list(queries)
    if hyde_doc:
        # HyDE: use hypothetical answer as a "query" for embedding (passage-style for consistency with index)
        search_queries.append(hyde_doc)

    if search_queries:
        search_tasks = [
            asyncio.to_thread(store.similarity_search_with_score, q, **search_kwargs)
            for q in search_queries
        ]
        search_results = await asyncio.gather(*search_tasks)
        for docs_with_scores in search_results:
            for d, score in docs_with_scores:
                key = (
                    d.metadata.get("source"),
                    d.metadata.get("article_number"),
                    d.page_content[:100],
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(d)
                    scores.append(score)
                    if len(merged) >= TOP_K_CANDIDATES:
                        break
            if len(merged) >= TOP_K_CANDIDATES:
                break

    ms = round((time.perf_counter() - t0) * 1000)
    return (
        merged[:TOP_K_CANDIDATES],
        scores[:TOP_K_CANDIDATES],
        {"retrieval_candidates_ms": ms, "n_candidates": len(merged)},
    )


# ─── 3. The Censor: Rerank with bge-reranker-v2-m3 → Top-5 + scores ──────────


async def _censor_rerank(
    query: str,
    candidates: List[Document],
    candidate_scores: Optional[List[float]],
    trace_id: str,
) -> Tuple[List[Document], List[float], dict]:
    """Rerank candidates with BGE-M3; return top RERANKER_TOP_N docs and their scores."""
    t0 = time.perf_counter()
    if not candidates:
        return [], [], {"censor_rerank_ms": 0}

    # Early exit: if top Pinecone score > 0.82, skip reranker
    if candidate_scores and max(candidate_scores) > 0.82:
        sorted_indices = sorted(
            range(len(candidates)), key=lambda i: candidate_scores[i], reverse=True
        )
        top_docs = [candidates[i] for i in sorted_indices[:RERANKER_TOP_N]]
        top_scores = [candidate_scores[i] for i in sorted_indices[:RERANKER_TOP_N]]
        return top_docs, top_scores, {"censor_rerank_ms": 0}

    # Limit reranker to top 10 if many candidates
    rerank_candidates = candidates[:10] if len(candidates) > 10 else candidates

    model = _get_reranker()
    pairs = [[query, d.page_content] for d in rerank_candidates]
    scores = await asyncio.to_thread(model.compute_score, pairs)
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


async def _cove_verify(
    response: str, source_docs: List[Document], trace_id: str
) -> Tuple[str, bool, dict]:
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
        safe_context = _escape_format_braces(context_str)
        safe_response = _escape_format_braces(response[:2000])
        resp = await asyncio.to_thread(
            llm.invoke, prompt.format(context=safe_context, response=safe_response)
        )
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


async def invoke_agentic_qa(
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

    # 1. Linguist: Query expansion and HyDE
    queries, m_ling_expand = await _linguist_expand(query, trace_id)
    hyde_doc, m_ling_hyde = await _linguist_hyde(query, trace_id)
    metrics.update(m_ling_expand)
    metrics.update(m_ling_hyde)

    # 2. Retrieval: CRAG loop when enabled, otherwise classic multi-query retrieval.
    if CRAG_ENABLED:
        candidates, candidate_scores, m3 = await _corrective_retrieve(
            query,
            history,
            trace_id,
            queries,
            hyde_doc,
        )
    else:
        candidates, candidate_scores, m3 = await _retrieve_candidates([query], "", trace_id)
        # Additional retrieval with expanded queries and HyDE
        if len(queries) > 1 or hyde_doc:
            additional_queries = queries[1:] if len(queries) > 1 else []
            if hyde_doc:
                additional_queries.append(hyde_doc)
            if additional_queries:
                additional_candidates, additional_scores, m3_add = (
                    await _retrieve_candidates(additional_queries, "", trace_id)
                )
                candidates.extend(additional_candidates)
                candidate_scores.extend(additional_scores)
                # Dedupe
                seen = {}
                for i, d in enumerate(candidates):
                    key = (d.metadata.get("source"), d.page_content[:100])
                    if key not in seen:
                        seen[key] = (d, candidate_scores[i])
                merged = list(seen.values())
                if merged:
                    candidates, candidate_scores = zip(*merged)
                    candidates = list(candidates)
                    candidate_scores = list(candidate_scores)
                else:
                    candidates, candidate_scores = [], []
                candidates = candidates[:TOP_K_CANDIDATES]
                candidate_scores = candidate_scores[:TOP_K_CANDIDATES]
                m3["retrieval_candidates_ms"] += m3_add.get("retrieval_candidates_ms", 0)
    metrics.update(m3)

    # 3. The Censor: rerank to top 5 with scores
    top_docs, top_scores, m4 = await _censor_rerank(
        query, candidates, candidate_scores, trace_id
    )
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
    qa_result = await asyncio.to_thread(
        rag_chain.invoke_qa_with_context, query, top_docs, history=history
    )
    metrics["qa_ms"] = round((time.perf_counter() - t_qa) * 1000)
    result = qa_result.get("result", "")
    source_documents = qa_result.get("source_documents", [])

    # 6. CoVe verification
    result, cove_ok, m5 = await _cove_verify(result, source_documents, trace_id)
    metrics.update(m5)
    if not cove_ok:
        metrics["cove_replaced"] = True

    metrics["python_rag_total"] = round((time.perf_counter() - t_total) * 1000)
    metrics["agentic"] = {
        "self_rag_rejected": False,
        "n_candidates": metrics.get("n_candidates", 0),
        "reranker_top_n": RERANKER_TOP_N,
        "best_reranker_score": max(top_scores) if top_scores else 0.0,
        "crag_enabled": bool(CRAG_ENABLED),
        "crag_rounds": int(metrics.get("crag_rounds", 0)),
        "crag_last_action": metrics.get("crag_last_action"),
        "crag_context_score": metrics.get("crag_context_score"),
        "crag_missing_reasons": metrics.get("crag_missing_reasons", []),
        "crag_contradictions": metrics.get("crag_contradictions", []),
    }

    return {
        "result": result,
        "source_documents": source_documents,
        "trace_report": {
            "metadata": {"id": trace_id},
            "metrics_ms": metrics,
        },
    }
